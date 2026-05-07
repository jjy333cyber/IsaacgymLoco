# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot_wtw import LeggedRobotwtw
from legged_gym.envs.cc1.cc1_jump_wtw_config import Cc1JumpwtwCfg, Cc1JumpwtwCfgPPO


class Cc1JumpHighwtw(LeggedRobotwtw):
    """Task-local reward/reference extensions for cc1_jump_high_wtw only."""

    def _jump_phase_masks(self):
        phase = torch.remainder(self.gait_indices, 1.0)

        compress_end = getattr(self.cfg.rewards, "jump_phase_compress_end", 0.15)
        push_end = getattr(self.cfg.rewards, "jump_phase_push_end", 0.35)
        flight_end = getattr(self.cfg.rewards, "jump_phase_flight_end", 0.70)
        land_end = getattr(self.cfg.rewards, "jump_phase_land_end", 0.85)
        flight_mid = push_end + 0.5 * max(flight_end - push_end, 1e-6)

        compress = phase < compress_end
        push = (phase >= compress_end) & (phase < push_end)
        flight_up = (phase >= push_end) & (phase < flight_mid)
        flight_down = (phase >= flight_mid) & (phase < flight_end)
        landing = (phase >= flight_end) & (phase < land_end)
        recovery = phase >= land_end
        flight = flight_up | flight_down
        return phase, compress, push, flight_up, flight_down, landing, recovery, flight

    def _reward_jump_mixed_contact(self):
        contact_count = torch.sum(self._jump_contact_mask().float(), dim=1)
        mixed_contact = (contact_count > 0.0) & (contact_count < len(self.feet_indices))
        return mixed_contact.float() * self._jump_motion_mask().float()

    def _reward_jump_flight_phase_air(self):
        _, _, _, _, _, _, _, flight = self._jump_phase_masks()
        return self._jump_all_air_mask().float() * flight.float() * self._jump_motion_mask().float()

    def _reward_jump_takeoff_z_vel(self):
        _, _, push, _, _, _, _, _ = self._jump_phase_masks()
        start_vel = getattr(self.cfg.rewards, "jump_takeoff_z_vel_min", getattr(self.cfg.rewards, "jump_z_vel_min", 0.0))
        target_vel = getattr(self.cfg.rewards, "jump_takeoff_z_vel_target", getattr(self.cfg.rewards, "jump_z_vel_target", start_vel + 0.5))
        vel_span = max(target_vel - start_vel, 1e-6)
        vel_reward = torch.clamp((self.base_lin_vel[:, 2] - start_vel) / vel_span, min=0.0, max=1.0)

        if getattr(self.cfg.rewards, "jump_takeoff_require_contact", True):
            contact_weight = torch.mean(self._jump_contact_mask().float(), dim=1)
        else:
            contact_weight = torch.ones(self.num_envs, device=self.device)
        return vel_reward * push.float() * contact_weight * self._jump_motion_mask().float()

    def _get_jump_dof_reference(self):
        _, compress, push, flight_up, flight_down, landing, recovery, _ = self._jump_phase_masks()
        if self.default_dof_pos.shape[0] == self.num_envs:
            ref_dof_pos = self.default_dof_pos.clone()
        else:
            ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1).clone()

        hip_y_ids = [1, 4, 7, 10]
        knee_ids = [2, 5, 8, 11]

        def apply_pose(mask, hip_y, knee):
            mask = mask.unsqueeze(1)
            ref_dof_pos[:, hip_y_ids] = torch.where(
                mask,
                torch.ones_like(ref_dof_pos[:, hip_y_ids]) * hip_y,
                ref_dof_pos[:, hip_y_ids],
            )
            ref_dof_pos[:, knee_ids] = torch.where(
                mask,
                torch.ones_like(ref_dof_pos[:, knee_ids]) * knee,
                ref_dof_pos[:, knee_ids],
            )

        apply_pose(
            compress,
            getattr(self.cfg.rewards, "jump_ref_hip_y_compress", -1.05),
            getattr(self.cfg.rewards, "jump_ref_knee_compress", 2.0),
        )
        apply_pose(
            push,
            getattr(self.cfg.rewards, "jump_ref_hip_y_push", -0.55),
            getattr(self.cfg.rewards, "jump_ref_knee_push", 1.15),
        )
        apply_pose(
            flight_up | flight_down,
            getattr(self.cfg.rewards, "jump_ref_hip_y_flight", -1.10),
            getattr(self.cfg.rewards, "jump_ref_knee_flight", 2.05),
        )
        apply_pose(
            landing,
            getattr(self.cfg.rewards, "jump_ref_hip_y_land", -0.75),
            getattr(self.cfg.rewards, "jump_ref_knee_land", 1.55),
        )
        apply_pose(
            recovery,
            getattr(self.cfg.rewards, "jump_ref_hip_y_recovery", -0.80),
            getattr(self.cfg.rewards, "jump_ref_knee_recovery", 1.60),
        )

        return ref_dof_pos

    def _reward_jump_ref_dof_pos(self):
        ref_dof_pos = self._get_jump_dof_reference()
        err = torch.mean(torch.square(self.dof_pos - ref_dof_pos), dim=1)
        sigma = max(getattr(self.cfg.rewards, "jump_ref_sigma_dof", 0.30), 1e-6)
        return torch.exp(-err / sigma) * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_landing_force_balance(self):
        _, _, _, _, _, landing, _, _ = self._jump_phase_masks()
        foot_forces = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)
        mean_force = torch.mean(foot_forces, dim=1, keepdim=True)
        max_force = max(getattr(self.cfg.rewards, "jump_landing_balance_force_norm", 120.0), 1e-6)
        imbalance = torch.mean(torch.square((foot_forces - mean_force) / max_force), dim=1)
        return imbalance * landing.float() * self._jump_motion_mask().float()

    def _reward_jump_hipx_landing(self):
        _, _, _, _, _, landing, recovery, _ = self._jump_phase_masks()
        if self.default_dof_pos.shape[0] == self.num_envs:
            default_dof_pos = self.default_dof_pos
        else:
            default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        hipx_ids = getattr(self.cfg.rewards, "jump_hipx_landing_ids", [0, 3, 6, 9])
        hipx_error = torch.mean(torch.square(self.dof_pos[:, hipx_ids] - default_dof_pos[:, hipx_ids]), dim=1)
        active = landing | recovery
        return hipx_error * active.float() * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_leg_symmetry(self):
        """Keep the four legs moving as one symmetric pronk group without forcing a specific tucked pose."""
        hipx_ids = [0, 3, 6, 9]
        hip_y_ids = [1, 4, 7, 10]
        knee_ids = [2, 5, 8, 11]

        if self.default_dof_pos.shape[0] == self.num_envs:
            default_dof_pos = self.default_dof_pos
        else:
            default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        hipx_error = torch.mean(torch.square(self.dof_pos[:, hipx_ids] - default_dof_pos[:, hipx_ids]), dim=1)
        hip_y_variance = torch.var(self.dof_pos[:, hip_y_ids], dim=1)
        knee_variance = torch.var(self.dof_pos[:, knee_ids], dim=1)
        return (hipx_error + hip_y_variance + knee_variance) * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _feet_pos_body_frame(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros_like(cur_footpos_translated)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        return footpos_in_body_frame

    def _get_jump_foot_pos_reference(self):
        phase, compress, push, flight_up, flight_down, landing, recovery, _ = self._jump_phase_masks()

        compress_end = getattr(self.cfg.rewards, "jump_phase_compress_end", 0.15)
        push_end = getattr(self.cfg.rewards, "jump_phase_push_end", 0.35)
        flight_end = getattr(self.cfg.rewards, "jump_phase_flight_end", 0.70)
        land_end = getattr(self.cfg.rewards, "jump_phase_land_end", 0.85)
        flight_mid = push_end + 0.5 * max(flight_end - push_end, 1e-6)

        x_front = getattr(self.cfg.rewards, "jump_ref_foot_x_front", 0.185)
        x_rear = getattr(self.cfg.rewards, "jump_ref_foot_x_rear", -0.185)
        y_left = getattr(self.cfg.rewards, "jump_ref_foot_y_left", 0.16)
        y_right = getattr(self.cfg.rewards, "jump_ref_foot_y_right", -0.16)
        z_stance = getattr(self.cfg.rewards, "jump_ref_foot_z_stance", -0.40)

        # Order follows CC1 body order: FL, FR, HL, HR.
        ref_pos = torch.tensor(
            [[x_front, y_left, z_stance],
             [x_front, y_right, z_stance],
             [x_rear, y_left, z_stance],
             [x_rear, y_right, z_stance]],
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)

        sweep_back = getattr(self.cfg.rewards, "jump_ref_foot_x_sweep_back", -0.035)
        sweep_forward = getattr(self.cfg.rewards, "jump_ref_foot_x_sweep_forward", 0.035)
        land_offset = getattr(self.cfg.rewards, "jump_ref_foot_x_land_offset", 0.010)

        x_offset = torch.zeros_like(phase)
        x_offset = torch.where(compress, self._jump_phase_lerp(phase, 0.0, compress_end, 0.0, 0.5 * sweep_back), x_offset)
        x_offset = torch.where(push, self._jump_phase_lerp(phase, compress_end, push_end, 0.5 * sweep_back, sweep_back), x_offset)
        x_offset = torch.where(flight_up, self._jump_phase_lerp(phase, push_end, flight_mid, sweep_back, sweep_forward), x_offset)
        x_offset = torch.where(flight_down, self._jump_phase_lerp(phase, flight_mid, flight_end, sweep_forward, land_offset), x_offset)
        x_offset = torch.where(landing, self._jump_phase_lerp(phase, flight_end, land_end, land_offset, 0.0), x_offset)
        x_offset = torch.where(recovery, 0.0 * phase, x_offset)
        ref_pos[:, :, 0] += x_offset.unsqueeze(1)

        z_flight = getattr(self.cfg.rewards, "jump_ref_foot_z_flight", z_stance)
        z_land = getattr(self.cfg.rewards, "jump_ref_foot_z_land", z_stance)
        ref_z = torch.ones_like(phase) * z_stance
        ref_z = torch.where(flight_up, self._jump_phase_lerp(phase, push_end, flight_mid, z_stance, z_flight), ref_z)
        ref_z = torch.where(flight_down, self._jump_phase_lerp(phase, flight_mid, flight_end, z_flight, z_land), ref_z)
        ref_z = torch.where(landing, self._jump_phase_lerp(phase, flight_end, land_end, z_land, z_stance), ref_z)
        ref_pos[:, :, 2] = ref_z.unsqueeze(1)

        active = flight_up | flight_down | landing
        return ref_pos, active

    def _reward_jump_ref_foot_pos(self):
        footpos_in_body_frame = self._feet_pos_body_frame()
        ref_pos, active = self._get_jump_foot_pos_reference()

        weights = torch.tensor(
            [getattr(self.cfg.rewards, "jump_ref_foot_pos_weight_x", 0.35),
             getattr(self.cfg.rewards, "jump_ref_foot_pos_weight_y", 1.0),
             getattr(self.cfg.rewards, "jump_ref_foot_pos_weight_z", 0.0)],
            device=self.device,
        )
        err = torch.sum(torch.square(footpos_in_body_frame - ref_pos) * weights.view(1, 1, 3), dim=(1, 2))
        err = err / torch.clamp(torch.sum(weights) * len(self.feet_indices), min=1e-6)
        sigma = max(getattr(self.cfg.rewards, "jump_ref_sigma_foot_pos", 0.015), 1e-6)
        return torch.exp(-err / sigma) * active.float() * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _get_jump_reference(self):
        phase, compress, push, flight_up, flight_down, landing, recovery, flight = self._jump_phase_masks()

        compress_end = getattr(self.cfg.rewards, "jump_phase_compress_end", 0.15)
        push_end = getattr(self.cfg.rewards, "jump_phase_push_end", 0.35)
        flight_end = getattr(self.cfg.rewards, "jump_phase_flight_end", 0.70)
        land_end = getattr(self.cfg.rewards, "jump_phase_land_end", 0.85)
        flight_mid = push_end + 0.5 * max(flight_end - push_end, 1e-6)

        h_stance = getattr(self.cfg.rewards, "jump_ref_height_stance", self.cfg.rewards.base_height_target)
        h_compress = getattr(self.cfg.rewards, "jump_ref_height_compress", h_stance)
        h_takeoff = getattr(self.cfg.rewards, "jump_ref_height_takeoff", h_stance)
        h_apex = getattr(self.cfg.rewards, "jump_ref_height_apex", h_takeoff)
        h_land = getattr(self.cfg.rewards, "jump_ref_height_land", h_stance)

        foot_stance = getattr(self.cfg.rewards, "jump_ref_foot_height_stance", 0.02)
        foot_flight = getattr(self.cfg.rewards, "jump_ref_foot_height_flight", self.cfg.rewards.target_foot_height)

        ref_height = torch.ones_like(phase) * h_stance
        ref_z_vel = torch.zeros_like(phase)
        ref_foot_height = torch.ones_like(phase) * foot_stance
        ref_contact = torch.ones(self.num_envs, len(self.feet_indices), device=self.device)

        ref_height = torch.where(compress, self._jump_phase_lerp(phase, 0.0, compress_end, h_stance, h_compress), ref_height)
        ref_height = torch.where(push, self._jump_phase_lerp(phase, compress_end, push_end, h_compress, h_takeoff), ref_height)
        ref_height = torch.where(flight_up, self._jump_phase_lerp(phase, push_end, flight_mid, h_takeoff, h_apex), ref_height)
        ref_height = torch.where(flight_down, self._jump_phase_lerp(phase, flight_mid, flight_end, h_apex, h_land), ref_height)
        ref_height = torch.where(landing, self._jump_phase_lerp(phase, flight_end, land_end, h_land, h_stance), ref_height)
        ref_height = torch.where(recovery, h_stance + 0.0 * phase, ref_height)

        ref_z_vel = torch.where(compress, torch.ones_like(phase) * getattr(self.cfg.rewards, "jump_ref_z_vel_compress", -0.2), ref_z_vel)
        ref_z_vel = torch.where(push, torch.ones_like(phase) * getattr(self.cfg.rewards, "jump_ref_z_vel_push", 1.0), ref_z_vel)
        flight_z_vel = getattr(self.cfg.rewards, "jump_ref_z_vel_flight", 0.0)
        flight_up_z_vel = getattr(self.cfg.rewards, "jump_ref_z_vel_flight_up", flight_z_vel)
        flight_down_z_vel = getattr(self.cfg.rewards, "jump_ref_z_vel_flight_down", flight_z_vel)
        ref_z_vel = torch.where(flight_up, torch.ones_like(phase) * flight_up_z_vel, ref_z_vel)
        ref_z_vel = torch.where(flight_down, torch.ones_like(phase) * flight_down_z_vel, ref_z_vel)
        ref_z_vel = torch.where(landing, torch.ones_like(phase) * getattr(self.cfg.rewards, "jump_ref_z_vel_land", -0.2), ref_z_vel)

        ref_foot_height = torch.where(flight, torch.ones_like(phase) * foot_flight, ref_foot_height)
        ref_contact[flight, :] = 0.0

        return ref_height, ref_z_vel, ref_foot_height, ref_contact


class Cc1JumpHighwtwCfg(Cc1JumpwtwCfg):
    """Stage-B pronk-like jump task: synchronized four-leg hopping with stable landing."""

    class commands(Cc1JumpwtwCfg.commands):
        # Keep the first pronk stage narrow: forward synchronized hopping before lateral/yaw agility.
        max_forward_curriculum = 2.0
        max_backward_curriculum = 0.0
        max_lat_curriculum = 0.0
        stand_still_command = False
        Rotate_command = False

        # Stay close to cc1_jump_wtw: the old synchronized gait was already stable.
        frequencies = 3.0
        phases = 0.0
        offsets = 0.0
        bounds = 0.0
        durations = 0.5

        class ranges(Cc1JumpwtwCfg.commands.ranges):
            lin_vel_x = [0.5, 1.4]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(Cc1JumpwtwCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 2.0]

        randomize_com_displacement = True
        com_displacement_range = dict(
            x=[-0.07, 0.07],
            y=[-0.07, 0.07],
            z=[-0.07, 0.07],
        )

    class rewards(Cc1JumpwtwCfg.rewards):
        class scales(Cc1JumpwtwCfg.rewards.scales):
            termination = -2.0
            tracking_lin_vel = 3.5
            tracking_ang_vel = 0.5

            # Keep the original jump style stable, but loosen vertical penalties enough to jump higher.
            lin_vel_z = -0.05
            ang_vel_xy = -0.10
            orientation = -3.0
            base_height = -6.0

            torques = -5e-5
            action_rate = -0.01
            smoothness = -0.005
            dof_acc = -2.5e-7
            stand_still = 0.0
            joint_power = -1e-5
            power_distribution = -5e-6
            collision = -0.5
            feet_contact_forces = -0.0001
            feet_slide = -0.1
            feet_soft_landing = -1.0  # 首次触地过重惩罚：降低落脚声和砸地感
            feet_air_time_variance_velocity = -8.0

            # Re-enable the original WTW contact structure; it prevents the floating-looking failure mode.
            raibert_heuristic = -5.0
            tracking_contacts_shaped_force = 1.0
            tracking_contacts_shaped_vel = 1.0
            feet_clearance_cmd_linear = -15.0

            jump = 3.0
            jump_air_time = 0.0
            jump_flight_phase_air = 0.0
            jump_mixed_contact = -5.0
            jump_height = 3.0
            jump_z_vel = 0.0
            jump_takeoff_z_vel = 2.5
            jump_ref_base_height = 0.0
            jump_ref_z_vel = 0.0
            jump_ref_foot_height = 0.0
            jump_ref_contact = 0.0
            jump_ref_dof_pos = 0.0
            jump_ref_foot_pos = 0.25
            jump_landing_force_balance = 0.0
            jump_hipx_landing = 0.0
            jump_leg_symmetry = -2.0

        only_positive_rewards = True
        base_height_target = 0.43
        base_height_target_vel = 0.8
        target_foot_height = 0.10
        target_foot_height_yaw = 0.10

        soft_landing_contact_threshold = 1.0
        soft_landing_max_z_vel = 0.35
        soft_landing_max_force = 90.0
        soft_landing_force_weight = 0.25

        jump_contact_force_threshold = 5.0
        jump_min_command_speed = 0.05
        jump_min_yaw_speed = 0.05
        jump_sync_air_only = False

        jump_height_min = 0.43
        jump_height_target = 0.58
        jump_z_vel_min = 0.2
        jump_z_vel_target = 1.0
        jump_takeoff_z_vel_min = 0.10
        jump_takeoff_z_vel_target = 1.25
        jump_takeoff_require_contact = True

        jump_phase_compress_end = 0.15
        jump_phase_push_end = 0.35
        jump_phase_flight_end = 0.70
        jump_phase_land_end = 0.85

        jump_ref_height_stance = 0.43
        jump_ref_height_compress = 0.37
        jump_ref_height_takeoff = 0.50
        jump_ref_height_apex = 0.58
        jump_ref_height_land = 0.43

        jump_ref_z_vel_compress = -0.25
        jump_ref_z_vel_push = 1.0
        jump_ref_z_vel_flight = 0.0
        jump_ref_z_vel_flight_up = 0.0
        jump_ref_z_vel_flight_down = 0.0
        jump_ref_z_vel_land = -0.25

        jump_ref_foot_height_stance = 0.02
        jump_ref_foot_height_flight = 0.10
        jump_ref_foot_x_front = 0.185
        jump_ref_foot_x_rear = -0.185
        jump_ref_foot_y_left = 0.16
        jump_ref_foot_y_right = -0.16
        jump_ref_foot_z_stance = -0.40
        jump_ref_foot_z_flight = -0.40
        jump_ref_foot_z_land = -0.40
        jump_ref_foot_x_sweep_back = -0.035
        jump_ref_foot_x_sweep_forward = 0.035
        jump_ref_foot_x_land_offset = 0.010
        jump_ref_foot_pos_weight_x = 0.35
        jump_ref_foot_pos_weight_y = 1.0
        jump_ref_foot_pos_weight_z = 0.0

        jump_ref_hip_y_compress = -0.8
        jump_ref_knee_compress = 1.6
        jump_ref_hip_y_push = -0.8
        jump_ref_knee_push = 1.6
        jump_ref_hip_y_flight = -0.8
        jump_ref_knee_flight = 1.6
        jump_ref_hip_y_land = -0.8
        jump_ref_knee_land = 1.6
        jump_ref_hip_y_recovery = -0.8
        jump_ref_knee_recovery = 1.6

        jump_ref_sigma_height = 0.05
        jump_ref_sigma_z_vel = 0.6
        jump_ref_sigma_foot = 0.05
        jump_ref_sigma_foot_pos = 0.015
        jump_ref_sigma_dof = 0.70
        jump_landing_balance_force_norm = 120.0
        jump_hipx_landing_ids = [0, 3, 6, 9]

        tracking_relief_enabled = True
        tracking_relief_tilt_start = 0.25
        tracking_relief_tilt_end = 0.55
        tracking_relief_min_scale = 0.15


class Cc1JumpHighwtwCfgPPO(Cc1JumpwtwCfgPPO):
    class algorithm(Cc1JumpwtwCfgPPO.algorithm):
        learning_rate = 5.e-4
        desired_kl = 0.01

    class runner(Cc1JumpwtwCfgPPO.runner):
        num_steps_per_env = 100
        max_iterations = 15000
        save_interval = 100
        experiment_name = 'jump_high_wtw_cc1'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
