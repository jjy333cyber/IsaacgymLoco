# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.cc1.cc1_jump_wtw_config import Cc1JumpwtwCfg, Cc1JumpwtwCfgPPO


class Cc1JumpHighwtwCfg(Cc1JumpwtwCfg):
    """Stage-B jump task: keep the WTW synchronized jump and push height/speed higher."""

    class commands(Cc1JumpwtwCfg.commands):
        max_forward_curriculum = 2.2
        max_backward_curriculum = 1.5
        max_lat_curriculum = 1.0

        frequencies = 3.0
        phases = 0.0
        offsets = 0.0
        bounds = 0.0
        durations = 0.5

        class ranges(Cc1JumpwtwCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand(Cc1JumpwtwCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [0.0, 4.0]

        randomize_com_displacement = True
        com_displacement_range = dict(
            x=[0.02, 0.12],
            y=[-0.03, 0.03],
            z=[-0.02, 0.02],
        )

    class rewards(Cc1JumpwtwCfg.rewards):
        class scales(Cc1JumpwtwCfg.rewards.scales):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 3.0

            lin_vel_z = -0.2
            orientation = -2.0
            base_height = -6.0

            action_rate = -0.02
            smoothness = -0.01
            dof_acc = -2.5e-7
            joint_power = -2e-5
            power_distribution = -10e-6
            feet_slide = -0.05
            feet_air_time_variance_velocity = -10.0

            raibert_heuristic = -10.0
            tracking_contacts_shaped_force = 1.0
            tracking_contacts_shaped_vel = 1.0
            feet_clearance_cmd_linear = -40.0

            jump = 2.0
            jump_air_time = 2.0
            jump_height = 3.0
            jump_z_vel = 1.5
            jump_ref_base_height = 2.0
            jump_ref_z_vel = 1.5
            jump_ref_foot_height = 1.5
            jump_ref_contact = 2.0

        only_positive_rewards = True
        base_height_target = 0.48
        base_height_target_vel = 0.8
        target_foot_height = 0.12
        target_foot_height_yaw = 0.10

        jump_contact_force_threshold = 5.0
        jump_min_command_speed = 0.2
        jump_min_yaw_speed = 0.05
        jump_sync_air_only = False

        jump_height_min = 0.43
        jump_height_target = 0.52
        jump_z_vel_min = 0.2
        jump_z_vel_target = 1.0

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
        jump_ref_z_vel_land = -0.25

        jump_ref_foot_height_stance = 0.02
        jump_ref_foot_height_flight = 0.16

        jump_ref_sigma_height = 0.04
        jump_ref_sigma_z_vel = 0.4
        jump_ref_sigma_foot = 0.04

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
