# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# from legged_gym import LEGGED_GYM_ROOT_DIR, envs
# from time import time
# from warnings import WarningMessage
# import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import math
import torch
import warp
import trimesh
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, farthest_point_sampling
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import random_quat
from .legged_robot_config import LeggedRobotCfg
# from rsl_rl.datasets.motion_loader import AMPLoader

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig

class LeggedRobotwtw(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg

        # 1. 确定 各地形的 起止索引
        self.flat_start_idx = 0
        self.flat_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:1]))
        self.rough_start_idx = self.flat_end_idx
        self.rough_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:2]))
        self.smoothslope_start_idx = self.rough_end_idx
        self.smoothslope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:3]))
        self.roughslope_start_idx = self.smoothslope_end_idx
        self.roughslope_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:4]))
        self.stairsup_start_idx = self.roughslope_end_idx
        self.stairsup_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:5]))
        self.stairsdown_start_idx = self.stairsup_end_idx
        self.stairsdown_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:6]))
        self.discreteobstacles_start_idx = self.stairsdown_end_idx
        self.discreteobstacles_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:7]))
        self.steppingstones_start_idx = self.discreteobstacles_end_idx
        self.steppingstones_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:8]))
        self.pit_start_idx = self.steppingstones_end_idx
        self.pit_end_idx = math.ceil(self.cfg.env.num_envs * sum(self.cfg.terrain.terrain_proportions[:9]))
        self.gap_start_idx = self.pit_end_idx
        self.gap_end_idx = self.cfg.env.num_envs

        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False

        # 2. ------ 初始化 ------
        self.init_done = False
        # 2.1 初始化RL训练中每个episode的总步数（1000步）、域随机化中施加推力的步数间隔（800步）、obs的scale、rewars的scale、commands的范围
        self._parse_cfg(self.cfg)
        # 2.2 调用父类 BaseTask 的初始化：
        #   获取 env_cfg 中的 envs个数、obs维度等
        #   调用 create_sim()，创建 sim, terrain and envs
        #   创建 viewer
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations  # 45
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs  # 45 + 3 + 3 + 187
        self.history_length = int(self.num_obs / self.num_one_step_obs)  # 45 * 6 / 45 = 6

        # 2.4 设置观察视角
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # 2.5 创建存储 仿真 state / obs / action 的 tensor
        self._init_buffers()
        # 2.5.1 初始化 WTW/步态相关的自定义 buffer（clock、foot phase、期望触地等）
        self._init_custom_buffers__()
        # 2.6 将使用的 奖励函数 存放到 self.reward_functions 中，并为每个奖励函数 创建一个(num_env,)的tensor，存储在episode中每个env的奖励累计值
        self._prepare_reward_function()

        self.init_done = True
        # ------ 初始化完成 ------

        # 3. 使用激光雷达
        if hasattr(self.cfg, "lidar") and getattr(self.cfg.lidar, "use_lidar", False):
            # 3.1 配置 LiDAR sensor 参数
            self.lidar_cfg = LidarConfig(
                sensor_type=self.cfg.lidar.sensor_type,
                dt=self.cfg.lidar.dt,
                num_sensors=self.cfg.lidar.num_sensors,
                update_frequency=self.cfg.lidar.update_frequency,
                max_range=self.cfg.lidar.max_range,
                enable_sensor_noise=self.cfg.lidar.enable_sensor_noise,
                random_distance_noise=self.cfg.lidar.random_distance_noise,
                pixel_dropout_prob=self.cfg.lidar.pixel_dropout_prob,
                nominal_position=self.cfg.lidar.nominal_position,
                nominal_orientation_euler_deg=self.cfg.lidar.nominal_orientation_euler_deg,
                randomize_placement=self.cfg.lidar.randomize_placement,
            )

            # 3.2 初始化
            self.sim_time = 0
            self.lidar_update_time = 0
            self.lidar_state_update_time = 0
            self.selected_env_idx = self.cfg.lidar.selected_env_idx  # debug时显示rays的env索引
            # self.save_lidar_data = self.cfg.lidar.save_data
            # self.save_lidar_interval = self.cfg.lidar.save_interval
            # self.save_time = 0
            # self.last_save_time = 0

            # 3.3 将 isaacgym 中创建的地形转换为 Warp 格式，使得激光雷达能够准确地与环境交互。并创建所需的一些数据tensor
            warp.init()  # initialize warp after sim
            self.create_warp_env()
            self.create_warp_tensor()
            # 3.4 创建 LiDAR 传感器
            self.lidar = LidarSensor(env=self.warp_tensor_dict, env_cfg=None, sensor_cfg=self.lidar_cfg, num_sensors=1, device=self.device)
            # 获取 lidar 数据
            # MID360: (num_envs, num_sensors, 20000, 1, 3), (num_envs, num_sensors, 20000, 1)
            self.lidar_tensor, self.lidar_dist_tensor = self.lidar.update()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 1. 当前env_step的 actions，裁剪到 [-100.0, 100.0]
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # 2. 计算延迟后的 actions（域随机化）
        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)  # (num_envs, 4, 12)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)  # 每个 env 延迟的步数 [0, 4)
        if self.cfg.domain_rand.delay:
            # 计算延迟的4个actions（ < 延迟步数，则为上一时刻的actions，>= 延迟步数，则为最新时刻的actions）
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)

        # 3. 渲染（非headless模式）
        self.render()

        # 4. 执行一个 env_step（包含 4 个sim_step，即依次执行延迟后的4个actions）
        for _ in range(self.cfg.control.decimation):
            # 从 actions 计算 扭矩 (num_envs, 12)
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            # 应用 该扭矩 到 仿真环境
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)  # 执行物理仿真
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)  # 获取仿真结果
            self.gym.refresh_dof_state_tensor(self.sim)  # 更新 关节状态

        # 5. 执行 4个物理仿真步后的 操作
        # (1) 更新机器人的姿态
        # (2) 计算高度场
        # (3) 给base施加干扰
        # (4) 计算 奖励
        # (5) 计算 新的观测
        # (6) 重置某些env
        # (7) 更新上一env_step的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）
        #   返回： 需要重置的env的 ID (num_envs_,) 以及这些env的特权观测 (num_envs_, 45+3+3+187)
        # termination_ids, termination_priveleged_obs = self.post_physics_step()
        termination_ids, termination_priveleged_obs, terminal_amp_states = self.post_physics_step()

        # 6. 裁剪 观测obs_buf 到 [-100., 100.]
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # 7. 裁剪 特权观测privileged_obs_buf 到 [-100., 100.]
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if self.cfg.env.using_amp:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs, terminal_amp_states
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """
        更新机器人的姿态，计算高度场，给base施加干扰，计算 奖励、观测，重置某些env，更新上一env_step的数据（action、关节位置、关节速度、扭矩、base的线速度和角速度）

        Returns:
            env_ids: 需要重置的env的 ID (num_envs_,)
            terminal_amp_states: 需要重置的 env 的 AMP观测 (num_envs_, 30)
        """
        if hasattr(self.cfg, "lidar") and getattr(self.cfg.lidar, "use_lidar", False):
            self.sim_time += self.dt
            self.lidar_update_time += self.dt
            self.lidar_state_update_time += self.dt
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()

        # 1. 从 Isaac Gym 仿真器中刷新各种状态张量，确保数据是最新的
        self.gym.refresh_actor_root_state_tensor(self.sim)   # 刷新 base的状态 张量
        self.gym.refresh_net_contact_force_tensor(self.sim)  # 刷新 关节接触力 张量
        self.gym.refresh_force_sensor_tensor(self.sim)       # 刷新 力传感器 张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)   # 刷新 刚体状态 张量

        # 2. 增加 当前回合的 步数计数器 和 通用步数计数器
        self.episode_length_buf += 1   # 当前回合的env_step数 +1
        self.common_step_counter += 1  # env_step数 +1

        # 3. 更新机器人的姿态、速度和重力投影信息
        self.base_pose = self.root_states[:, :7]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]  # 更新机器人 base 的旋转四元数（世界坐标系）
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])   # 更新机器人 base 的 线速度（body坐标系）
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])  # 更新机器人 base 的 角速度（body坐标系）
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 更新投影到机器人坐标系的 重力向量（body坐标系）
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_base_lin_vel) / self.dt  # base的 线加速度（暂时没用到）
        self.base_ang_acc = (self.root_states[:, 10:13] - self.last_base_ang_vel) / self.dt

        if hasattr(self.cfg, "lidar") and getattr(self.cfg.lidar, "use_lidar", False):
            # update lidar position and orientation
            lidar_pos = self.base_pos + quat_apply(self.base_quat, self.lidar_translation)
            lidar_quat = quat_mul(self.base_quat, self.lidar_offset_quat)
            self.lidar_pos_tensor[:, :] = lidar_pos
            self.lidar_quat_tensor[:, :] = lidar_quat

            # update lidar data
            # MID360: 点云 (num_envs, num_sensors, 20000, 1, 3), 距离 (num_envs, num_sensors, 20000, 1)
            self.lidar_tensor, self.lidar_dist_tensor = self.lidar.update()
            # (num_envs, num_sensors, 2000, 3)
            self.downsampled_lidar_cloud = farthest_point_sampling(self.lidar_tensor.view(self.num_envs, self.lidar_cfg.num_sensors,
                                                                                          self.lidar_tensor.shape[2], 3), sample_size=2000)
            # print(f"LiDAR distance range: {self.lidar_dist_tensor.min():.2f} - {self.lidar_dist_tensor.max():.2f}")

            # debug LiDAR rays in viewer
            if self.cfg.lidar.debug_vis and (self.lidar_update_time > (1 / self.lidar_cfg.update_frequency)):
                self.gym.clear_lines(self.viewer)
                self.draw_lidar_vis()
                self.lidar_update_time = 0

        # 四足的 位置 和 线速度（世界坐标系）
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # 四足的 接触力 是否 > 1，来判断是否接触地面
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.prev_contacts[:] = self.last_contacts
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        if hasattr(self, "lateral_pair_last_contacts"):
            lateral_pair_contact = self._lateral_pair_contact_mask()
            self.lateral_pair_prev_contacts[:] = self.lateral_pair_last_contacts
            self.lateral_pair_last_contacts[:] = lateral_pair_contact
            self.lateral_pair_prev_contact_duration[:] = self.lateral_pair_contact_duration
            self.lateral_pair_contact_duration[:] = torch.where(
                lateral_pair_contact,
                self.lateral_pair_contact_duration + self.dt,
                torch.zeros_like(self.lateral_pair_contact_duration),
            )

        # 原代码在计算奖励前调用 _post_physics_step_callback()，这可能不合理。例如，当前动作遵循当前命令，而 _post_physics_step_callback() 可能会重新采样命令，导致奖励较低。
        # 4. 每500个env_step重新采样这些env的commands，
        # 计算地形高度场，
        # 给base施加 水平速度干扰
        # 给base施加 力干扰
        self._post_physics_step_callback()

        # 5. 检查环境是否需要 重置
        self.check_termination()

        # 6. 计算 奖励
        self.compute_reward()

        # 7. 重置某些 env
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()  # 获取需要重置的 env ID
        # 获取需要重置的 env 的特权观测 (num_envs_, 45+3+3+187)
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        terminal_amp_states = self.get_amp_observations(env_ids)
        self.reset_idx(env_ids)  # 重置这些 env

        # 8. 计算 观测
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 9. 更新上一env_step的 actions、关节位置、关节速度、扭矩、base的线速度和角速度
        self.disturbance[:, :, :] = 0.0  # 给各刚体的扰动力 清零
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        # return env_ids, termination_privileged_obs
        return env_ids, termination_privileged_obs, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_counts = {}
        # (1) 触发终止部位的接触力 > 1N，则需要重置 (num_envs,)
        contact_force_cond = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.contact_termination_buf = contact_force_cond
        self.reset_buf = contact_force_cond
        termination_counts["contact_force"] = (contact_force_cond.sum().item() / self.num_envs) * 100
        # print(f'[legged_robot] termination_counts contact_force (%): {termination_counts["contact_force"]}]')

        # (2) episode步数 > 1000
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        termination_counts["time_out"] = (self.time_out_buf.sum().item() / self.num_envs) * 100
        # print(f'[legged_robot] termination_counts time_out (%): {termination_counts["time_out"]}]')

        # (3) base速度 与 命令速度（因摔倒恢复训练，需关闭这个）
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "base_vel_violate_commands", False):
            vel_error = self.base_lin_vel[:, 0] - self.commands[:, 0]
            self.vel_violate = ((vel_error > 2) & (self.commands[:, 0] < 0.)) | ((vel_error < -2) & (self.commands[:, 0] > 0.))
            self.vel_violate *= (self.terrain_levels > 3)
            self.reset_buf |= self.vel_violate
            termination_counts["vel_violate"] = (self.vel_violate.sum().item() / self.num_envs) * 100
            # print(f'[legged_robot] termination_counts vel_violate (%): {termination_counts["vel_violate"]}]')

        # (4) env走出地形的边界
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "out_of_border", False):
            if not hasattr(self, "terrain"):
                self.out_border = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                self.out_border = self.terrain.in_terrain_range(self.root_states[:, :3], device=self.device).logical_not()
            self.reset_buf |= self.out_border
            termination_counts["out_border"] = (self.out_border.sum().item() / self.num_envs) * 100
            # print(f'[legged_robot] termination_counts out_border (%): {termination_counts["out_border"]}]')

        # (5) base的z方向线速度 < -5 （即跌落）# 或 重力投影 为 Z轴向上（因摔倒恢复训练，需关闭这个，避免刚重置就终止了）
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "fall_down", False):
            self.fall_down = (self.root_states[:, 9] < -5.)  #  | (self.projected_gravity[:, 2] > 0.)
            self.reset_buf |= self.fall_down
            termination_counts["fall_down"] = (self.fall_down.sum().item() / self.num_envs) * 100
            # print(f'[legged_robot] termination_counts fall_down (%): {termination_counts["fall_down"]}]')

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # 1. 更新地形课程（根据机器人表现调整地形难度）
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # 2. 更新 commands 课程（调整速度命令范围）
        # 避免每步都更新，因为最大命令对所有env是共享的
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # 重置关节、base状态
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Keep cached body-frame states consistent with the newly written root
        # state before observations are built.
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13]
        )
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids]
        )

        self.episode_start_pos[env_ids] = self.root_states[env_ids, :3]
        episode_forward = quat_apply(
            self.root_states[env_ids, 3:7], self.forward_vec[env_ids]
        )
        episode_forward[:, 2] = 0.
        self.episode_start_forward[env_ids] = episode_forward / torch.clamp(
            torch.norm(episode_forward, dim=1, keepdim=True),
            min=1e-6,
        )

        # 4. 为重置的 env 重新采样 commands
        self._resample_commands(env_ids)
        if getattr(self.cfg.commands, "single_jump_flag_mode", False):
            self._reset_single_jump_flag_state(env_ids)
        self.sudden_stop_time_left[env_ids] = 0.
        self.sudden_stop_pending[env_ids] = False
        self.sudden_stop_seen_air[env_ids] = False
        self.sudden_stop_saved_commands[env_ids] = 0.

        # 6. 重置各种缓冲区
        if getattr(self.cfg.commands, "single_jump_mode", False):
            self.gait_indices[env_ids] = 0.
        self.lateral_pair_touchdown_armed[env_ids] = True
        self.lateral_pair_takeoff_armed[env_ids] = True
        self.lateral_pair_touchdown_seen_in_stance[env_ids] = False
        self.lateral_pair_prev_contacts[env_ids] = False
        self.lateral_pair_last_contacts[env_ids] = False
        self.lateral_pair_last_phase[env_ids] = 0.
        self.lateral_pair_touchdown_cooldown[env_ids] = 0.
        self.lateral_pair_contact_duration[env_ids] = 0.
        self.lateral_pair_prev_contact_duration[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # update height measurements only for reset environments
        if self.cfg.terrain.measure_heights:
            self.measured_heights[env_ids] = self._get_heights(env_ids)
        
         # 重新获取 域随机化 数据
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        # 重新获取 env的摩擦系数、弹性系数，并设置给env的各部位
        if getattr(
            self.cfg.domain_rand, "randomize_rigid_shape_props_on_reset", True
        ):
            self.refresh_actor_rigid_shape_props(env_ids)
        
        # 记录episode信息
        self.extras["episode"] = {}
        self.extras["episode"]["reset_fraction"] = torch.tensor(
            len(env_ids) / self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.extras["episode"]["contact_termination_fraction"] = torch.mean(
            self.contact_termination_buf.float()
        )
        for key in self.episode_sums.keys():
            # 遍历每个奖励函数，计算对应的 这些重置的env在当前episode内的 (平均奖励值 / 0.02)的均值
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # 记录课程信息
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            # 记录当前命令范围
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        #  发送超时信息给算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.  # (num_envs,)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew  # 对应env的 所有奖励之和 (num_envs,)
            self.episode_sums[name] += rew  # 对应奖励函数 在当前episode内的 对应env的 奖励之和 (num_envs,)
        if self.cfg.rewards.only_positive_rewards:  # 默认不执行
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # 在裁剪至0后，重新赋值终止的惩罚
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]  # (num_envs,)
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _observed_command_dim(self):
        if getattr(self.cfg.commands, "observe_jump_commands", False) and self.cfg.commands.num_commands >= 6:
            return 6
        return 3

    def _get_observed_commands(self):
        observed_commands = self.commands[:, :3] * self.commands_scale
        if self._observed_command_dim() > 3:
            jump_commands = self.commands[:, 3:6] * self.jump_commands_scale
            observed_commands = torch.cat((observed_commands, jump_commands), dim=-1)
        return observed_commands

    def _proprioceptive_obs_dim(self):
        return 6 + self._observed_command_dim() + 4 * self.num_actions + 4

    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.base_ang_vel  * self.obs_scales.ang_vel,  # 0.25
                                    self.projected_gravity,
                                    self._get_observed_commands(),
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 1.0
                                    self.dof_vel * self.obs_scales.dof_vel,  # 0.05
                                    self.actions,  
                                    self.last_actions,
                                    self.clock_inputs
                                    ),dim=-1)
        proprioceptive_obs_dim = self._proprioceptive_obs_dim()
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:proprioceptive_obs_dim]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)  # base线速度 * 2.0, 给base施加的随机扰动力(xyz方向)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[proprioceptive_obs_dim:(proprioceptive_obs_dim + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)  # 6 steps
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def get_amp_observations(self, env_ids=None):
        if env_ids is not None and len(env_ids) == 0:
            amp_obs_dim = (
                43
                if getattr(
                    self.cfg.env, "amp_include_foot_pos_lin_vel", False
                )
                else 28
            )
            return torch.empty(
                (0, amp_obs_dim),
                dtype=self.dof_pos.dtype,
                device=self.device,
            )

        if env_ids is None:
            joint_pos = self.dof_pos
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            joint_vel = self.dof_vel
            root_states = self.root_states
            base_quat = self.base_quat
            feet_pos = self.feet_pos
        else:
            joint_pos = self.dof_pos[env_ids]
            base_lin_vel = self.base_lin_vel[env_ids]
            base_ang_vel = self.base_ang_vel[env_ids]
            joint_vel = self.dof_vel[env_ids]
            root_states = self.root_states[env_ids]
            base_quat = self.base_quat[env_ids]
            feet_pos = self.feet_pos[env_ids]

        z_pos = root_states[:, 2:3]
        if self.cfg.terrain.measure_heights:
            measured_heights = (
                self.measured_heights
                if env_ids is None
                else self.measured_heights[env_ids]
            )
            z_pos = z_pos - torch.mean(
                measured_heights, dim=-1, keepdim=True
            )
        if getattr(self.cfg.env, "amp_include_foot_pos_lin_vel", False):
            foot_pos_world = feet_pos - root_states[:, :3].unsqueeze(1)
            num_states = root_states.shape[0]
            foot_pos = quat_rotate_inverse(
                base_quat.unsqueeze(1).repeat(
                    1, len(self.feet_indices), 1
                ).reshape(-1, 4),
                foot_pos_world.reshape(-1, 3),
            ).reshape(num_states, len(self.feet_indices) * 3)
            # These features prevent a kneeling policy from matching the expert
            # using only similar joint angles and base height.
            return torch.cat(
                (joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos),
                dim=-1,
            )
        return torch.cat((joint_pos, base_ang_vel, joint_vel, z_pos), dim=-1)

    def get_current_obs(self):
        current_obs = torch.cat((   self.base_ang_vel  * self.obs_scales.ang_vel,  # 0.25
                                    self.projected_gravity,
                                    self._get_observed_commands(),
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 1.0
                                    self.dof_vel * self.obs_scales.dof_vel,  # 0.05
                                    self.actions,  
                                    self.last_actions,
                                    self.clock_inputs
                                    ),dim=-1)
        proprioceptive_obs_dim = self._proprioceptive_obs_dim()
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:proprioceptive_obs_dim]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[proprioceptive_obs_dim:(proprioceptive_obs_dim + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations (num_envs, 45+3+3+187)
        """
        current_obs = torch.cat((   self.base_ang_vel  * self.obs_scales.ang_vel,  # 0.25
                                    self.projected_gravity,
                                    self._get_observed_commands(),
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 1.0
                                    self.dof_vel * self.obs_scales.dof_vel,  # 0.05
                                    self.actions,  
                                    self.last_actions,
                                    self.clock_inputs
                                    ),dim=-1)
        proprioceptive_obs_dim = self._proprioceptive_obs_dim()
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:proprioceptive_obs_dim]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[proprioceptive_obs_dim:(proprioceptive_obs_dim + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # 1. 创建 sim
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # 2. 创建 terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        # 3. 创建 agents
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # 为每个env生成一个随机摩擦系数，并将同一个env的所有刚体部位的摩擦系数 都设置为相同的数值
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)
                # 为每个env生成一个随机摩擦数 (num_env, 1)
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        # 为每个env生成一个随机弹性系数，并将同一个env的所有刚体部位的弹性系数 都设置为相同的数值
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1), device=self.device)
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """
         存储、处理、随机 关节属性，包括：位置限制、速度限制、力矩限制（env 创建期间被调用）

        Args:
            props (numpy.array): 每个关节的属性数组，包含 位置/速度/力矩
            env_id (int): 当前环境ID，用于判断是否需要初始化限制参数

        Returns:
            [numpy.array]: 原始属性（未修改）
        """
        # 只在第一个环境初始化时设置关节限制
        if env_id==0:
            # 初始化存储关节限制的张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)  # (num_dof, 2)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # (num_dof,)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)   # (num_dof,)
            # 遍历每个关节属性
            for i in range(len(props)):
                # 存储原始关节限制
                self.dof_pos_limits[i, 0] = props["lower"][i].item()  # 最小 位置 限制
                self.dof_pos_limits[i, 1] = props["upper"][i].item()  # 最大 位置 限制
                self.dof_vel_limits[i] = props["velocity"][i].item()  # 最大 速度 限制
                self.torque_limits[i] = props["effort"][i].item()  # 最大 力矩 限制

                # 计算软限制（比硬件限制更宽松的范围）
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2  # 中间值
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]  # 范围
                # 根据配置设置软限制范围（self.cfg.rewards.soft_dof_pos_limit 通常为0.9）
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props  # 返回原始属性（未修改）

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # 随机更改base的质量
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]

        # 随机更改base的质心
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        # 随机更改env各刚体部位（除了base）的质量
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props

    def _sample_com_displacement(self, num_envs):
        com_range = self.cfg.domain_rand.com_displacement_range
        if isinstance(com_range, dict):
            x_range = com_range.get("x", [0.0, 0.0])
            y_range = com_range.get("y", [0.0, 0.0])
            z_range = com_range.get("z", [0.0, 0.0])
            return torch.cat((
                torch_rand_float(x_range[0], x_range[1], (num_envs, 1), device=self.device),
                torch_rand_float(y_range[0], y_range[1], (num_envs, 1), device=self.device),
                torch_rand_float(z_range[0], z_range[1], (num_envs, 1), device=self.device),
            ), dim=1)
        return torch_rand_float(com_range[0], com_range[1], (num_envs, 3), device=self.device)
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 1. envs中 当其episode的 env_step数达到500步，则重新采样commands
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # 2. 根据目标航向角偏差，计算commands的角速度
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)  # 当前base的前进方向（世界坐标系）(num_envs, 3)
            heading = torch.atan2(forward[:, 1], forward[:, 0])  # 当前base的前进方向的 航向角
            # 命令的角速度 = 0.5 * (目标航向 - 当前航向)[-pi, pi] ==> 裁剪到[-2, 2]
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        self._apply_single_jump_flag_commands()
        self._apply_single_jump_commands()
        self._apply_sudden_stop_commands()

        # 3. 计算采样点的高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 3.5 更新步态相关的期望触地/时钟输入（若未启用 observe_gait_commands，该函数内部会直接跳过）
        self._step_contact_targets()

        # 4. 每16/0.02s个 env_step，给base在水平方向施加一个速度
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        # 5. 每8个 env_step，给base施加一个随机的力
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     # 重新采样env_ids的commands
    #     self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     else:
    #         self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    #     # 处理高速 env 的commands（前 20%）
    #     high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
    #     high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
    #     self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)
    #     # set y commands of high vel envs to zero
    #     self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

    #     # set small commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _resample_commands(self, env_ids):
        """ Randomly select commands of some environments """

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],self.command_ranges["lin_vel_y"][1],(len(env_ids), 1),device=self.device,).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],self.command_ranges["heading"][1],(len(env_ids), 1),device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],self.command_ranges["ang_vel_yaw"][1],(len(env_ids), 1),device=self.device).squeeze(1)

        # 2. 高速 env（前20%）
        high_vel_mask = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_mask.nonzero(as_tuple=True)]
        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],self.command_ranges["lin_vel_x"][1],(len(high_vel_env_ids), 1),device=self.device).squeeze(1)
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # 3. 新增：随机选10%完全静止
        if self.cfg.commands.stand_still_command:
            num_envs = len(env_ids)
            perm = torch.randperm(num_envs, device=self.device)
            num_zero = int(0.2 * num_envs)
            zero_env_ids = env_ids[perm[:num_zero]]
            self.commands[zero_env_ids, 0:3] = 0.0

        # 4. 新增：随机选30% 只转向
        if self.cfg.commands.Rotate_command:
            num_turn = int(0.2 * num_envs)
            turn_env_ids = env_ids[perm[num_zero:num_zero + num_turn]]
            self.commands[turn_env_ids, 0:2] = 0.0
            
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        if getattr(self.cfg.commands, "single_jump_flag_mode", False):
            if getattr(self.cfg.commands, "single_jump_zero_velocity_commands", True):
                self.commands[env_ids, 0:3] = 0.0
            if self.cfg.commands.num_commands >= 6:
                x_range = getattr(self.cfg.commands, "jump_target_x_range", [0.8, 1.2])
                y_range = getattr(self.cfg.commands, "jump_target_y_range", [0.0, 0.0])
                self.commands[env_ids, 3] = torch_rand_float(x_range[0], x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.commands[env_ids, 4] = torch_rand_float(y_range[0], y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.commands[env_ids, 5] = 0.0

    def _single_jump_flag_enabled(self):
        return getattr(self.cfg.commands, "single_jump_flag_mode", False) and self.cfg.commands.num_commands >= 6

    def _single_jump_flag(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # The command is written after physics, so rewards/state-machine use it one
        # step later. This lets the policy observe jump_flag before being rewarded
        # for reacting to it.
        return (self.commands[:, 5] > 0.5) & (self.episode_length_buf > self.command_frame)

    def _reset_single_jump_flag_state(self, env_ids):
        if len(env_ids) == 0 or not self._single_jump_flag_enabled():
            return

        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False
        self.jump_assist_used[env_ids] = False
        self.landing_pos[env_ids] = 0.0
        self.landing_forward_distance[env_ids] = 0.0
        self.landing_lateral_distance[env_ids] = 0.0

        frame_range = getattr(self.cfg.commands, "single_jump_command_frame_range", [50, 60])
        low = int(frame_range[0])
        high = int(frame_range[1]) + 1
        self.command_frame[env_ids] = torch.randint(low, high, (len(env_ids),), device=self.device)

        if getattr(self.cfg.commands, "single_jump_zero_velocity_commands", True):
            self.commands[env_ids, 0:3] = 0.0

        x_range = getattr(self.cfg.commands, "jump_target_x_range", [0.8, 1.2])
        y_range = getattr(self.cfg.commands, "jump_target_y_range", [0.0, 0.0])
        self.commands[env_ids, 3] = torch_rand_float(x_range[0], x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 4] = torch_rand_float(y_range[0], y_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 5] = 0.0

    def _apply_single_jump_flag_commands(self):
        if not self._single_jump_flag_enabled():
            return

        if getattr(self.cfg.commands, "single_jump_zero_velocity_commands", True):
            self.commands[:, 0:3] = 0.0

        pre_trigger = self.episode_length_buf < self.command_frame
        trigger = (self.episode_length_buf >= self.command_frame) & (~self.has_jumped)
        self.commands[pre_trigger, 5] = 0.0
        self.commands[trigger | self.has_jumped, 5] = 1.0

        self._update_single_jump_flag_state()
        self._push_towards_jump_goal()

    def _update_single_jump_flag_state(self):
        jump_flag = self._single_jump_flag()
        contact = self._jump_contact_mask()
        all_air = torch.logical_not(torch.any(contact, dim=1))
        any_contact = torch.any(contact, dim=1)

        active_jump = jump_flag & (~self.has_jumped)
        self.was_in_flight |= active_jump & all_air

        landed = self.was_in_flight & any_contact & (~self.has_jumped)
        if torch.any(landed):
            self.landing_pos[landed] = self.root_states[landed, :3]
            displacement = self.landing_pos[landed] - self.episode_start_pos[landed]
            forward = self.episode_start_forward[landed]
            lateral = torch.stack((-forward[:, 1], forward[:, 0], torch.zeros_like(forward[:, 2])), dim=1)
            self.landing_forward_distance[landed] = torch.sum(displacement * forward, dim=1)
            self.landing_lateral_distance[landed] = torch.sum(displacement * lateral, dim=1)
            self.has_jumped[landed] = True
            self.was_in_flight[landed] = False

    def _push_towards_jump_goal(self):
        if not (self._single_jump_flag_enabled() and getattr(self.cfg.commands, "push_towards_goal", False)):
            return

        eligible = self._single_jump_flag() & (~self.was_in_flight) & (~self.has_jumped) & (~self.jump_assist_used)
        if not torch.any(eligible):
            return

        start_prob = getattr(self.cfg.commands, "jump_assist_start_prob", 0.6)
        end_prob = getattr(self.cfg.commands, "jump_assist_end_prob", 0.0)
        decay_steps = max(getattr(self.cfg.commands, "jump_assist_decay_steps", 500000), 1)
        progress = min(float(self.common_step_counter) / float(decay_steps), 1.0)
        assist_prob = start_prob + (end_prob - start_prob) * progress
        selected = eligible & (torch.rand(self.num_envs, device=self.device) < assist_prob)
        if not torch.any(selected):
            return

        env_ids = selected.nonzero(as_tuple=False).flatten()
        x_range = getattr(self.cfg.commands, "jump_assist_x_vel_range", [0.1, 0.4])
        z_range = getattr(self.cfg.commands, "jump_assist_z_vel_range", [0.4, 0.9])
        x_vel = torch_rand_float(x_range[0], x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        z_vel = torch_rand_float(z_range[0], z_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 7:9] += self.episode_start_forward[env_ids, :2] * x_vel.unsqueeze(1)
        self.root_states[env_ids, 9] += z_vel
        self.jump_assist_used[env_ids] = True

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        
    def _single_jump_active_mask(self):
        if not getattr(self.cfg.commands, "single_jump_mode", False):
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        active_time_s = getattr(self.cfg.commands, "single_jump_command_time_s", 1.05)
        active_steps = max(int(active_time_s / self.dt), 1)
        return self.episode_length_buf <= active_steps

    def _apply_single_jump_commands(self):
        if getattr(self.cfg.commands, "single_jump_flag_mode", False) or not getattr(self.cfg.commands, "single_jump_mode", False):
            return

        if getattr(self.cfg.commands, "single_jump_forward_only", True):
            self.commands[:, 1:3] = 0.0

        active = self._single_jump_active_mask().to(dtype=self.commands.dtype).unsqueeze(1)
        self.commands[:, 0:3] *= active

    def _apply_sudden_stop_commands(self):
        if not getattr(self.cfg.commands, "sudden_stop_command", False):
            return

        active_stop = self.sudden_stop_time_left > 0.0
        active_stop_f = active_stop.to(dtype=self.commands.dtype)
        self.commands[:, 0:3] *= (1.0 - active_stop_f).unsqueeze(1)
        self.sudden_stop_time_left.sub_(self.dt * active_stop_f).clamp_(min=0.0)
        finished_stop = active_stop & (self.sudden_stop_time_left <= 0.0)
        if getattr(self.cfg.commands, "sudden_stop_restore_command", False):
            self.commands[finished_stop, 0:3] = self.sudden_stop_saved_commands[finished_stop]

        phase_aware = getattr(self.cfg.commands, "sudden_stop_phase_aware", False)
        if phase_aware:
            contact = self._jump_contact_mask()
            all_air = torch.logical_not(torch.any(contact, dim=1))
            all_contact = torch.all(contact, dim=1)
            self.sudden_stop_seen_air |= self.sudden_stop_pending & all_air

            ready_stop = (
                self.sudden_stop_pending
                & self.sudden_stop_seen_air
                & all_contact
                & (~active_stop)
            )
            duration_range = getattr(self.cfg.commands, "sudden_stop_duration_s", [0.4, 1.0])
            durations = torch_rand_float(
                duration_range[0],
                duration_range[1],
                (self.num_envs, 1),
                device=self.device,
            ).squeeze(1)
            self.sudden_stop_time_left = torch.where(
                ready_stop,
                durations,
                self.sudden_stop_time_left,
            )
            self.sudden_stop_pending &= ~ready_stop
            self.sudden_stop_seen_air &= ~ready_stop
            self.commands[ready_stop, 0:3] = 0.0

        interval_s = getattr(self.cfg.commands, "sudden_stop_interval_s", 2.0)
        interval_steps = max(int(interval_s / self.dt), 1)
        if self.common_step_counter % interval_steps != 0:
            return

        min_speed = getattr(self.cfg.commands, "sudden_stop_min_speed", 0.25)
        min_yaw = getattr(self.cfg.commands, "sudden_stop_min_yaw_speed", 0.25)
        min_episode_time_s = getattr(self.cfg.commands, "sudden_stop_min_episode_time_s", 1.0)
        min_episode_steps = max(int(min_episode_time_s / self.dt), 1)
        moving = (torch.norm(self.commands[:, :2], dim=1) > min_speed) | (torch.abs(self.commands[:, 2]) > min_yaw)
        eligible = (
            moving
            & (~active_stop)
            & (~self.sudden_stop_pending)
            & (~finished_stop)
            & (self.episode_length_buf > min_episode_steps)
        )

        ratio_range = getattr(self.cfg.commands, "sudden_stop_env_ratio_range", [0.02, 0.08])
        stop_prob = torch_rand_float(ratio_range[0], ratio_range[1], (1, 1), device=self.device).squeeze()
        selected = eligible & (torch.rand(self.num_envs, device=self.device) < stop_prob)

        self.sudden_stop_saved_commands[selected] = self.commands[selected, 0:3]
        if phase_aware:
            self.sudden_stop_pending |= selected
            self.sudden_stop_seen_air |= selected & self._jump_all_air_mask()
        else:
            duration_range = getattr(self.cfg.commands, "sudden_stop_duration_s", [0.4, 1.0])
            durations = torch_rand_float(
                duration_range[0],
                duration_range[1],
                (self.num_envs, 1),
                device=self.device,
            ).squeeze(1)
            self.sudden_stop_time_left = torch.where(selected, durations, self.sudden_stop_time_left)
            self.commands[:, 0:3] *= (1.0 - selected.to(dtype=self.commands.dtype)).unsqueeze(1)

    def _step_contact_targets(self):
        """根据步态相关 command 更新期望触地状态/时钟输入。

        当开启 observe_gait_commands 时，policy 不仅接收速度指令，还会接收/输出步态参数。
        这里会把这些参数变换成：
        - 每条腿的 phase clock（sin 时钟信号，含 1x/2x/0.5x 频率）；
        - desired_contact_states：每条腿在当前时刻“应该处于支撑相”的概率/软指示。
        """
        if self.cfg.env.observe_gait_commands:
            # commands 各维含义（与配置对应）：
            # [4]=frequency, [5]=phase, [6]=offset, [7]=bound, [8]=duration(占空比/支撑比例)
            # frequencies = self.cfg.commands.frequencies
            # phases = self.cfg.commands.phases   
            # offsets = self.cfg.commands.offsets
            # bounds = self.cfg.commands.bounds
            # durations = self.cfg.commands.durations
            frequencies = torch.ones(self.num_envs, device=self.device) * self.cfg.commands.frequencies
            phases      = torch.ones(self.num_envs, device=self.device) * self.cfg.commands.phases
            offsets     = torch.ones(self.num_envs, device=self.device) * self.cfg.commands.offsets
            bounds      = torch.ones(self.num_envs, device=self.device) * self.cfg.commands.bounds
            durations   = torch.ones(self.num_envs, device=self.device) * self.cfg.commands.durations

            # gait_indices 是一个全局相位指针，每步按 frequency 推进，并取模到 [0,1)。
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            if self.cfg.commands.pacing_offset:
                # pacing_offset=True 时，四条腿的相位组合顺序按 pacing 习惯调整。
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:
                # 默认情况下：FL/FR/RL/RR 的相位由 (global + phase/offset/bound) 线性组合得到。
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]
                
            condition = (torch.norm(self.commands[:, :2], dim=1) < 0.2) & (torch.abs(self.commands[:, 2]) < 0.05)
            if torch.any(condition):
                for i in range(4):
                    foot_indices[i][condition] = 0.25

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                # 依据 duration（支撑占空比）把一个周期切成 stance/swing 两段。
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                # 将 stance 映射到 [0, 0.5)，swing 映射到 [0.5, 1)。
                # 这样 sin clock 在 stance/swing 两段的形状更一致，便于 policy 使用。
                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # 基于每条腿的相位 idx，构造不同频率的正弦 clock 输入（常见于步态条件策略）。

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # 用“平滑的周期性方波”表示期望触地（支撑）概率：
            # 这里用 Normal CDF 来实现 0/1 之间的软过渡（kappa 越大，过渡越平滑/越宽）。
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            # 若 command 额外包含 foot swing height，则直接从 command 读取。
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions = self.motor_strength * actions
        # 1. 根据输入的 actions，计算关节目标位置 = default_dof_pos + actions * 0.5
        actions_scaled = actions * self.cfg.control.action_scale  # actions * 0.5
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_reduction  # hip关节的再 * 1.0
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        # 2. 根据控制类型计算扭矩
        control_type = self.cfg.control.control_type
        if control_type=="P":  # 位置控制模式
            # 扭矩 = P增益 * 域随机化系数 * (关节目标位置 - 关节当前位置) + D增益 * 域随机化系数 * 关节当前速度
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":  # 速度控制模式
            # 扭矩 = P增益 * (目标速度 - 当前速度) + D增益 * 加速度              其中，加速度 = (当前速度 - 上一时刻速度) / 物理时间步长
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type=="T":  # 扭矩控制模式
            # 直接使用 缩放后的action 作为 扭矩
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)  # 裁剪到扭矩限制范围内

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if getattr(self.cfg.domain_rand, "dof_init_pos_ratio_range", None) is not None:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
                self.cfg.domain_rand.dof_init_pos_ratio_range[0],
                self.cfg.domain_rand.dof_init_pos_ratio_range[1],
                (len(env_ids), self.num_dof),
                device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos

        if getattr(self.cfg.domain_rand, "randomize_dof_vel", False):
            dof_vel_range = getattr(self.cfg.domain_rand, "init_dof_vel_range", [-1.0, 1.0])
            self.dof_vel[env_ids] = torch.rand_like(self.dof_vel[env_ids]) * abs(dof_vel_range[1] - dof_vel_range[0]) + min(dof_vel_range)
        else:
            self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if hasattr(self.cfg.domain_rand, "base_init_pos_range"):
                self.root_states[env_ids, 0:1] += torch_rand_float(
                    *self.cfg.domain_rand.base_init_pos_range["x"],
                    (len(env_ids), 1),
                    device=self.device
                )
                self.root_states[env_ids, 1:2] += torch_rand_float(
                    *self.cfg.domain_rand.base_init_pos_range["y"],
                    (len(env_ids), 1),
                    device=self.device
                )
                # random height
                self.root_states[env_ids, 2:3] += torch_rand_float(
                    *self.cfg.domain_rand.base_init_pos_range["z"],
                    (len(env_ids), 1),
                    device=self.device
                )
            else:  # 默认x,y方向为 [-1, 1], z方向为 0
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base rotation (roll and pitch)
        if hasattr(self.cfg.domain_rand, "base_init_rot_range"):
            base_roll = torch_rand_float(
                *self.cfg.domain_rand.base_init_rot_range["roll"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_pitch = torch_rand_float(
                *self.cfg.domain_rand.base_init_rot_range["pitch"],
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_yaw = torch_rand_float(
                *self.cfg.domain_rand.base_init_rot_range.get("yaw", [-np.pi, np.pi]),
                (len(env_ids), 1),
                device=self.device,
            )[:, 0]
            base_quat = quat_from_euler_xyz(base_roll, base_pitch, base_yaw)
            self.root_states[env_ids, 3:7] = base_quat

        # base velocities
        if getattr(self.cfg.domain_rand, "base_init_vel_range", None) is not None:
            base_vel_range = self.cfg.domain_rand.base_init_vel_range
        else:
            base_vel_range = (-0.5, 0.5)
        if isinstance(base_vel_range, (tuple, list)):
            self.root_states[env_ids, 7:13] = torch_rand_float(
                *base_vel_range,
                (len(env_ids), 6),
                device=self.device
            ) # [7:10]: lin vel, [10:13]: ang vel
        elif isinstance(base_vel_range, dict):
            self.root_states[env_ids, 7:8] = torch_rand_float(
                *base_vel_range["x"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 8:9] = torch_rand_float(
                *base_vel_range["y"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 9:10] = torch_rand_float(
                *base_vel_range["z"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 10:11] = torch_rand_float(
                *base_vel_range["roll"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 11:12] = torch_rand_float(
                *base_vel_range["pitch"],
                (len(env_ids), 1),
                device=self.device
            )
            self.root_states[env_ids, 12:13] = torch_rand_float(
                *base_vel_range["yaw"],
                (len(env_ids), 1),
                device=self.device
            )
        else:
            raise NameError(f"Unknown base_vel_range type: {type(base_vel_range)}")

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. (瞬时的)
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy  # 获取推动env的最大线速度 [1m/s]
        # 给 base 的xy方向线速度 上再添加随机 速度
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_reward_curriculum(self, current_iter):
        for i in range(len(self.cfg.rewards.reward_curriculum_schedule)):
            percentage = (current_iter - self.cfg.rewards.reward_curriculum_schedule[i][0]) / \
                         (self.cfg.rewards.reward_curriculum_schedule[i][1] - self.cfg.rewards.reward_curriculum_schedule[i][0])
            percentage = max(min(percentage, 1), 0)
            self.reward_curriculum_coef[i] = (1 - percentage) * self.cfg.rewards.reward_curriculum_schedule[i][2] + \
                                          percentage * self.cfg.rewards.reward_curriculum_schedule[i][3]

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        # [-30, 30] N
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance  # 给 base 添加随机扰动力
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        linear_tracking_ready = (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        )
        if linear_tracking_ready:
            # [-2, 2] ==> [-1.0, 1.5]
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -self.cfg.commands.max_backward_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., self.cfg.commands.max_forward_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.1, -self.cfg.commands.max_lat_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.1, 0., self.cfg.commands.max_lat_curriculum)

        max_yaw_curriculum = getattr(self.cfg.commands, "max_yaw_curriculum", None)
        if max_yaw_curriculum is not None and "tracking_ang_vel" in self.episode_sums:
            angular_tracking_ready = (
                torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length
                > 0.8 * self.reward_scales["tracking_ang_vel"]
            )
            if angular_tracking_ready:
                self.command_ranges["ang_vel_yaw"][0] = np.clip(
                    self.command_ranges["ang_vel_yaw"][0] - 0.1,
                    -max_yaw_curriculum,
                    0.0,
                )
                self.command_ranges["ang_vel_yaw"][1] = np.clip(
                    self.command_ranges["ang_vel_yaw"][1] + 0.1,
                    0.0,
                    max_yaw_curriculum,
                )


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\
        command_obs_dim = 6 if getattr(self.cfg.commands, "observe_jump_commands", False) and self.cfg.commands.num_commands >= 6 else 3
        proprioceptive_obs_dim = 6 + command_obs_dim + 4 * self.num_actions + 4
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(proprioceptive_obs_dim + 187, device=self.device)
        else:
            noise_vec = torch.zeros(proprioceptive_obs_dim, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        dof_pos_start = 6 + command_obs_dim
        noise_vec[dof_pos_start:(dof_pos_start + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(dof_pos_start + self.num_actions):(dof_pos_start + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(dof_pos_start + 2 * self.num_actions):proprioceptive_obs_dim] = 0. # actions, previous actions, clock inputs
        if self.cfg.terrain.measure_heights:
            noise_vec[proprioceptive_obs_dim:(proprioceptive_obs_dim + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #noise_vec[232:] = 0
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # 从Isaac Gym仿真器中获取各种 state tensor
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # base的状态
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 关节状态
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 每个刚体的接触力
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 刚体（包含base和各部件的）的状态

        # 刷新这些张量以确保数据最新
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 将获取的原始张量包装成PyTorch张量
        # base的状态 (num_envs, 13)，[0:3] base的位置, [3:7] base的旋转四元数，[7:10] base的线速度，[10:13] base的角速度
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # 关节状态 (num_envs * num_dof, 2)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 当前 关节位置 (num_env, 12, 1)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 当前 关节速度 (num_env, 12, 1)
        self.base_quat = self.root_states[:, 3:7]  # base 的旋转四元数（世界坐标系）

        # 刚体（包含base和各部件的）的状态 (num_env, num_bodies 17, 3)，[0:3]是位置，[7:10]是线速度
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # 四足的 位置 和 线速度（世界坐标系）
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # 存储每个刚体在xyz方向的接触力，(num_envs, num_bodies 17, 3)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)

        # initialize some data used later on
        # 初始化计数器、额外数据、重力向量等
        self.common_step_counter = 0  # 步数计数器
        self.extras = {}  # 额外数据字典
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))  # [0., 0., -1.]: 重力轴方向
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))    # 机器人的前进方向（base坐标系）
        self.episode_start_pos = self.root_states[:, :3].clone()
        self.episode_start_forward = quat_apply(self.base_quat, self.forward_vec)
        self.episode_start_forward[:, 2] = 0.
        self.episode_start_forward = self.episode_start_forward / torch.clamp(
            torch.norm(self.episode_start_forward, dim=1, keepdim=True),
            min=1e-6,
        )
        # 初始化 torques
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # (num_envs, 12)
        # 初始化 PD增益
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # P增益 (num_envs, 12)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # D增益 (num_envs, 12)
        # 初始化 actions，四肢的关节位置（按腿的顺序：FL, FR, RL, RR），(num_envs, 12)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])  # base的 线速度 + 角速度

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading/jump commands
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.jump_commands_scale = torch.tensor(
            [
                getattr(self.obs_scales, "jump_distance", self.obs_scales.lin_vel),
                getattr(self.obs_scales, "jump_distance", self.obs_scales.lin_vel),
                getattr(self.obs_scales, "jump_flag", 1.0),
            ],
            device=self.device,
            requires_grad=False,
        )
        self.sudden_stop_time_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.sudden_stop_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.sudden_stop_seen_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.sudden_stop_saved_commands = torch.zeros(
            self.num_envs,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.jump_assist_used = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.command_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.landing_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_forward_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_lateral_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.lateral_pair_touchdown_armed = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        self.lateral_pair_takeoff_armed = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        self.lateral_pair_touchdown_seen_in_stance = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        self.lateral_pair_prev_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.lateral_pair_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.lateral_pair_last_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.lateral_pair_touchdown_cooldown = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.lateral_pair_contact_duration = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.lateral_pair_prev_contact_duration = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_termination_buf = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.base_pose = self.root_states[:, 0:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.last_projected_gravity = self.projected_gravity.clone()

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = self._get_heights()

        else:  # 未启用高度测量
            self.num_height_points = 0  # 保持一致的属性存在
            self.measured_heights = torch.zeros(self.num_envs, 0, device=self.device, requires_grad=False)  # 空高度张量
        
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # motor_strength
        self.motor_strength = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        if getattr(self.cfg.domain_rand, "randomize_motor_strength", False):
            mtr_rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 给各刚体施加的 干扰力
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = self._sample_com_displacement(self.num_envs)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_custom_buffers__(self):
        # # domain randomization properties
        # self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
        #                                                           requires_grad=False)
        # self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
        #                                                           requires_grad=False)
        # self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
        #                                      requires_grad=False)
        # self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
        #                                   requires_grad=False)
        # self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
        #                                  requires_grad=False)
        # self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
        #                              requires_grad=False)
        # self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
        #                              requires_grad=False)
        # self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
        #                              requires_grad=False)
        # self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
        #     (self.num_envs, 1))

        # # if custom initialization values were passed in, set them here
        # dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
        #                    "Kp_factors", "Kd_factors"]
        # if self.initial_dynamics_dict is not None:
        #     for k, v in self.initial_dynamics_dict.items():
        #         if k in dynamics_params:
        #             setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

        # WTW gait-related buffers (always initialize to avoid AttributeError
        # when gait rewards are enabled but observe_gait_commands is False).
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_footswing_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # 从所有奖励函数中 去除 env_cfg 中 rewards.scales 为 0 的项
        # 非0的各奖励函数的 scales * self.dt (0.02)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)  # reward 名称列表，没有 _reward_前缀
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # 对应奖励函数 在当前episode内的 对应env的 奖励之和（为 每个scale非0的 奖励函数 创建一个 (num_env,) tensor）
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        # 总地形网格（水平方向）的 个数 (10 * 80 + 2 * 150, 20 * 80 + 2 * 150)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        # 包含刚体属性的列表，即包含机器狗的每个部位，[0]通常表示base
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        # ['base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        shank_names = [s for s in body_names if "SHANK" in s]

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins() # 获取每个env初始化时在地形中的位置 = 对应子地形的中心位置 (num_envs, 3)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self._init_custom_buffers__()

        # for domain randomization
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # 获取给env的质量 加减的范围
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        # 获取给base的 位置（xyz）加减的范围
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = self._sample_com_displacement(self.num_envs)

        # 创建每一个env
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()  # 该env的初始位置
            # (1) env初始位置随机 xy方向 随机加减1
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # (2) 为每个env生成一个随机摩擦系数、弹性系数
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # (3) 计算关节的属性限制（位置、速度、力矩）
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            # (4) 随机更改env的 base质量、质心偏移、其他刚体部位质量
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass

            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.shank_indices = torch.zeros(
            len(shank_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i, shank_name in enumerate(shank_names):
            self.shank_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], shank_name
            )

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)  # (num_envs, 8)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            # 在 [0, max_init_level + 1] 范围中随机生成每个env的 初始地形等级 (num_env,)
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            # env 平均分布在每列地形中 的编号 (num_envs / num_cols,)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            # 各子地形的中心位置 (num_rows, num_cols, 3)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # 每个env初始化时在地形中的位置 = 对应子地形的中心位置 (num_envs, 3)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # policy步长（env_step步长） = 0.02 = 4 * 0.005(物理仿真步长)
        self.obs_scales = self.cfg.normalization.obs_scales  # 各观测值的 缩放系数 (2.0, 0.25, 1.0, 0.05, 5.0)
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)  # 各奖励项的 缩放系数
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)  # 各 command 的 范围
        # 非网格地形，则禁用 课程学习
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s  # 20s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # 20s / 0.02 = 1000 steps

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 16s / 0.02 = 800 steps

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if (not hasattr(self, "terrain")) or (not self.terrain.cfg.measure_heights):
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()  # 9 * 7 = 63
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        num_envs = self.num_envs if env_ids is None else len(env_ids)
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(
                num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()  # 四足的 高度 (num_envs, 4, 1)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()  # 四足的位置 (num_envs, 4, 3)

        # 测量 四足位置下方的 地形高度
        points += self.terrain.cfg.border_size  # + 边界的偏移 25
        points = (points / self.terrain.cfg.horizontal_scale).long()  # / 0.1，归一化到地形网格坐标
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        ground_heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale  # 地形高度 转换为 实际的米单位 (num_evns, 4)

        feet_height = self.feet_pos[:, :, 2] - ground_heights  # 四足相对地形的 高度

        return feet_height

    def create_warp_env(self):
        terrain_mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        # save terrain mesh
        transform = np.zeros((3,))
        transform[0] = -self.terrain.cfg.border_size
        transform[1] = -self.terrain.cfg.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)

        if self.cfg.lidar.consider_self_occlusion:
            # add obstacles for self-occlusion
            robots_resource_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "robots", "aliengo")
            robot_path = os.path.join(robots_resource_dir, "robot_combined.stl")

            robot_mesh = trimesh.load(robot_path)
            transaltion = np.zeros((3,))
            transaltion[0] = self.root_states[0, 0]
            transaltion[1] = self.root_states[0, 1]
            transaltion[2] = self.root_states[0, 2]
            translation = trimesh.transformations.translation_matrix(transaltion)
            robot_mesh.apply_transform(translation)

            combined_mesh = trimesh.util.concatenate([terrain_mesh, robot_mesh])
            # save combined mesh
            combined_mesh.export(os.path.join(robots_resource_dir, "robot_terrain_combined.stl"))
        else:
            combined_mesh = terrain_mesh

        vertices = combined_mesh.vertices
        triangles = combined_mesh.faces
        vertex_tensor = torch.tensor(
            vertices,
            device=self.device,
            requires_grad=False,
            dtype=torch.float32,
        )

        # if none type in vertex_tensor
        if vertex_tensor.any() is None:
            print("vertex_tensor is None")
        vertex_vec3_array = warp.from_torch(vertex_tensor, dtype=warp.vec3)
        faces_warp_int32_array = warp.from_numpy(triangles.flatten(), dtype=warp.int32, device=self.device)

        self.warp_meshes = warp.Mesh(points=vertex_vec3_array, indices=faces_warp_int32_array)
        self.mesh_ids = warp.array([self.warp_meshes.id], dtype=warp.uint64)

    def create_warp_tensor(self):
        self.warp_tensor_dict = {}
        # (num_envs, num_sensors, num_ver_line, num_hor_line, 3)
        self.lidar_tensor = torch.zeros(
            (
                self.num_envs,
                self.lidar_cfg.num_sensors,  # 1
                self.lidar_cfg.vertical_line_num,  # 50
                self.lidar_cfg.horizontal_line_num,  # 80
                3,  # 3
            ),
            device=self.device,
            requires_grad=False,
        )
        # (num_envs, num_sensors, num_ver_line, num_hor_line)
        self.lidar_dist_tensor = torch.zeros(
            (
                self.num_envs,
                self.lidar_cfg.num_sensors,  # 1
                self.lidar_cfg.vertical_line_num,  # 50
                self.lidar_cfg.horizontal_line_num,  # 80
            ),
            device=self.device,
            requires_grad=False,
        )

        self.lidar_pos_tensor = torch.zeros_like(self.root_states[:, 0:3])
        self.lidar_quat_tensor = torch.zeros_like(self.root_states[:, 3:7])

        self.lidar_translation = torch.tensor(self.lidar_cfg.nominal_position, device=self.device).repeat((self.num_envs, 1))
        rpy_offset = torch.tensor(self.lidar_cfg.nominal_orientation_euler_deg, device=self.device)
        self.lidar_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))

        self.warp_tensor_dict["lidar_dist_tensor"] = self.lidar_dist_tensor
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.lidar_cfg.num_sensors
        self.warp_tensor_dict['lidar_pos_tensor'] = self.lidar_pos_tensor
        self.warp_tensor_dict['lidar_quat_tensor'] = self.lidar_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids

    def draw_lidar_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines

        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))

        if self.lidar_cfg.pointcloud_in_world_frame:
            self.global_pixels = self.downsampled_lidar_cloud
            for i in range(self.selected_env_idx, self.selected_env_idx + 1):
                for j in range(int(self.global_pixels.shape[2])):
                    for k in range(self.global_pixels.shape[3]):
                        x = self.global_pixels[i, 0, j, k, 0]  # +self.root_states[:1, 0]
                        y = self.global_pixels[i, 0, j, k, 1]
                        z = self.global_pixels[i, 0, j, k, 2]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        else:
            self.local_pixels_downsampled = self.downsampled_lidar_cloud.reshape(-1, 3)
            self.lidar_axis = self.lidar_pos_tensor[:, :]
            pixels = self.local_pixels_downsampled.view(self.num_envs, -1, 3)
            pixels_num = pixels.shape[1]
            lidar_axis_shaped = self.lidar_axis.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 3)
            lidar_quat = self.lidar_quat_tensor.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 4)
            self.global_pixels = lidar_axis_shaped + quat_apply(lidar_quat, pixels)

            self.global_pixels.view(self.num_envs, -1, 3)
            for i in range(self.selected_env_idx, self.selected_env_idx + 1):
                for j in range(0, self.global_pixels.shape[1]):
                    x = self.global_pixels[i, j, 0]
                    y = self.global_pixels[i, j, 1]
                    z = self.global_pixels[i, j, 2]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    # ------------ reward functions ------------
    def _reward_tracking_lin_vel(self):
        # 奖励 跟踪 commands 中XY方向的 线速度 (>= 0.1m/s时)
        if self._single_jump_flag_enabled():
            target_vel = torch.zeros_like(self.commands[:, :2])
            jump_active = self._single_jump_flag() & (~self.has_jumped)
            jump_time = max(getattr(self.cfg.rewards, "jump_tracking_lin_vel_time_s", 0.75), 1e-6)
            target_vel[jump_active] = self.commands[jump_active, 3:5] / jump_time
            max_vel = getattr(self.cfg.rewards, "jump_tracking_lin_vel_max", 2.0)
            target_vel = torch.clamp(target_vel, min=-max_vel, max=max_vel)
            lin_vel_error = torch.sum(torch.square(target_vel - self.base_lin_vel[:, :2]), dim=1)
            return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

        small_commands = torch.norm(self.commands[:, :2], dim=1) < 0.2
        track_commands = self.commands[:, :2] * (~small_commands.unsqueeze(-1))
        lin_vel_error = torch.sum(
            torch.square(track_commands - self.base_lin_vel[:, :2]),
            dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # 奖励 跟踪 commands 中yaw方向角速度
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _straight_command_weight(self):
        min_speed = getattr(self.cfg.rewards, "straight_min_forward_speed", 0.25)
        full_speed = getattr(self.cfg.rewards, "straight_full_forward_speed", 0.6)
        cmd_y_threshold = getattr(self.cfg.rewards, "straight_cmd_y_threshold", 0.15)
        cmd_yaw_threshold = getattr(self.cfg.rewards, "straight_cmd_yaw_threshold", 0.15)

        forward_weight = torch.clamp(
            (self.commands[:, 0] - min_speed) / max(full_speed - min_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        cmd_y_weight = torch.clamp(
            (cmd_y_threshold - torch.abs(self.commands[:, 1])) / max(cmd_y_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        cmd_yaw_weight = torch.clamp(
            (cmd_yaw_threshold - torch.abs(self.commands[:, 2])) / max(cmd_yaw_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        return forward_weight * cmd_y_weight * cmd_yaw_weight

    def _reward_straight_lin_vel_y(self):
        return torch.square(self.base_lin_vel[:, 1]) * self._straight_command_weight()

    def _reward_straight_yaw_vel(self):
        return torch.square(self.base_ang_vel[:, 2]) * self._straight_command_weight()

    def _reward_forward_pitch(self):
        # Only penalize nose-down pitch during forward straight walking.
        # In this convention projected_gravity[:, 0] grows positive when the base pitches forward.
        deadband = getattr(self.cfg.rewards, "forward_pitch_deadband", 0.04)
        pitch_direction = getattr(self.cfg.rewards, "forward_pitch_direction", 1.0)
        forward_pitch = self.projected_gravity[:, 0] * pitch_direction
        pitch_excess = torch.clamp(forward_pitch - deadband, min=0.0)
        return torch.square(pitch_excess) * self._straight_command_weight()

    def _reward_forward_backward_orientation(self):
        # Combined forward-fall and backward-fall guard. This keeps the old
        # forward_pitch/backward_orientation behavior in one reward term.
        deadband = getattr(self.cfg.rewards, "forward_pitch_deadband", 0.04)
        pitch_direction = getattr(self.cfg.rewards, "forward_pitch_direction", 1.0)
        forward_pitch = self.projected_gravity[:, 0] * pitch_direction
        forward_pitch_excess = torch.clamp(forward_pitch - deadband, min=0.0)
        nose_up_weight = getattr(self.cfg.rewards, "forward_nose_up_weight", 0.0)
        nose_up_excess = torch.clamp(-forward_pitch - deadband, min=0.0)
        forward_error = (
            torch.square(forward_pitch_excess)
            + nose_up_weight * torch.square(nose_up_excess)
        ) * self._straight_command_weight()

        min_speed = getattr(self.cfg.rewards, "backward_orientation_min_speed", 0.05)
        full_speed = getattr(self.cfg.rewards, "backward_orientation_full_speed", 0.35)
        backward_weight = torch.clamp(
            (-self.commands[:, 0] - min_speed) / max(full_speed - min_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        backward_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * backward_weight
        return forward_error + backward_error

    def _reward_feet_air_time(self):
        # 奖励 四足的空中时间接近0.5s (原地不动时除外)
        # 需过滤接触力信号，因为PhysX引擎在复杂地形上接触力检测不可靠
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.  # 检测z轴力 > 1N 的接触
        contact_filt = torch.logical_or(contact, self.last_contacts)  # 当前帧和上一帧的 有1次触地即可
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt  # 只考虑从空中首次触地的情况
        self.feet_air_time += self.dt  # 累加 policy 步长（0.02s）
        air_time_error = -torch.abs(self.feet_air_time - self.cfg.rewards.feet_air_time_target)
        rew_airTime = torch.sum(air_time_error * first_contact, dim=1)  # 仅奖励第一次触地，按配置的目标空中时间计算偏差
        condition = (torch.norm(self.commands[:, :2], dim=1) > 0.2) | (
                    torch.abs(self.commands[:, 2]) > 0.05)  # commands XY方向线速度 > 0.1m/s 或 yaw方向角速度 > 0.05rad/s 时才奖励
        rew_airTime *= condition.float()
        self.feet_air_time *= ~contact_filt  # 当前帧 触地的足 空中时间清0
        return rew_airTime
    
    def _reward_feet_air_time_variance_velocity(self):
        """
        惩罚四条腿 air time 不一致（步态节奏不均匀）
        只在运动时生效
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time * first_contact

        variance = torch.var(air_time, dim=1)
        
        condition = (torch.norm(self.commands[:, :2], dim=1) > 0.2) | (
                    torch.abs(self.commands[:, 2]) > 0.05)  # commands XY方向线速度 > 0.1m/s 或 yaw方向角速度 > 0.05rad/s 时才奖励
        # ------------------- 最终 reward（注意是 penalty） -------------------
        reward = variance * condition.float()
        # ------------------- reset air time（触地清零） -------------------
        self.feet_air_time *= ~contact_filt
        return reward

    def _reward_upward(self):
        # 奖励 重力投影向下
        return 1 - self.projected_gravity[:, 2]

    def _reward_has_contact(self):
        # 奖励 (base 原地不动) 时的 四足触地个数
        contact_filt = 1. * self.contact_filt
        condition = (torch.norm(self.commands[:, :2], dim=1) < 0.2) & (torch.abs(self.commands[:, 2]) < 0.05)
        return condition.float() * torch.sum(contact_filt, dim=-1) / 4

    # ------------ penalty functions ------------
    def _reward_lin_vel_z(self):
        # 惩罚 base 的 Z 轴线速度（防止跳跃）
        return torch.square(self.base_lin_vel[:, 2])
    def _reward_lin_vel_z_up(self):
        return torch.square(self.base_lin_vel[:, 2]) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)
    
    def _reward_lin_vel_z_abs(self):
        return torch.exp(torch.abs(self.base_lin_vel[:, 2]))
    
    def _reward_ang_vel_xy(self):
        # 惩罚 base 的 roll, pitch 轴角速度, 防止翻滚
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    def _reward_ang_vel_xy_up(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_ang_vel_xy_exp_abs(self):
        return torch.exp(-torch.norm(torch.abs(self.base_ang_vel[:, :2]), dim=1))
    
    def _reward_orientation(self):
        # 惩罚 base 非水平姿态
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    def _reward_backward_orientation(self):
        # 后退时额外抑制身体俯仰/横滚，避免高站姿下向后仰倒。
        min_speed = getattr(self.cfg.rewards, "backward_orientation_min_speed", 0.05)
        full_speed = getattr(self.cfg.rewards, "backward_orientation_full_speed", 0.35)
        backward_weight = torch.clamp(
            (-self.commands[:, 0] - min_speed) / max(full_speed - min_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * backward_weight
    def _zero_command_weight(self):
        lin_threshold = getattr(self.cfg.rewards, "zero_command_lin_vel_threshold", 0.25)
        yaw_threshold = getattr(self.cfg.rewards, "zero_command_yaw_vel_threshold", 0.20)
        lin_weight = torch.clamp(
            (lin_threshold - torch.norm(self.commands[:, :2], dim=1)) / max(lin_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        yaw_weight = torch.clamp(
            (yaw_threshold - torch.abs(self.commands[:, 2])) / max(yaw_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        return lin_weight * yaw_weight

    def _reward_stop_orientation(self):
        # 零速/急停时额外压住身体点头和侧倾，减少从站立切入RL或刹停时的前倾。
        pitch_weight = getattr(self.cfg.rewards, "stop_pitch_weight", 2.0)
        roll_weight = getattr(self.cfg.rewards, "stop_roll_weight", 1.0)
        tilt_error = (
            pitch_weight * torch.square(self.projected_gravity[:, 0])
            + roll_weight * torch.square(self.projected_gravity[:, 1])
        )
        return tilt_error * self._zero_command_weight()
    def _reward_stop_ang_vel_xy(self):
        # 零速/急停时抑制 base roll/pitch 角速度，避免停下瞬间身体继续往前点。
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self._zero_command_weight()
    def _reward_stand_base_height(self):
        # Zero-command stance should not solve stability by crouching. Only
        # penalize being below the target so a slightly taller stance is allowed.
        base_height = self._get_base_heights()
        target = getattr(self.cfg.rewards, "stand_base_height_target", self.cfg.rewards.base_height_target)
        low_error = torch.clamp(target - base_height, min=0.0)
        return torch.square(low_error) * self._zero_command_weight()
    def _reward_orientation_up(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_orientation_exp(self):
        return torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1)*10)
    
    def _reward_base_height(self):
        # 惩罚 base 偏离目标高度
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    def _moving_command_weight(self):
        lin_cmd = torch.norm(self.commands[:, :2], dim=1)
        yaw_cmd = torch.abs(self.commands[:, 2])
        min_lin = getattr(self.cfg.rewards, "moving_height_min_lin_cmd", 0.05)
        full_lin = getattr(self.cfg.rewards, "moving_height_full_lin_cmd", 0.35)
        min_yaw = getattr(self.cfg.rewards, "moving_height_min_yaw_cmd", 0.05)
        full_yaw = getattr(self.cfg.rewards, "moving_height_full_yaw_cmd", 0.4)
        lin_weight = torch.clamp((lin_cmd - min_lin) / max(full_lin - min_lin, 1e-6), min=0.0, max=1.0)
        yaw_weight = torch.clamp((yaw_cmd - min_yaw) / max(full_yaw - min_yaw, 1e-6), min=0.0, max=1.0)
        return torch.maximum(lin_weight, yaw_weight)
    def _reward_moving_base_height(self):
        # During gait the policy may crouch to satisfy contact rewards. Only penalize
        # being below the moving target so standing height and occasional tall steps stay free.
        base_height = self._get_base_heights()
        target = getattr(self.cfg.rewards, "moving_base_height_target", self.cfg.rewards.base_height_target)
        low_error = torch.clamp(target - base_height, min=0.0)
        return torch.square(low_error) * self._moving_command_weight()
    def _reward_moving_orientation_flat(self):
        # Keep the base level while moving. This is symmetric in pitch, so it
        # still works if the projected_gravity pitch sign differs between sims.
        pitch_deadband = getattr(self.cfg.rewards, "moving_orientation_pitch_deadband", 0.02)
        roll_deadband = getattr(self.cfg.rewards, "moving_orientation_roll_deadband", 0.03)
        pitch_weight = getattr(self.cfg.rewards, "moving_orientation_pitch_weight", 2.0)
        roll_weight = getattr(self.cfg.rewards, "moving_orientation_roll_weight", 1.0)
        pitch_error = torch.clamp(torch.abs(self.projected_gravity[:, 0]) - pitch_deadband, min=0.0)
        roll_error = torch.clamp(torch.abs(self.projected_gravity[:, 1]) - roll_deadband, min=0.0)
        return (
            pitch_weight * torch.square(pitch_error)
            + roll_weight * torch.square(roll_error)
        ) * self._moving_command_weight()
    def _reward_base_height_up(self):
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_base_height_exp(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)*10)*(torch.norm(self.commands[:, :2], dim=1) < 0.2)

    def _reward_base_height_vel(self):
        base_height = self._get_base_heights()
        base_lin_vel = torch.norm(self.base_lin_vel, dim=1)
        return base_lin_vel * torch.square(base_height - self.cfg.rewards.base_height_target_vel)


    # --- dof velocity ---
    def _reward_dof_vel(self):
        # 惩罚 关节速度
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # 惩罚 关节加速度
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_acc_dt(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel)), dim=1)
    
    def _reward_dof_vel_limits(self):
        # 惩罚 关节速度接近极限
        # 裁剪至 max error = 每个关节 1 rad/s，以避免 巨大惩罚
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # --- dof position ---
    def _reward_dof_pos_dif(self):
        # 惩罚 关节位置 的变化
        return torch.sum(torch.square(self.last_dof_pos - self.dof_pos), dim=1)

    def _reward_dof_pos_limits(self):
        # 惩罚 关节位置接近极限
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # --- actions ---
    def _reward_action_rate(self):
        # 惩罚 action 的变化（使机器人运动更加平滑连续）
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # 惩罚 action 的二阶平滑性（使动作更加平缓）
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)

    # --- torques ---
    def _reward_torques(self):
        # 惩罚 关节扭矩过大（防止关节过热或损坏）
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torques_distribution(self):
        # 惩罚 关节扭矩分布不均
        return torch.var(torch.abs(self.torques), dim=1)

    def _reward_torques_dif(self):
        # 惩罚 关节扭矩的变化
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_torque_limits(self):
        # 惩罚 关节扭矩接近极限
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tn_curve(self):
        """Penalize joints outside the motor torque-speed envelope."""
        if not hasattr(self, "_tn_curve_torque_limits"):
            torque_limits_cfg = getattr(self.cfg.rewards, "tn_curve_torque_limits", {})
            speed_limits_cfg = getattr(self.cfg.rewards, "tn_curve_speed_limits", {})
            soft_ratio = getattr(self.cfg.rewards, "tn_curve_soft_ratio", 1.0)

            torque_limits = []
            speed_limits = []
            for dof_name in self.dof_names:
                if "HipX" in dof_name:
                    joint_type = "hipx"
                elif "HipY" in dof_name:
                    joint_type = "hipy"
                elif "Knee" in dof_name:
                    joint_type = "knee"
                else:
                    joint_type = ""

                torque_limits.append(float(torque_limits_cfg.get(joint_type, 1e6)) * soft_ratio)
                speed_limits.append(float(speed_limits_cfg.get(joint_type, 1e6)) * soft_ratio)

            self._tn_curve_torque_limits = torch.tensor(
                torque_limits, dtype=torch.float, device=self.device
            ).unsqueeze(0)
            self._tn_curve_speed_limits = torch.tensor(
                speed_limits, dtype=torch.float, device=self.device
            ).unsqueeze(0)

        abs_speed = torch.abs(self.dof_vel)
        abs_torque = torch.abs(self.torques)
        speed_limits = torch.clamp(self._tn_curve_speed_limits, min=1e-6)
        torque_limits = torch.clamp(self._tn_curve_torque_limits, min=1e-6)

        speed_ratio = abs_speed / speed_limits
        allowed_torque = torque_limits * torch.clamp(1.0 - speed_ratio, min=0.0, max=1.0)
        torque_excess = torch.clamp((abs_torque - allowed_torque) / torque_limits, min=0.0)
        speed_excess = torch.clamp(speed_ratio - 1.0, min=0.0)
        return torch.mean(torch.square(torque_excess) + torch.square(speed_excess), dim=1)

    # --- power ---
    def _reward_joint_power(self):
        # 惩罚 高功率
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_power(self):
        # 惩罚 关节功率消耗（扭矩 * 关节速度）
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_power_distribution(self):
        # 惩罚 关节功率消耗分布不均
        return torch.var(torch.abs(self.torques * self.dof_vel), dim=1)

    # --- collision, termination
    def _reward_collision(self):
        # 惩罚 指定关节的碰撞
        # 当指定关节接触力的 模 > 0.1N，则判定发生碰撞，计为 1
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    def _reward_collision_up(self):
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_termination(self):
        # Terminal reward / penalty
        rewards = self.reset_buf * ~self.time_out_buf
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "out_of_border", False):
            rewards * ~self.out_border
        if hasattr(self.cfg, "termination") and getattr(self.cfg.termination, "fall_down", False):
            rewards * ~self.fall_down
        return rewards

    # --- feet contact ---
    def _reward_feet_contact_forces(self):
        # 惩罚 四足接触力过大（需<100）
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_feet_stumble(self):
        # 惩罚 四足接触到垂直表面 (只在上楼梯，discrete_obstacle, pit地形)
        # 判定条件： XY方向 足部接触力 与 Z轴接触力 之比 > 5
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
             4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew * (self.terrain_levels > 3)
        rew = rew.float()
        stumble_reward = torch.zeros_like(rew)
        stumble_reward[self.stairsup_start_idx: self.stairsup_end_idx] = rew[self.stairsup_start_idx: self.stairsup_end_idx]
        stumble_reward[self.discreteobstacles_start_idx: self.discreteobstacles_end_idx] = rew[self.discreteobstacles_start_idx: self.discreteobstacles_end_idx]
        stumble_reward[self.pit_start_idx: self.gap_end_idx] = rew[self.pit_start_idx: self.gap_end_idx]
        return stumble_reward
    def _reward_feet_stumble_up(self):
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
             4 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        rew = rew * (self.terrain_levels > 3)
        rew = rew.float()
        stumble_reward = torch.zeros_like(rew)
        stumble_reward[self.stairsup_start_idx: self.stairsup_end_idx] = rew[self.stairsup_start_idx: self.stairsup_end_idx]
        stumble_reward[self.discreteobstacles_start_idx: self.discreteobstacles_end_idx] = rew[self.discreteobstacles_start_idx: self.discreteobstacles_end_idx]
        stumble_reward[self.pit_start_idx: self.gap_end_idx] = rew[self.pit_start_idx: self.gap_end_idx]
        return stumble_reward * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_feet_slide(self):
        # 惩罚 触地时 四足相对base的速度（避免滑动）
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)  # 当前四足相对base的 线速度（世界坐标系）
        # 当前四足相对base的 线速度（body坐标系）
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        # 四足相对base的 线速度 的模
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(self.contact_filt * foot_leteral_vel, dim=1)
    def _reward_feet_slide_up(self):
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(self.contact_filt * foot_leteral_vel, dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_feet_contact_forces(self):
        # 惩罚 四足的接触力 > 100N
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_soft_landing(self):
        contact_threshold = getattr(self.cfg.rewards, "soft_landing_contact_threshold", 1.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > contact_threshold
        first_contact = contact & ~self.prev_contacts
        foot_ids = getattr(self.cfg.rewards, "soft_landing_foot_ids", None)
        if foot_ids is not None:
            if len(foot_ids) == 0:
                return torch.zeros(self.num_envs, device=self.device)
            first_contact = first_contact[:, foot_ids]

        max_z_vel = getattr(self.cfg.rewards, "soft_landing_max_z_vel", 0.25)
        max_force = getattr(self.cfg.rewards, "soft_landing_max_force", 80.0)
        force_weight = getattr(self.cfg.rewards, "soft_landing_force_weight", 0.25)

        down_vel_excess = torch.clamp(-self.feet_vel[:, :, 2] - max_z_vel, min=0.0)
        force_excess = torch.clamp(self.contact_forces[:, self.feet_indices, 2] - max_force, min=0.0) / max(max_force, 1e-6)
        impact = torch.square(down_vel_excess) + force_weight * torch.square(force_excess)
        if foot_ids is not None:
            impact = impact[:, foot_ids]
        return torch.sum(first_contact.float() * impact, dim=1)

    def _reward_feet_mirror(self):
        # 惩罚 斜对称腿的关节位置偏差
        diff1 = torch.sum(torch.square(self.dof_pos[:, [1, 2]] - self.dof_pos[:, [10, 11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:, [4, 5]] - self.dof_pos[:, [7, 8]]),dim=-1)
        return 0.5 * (diff1 + diff2)
    def _reward_feet_mirror_up(self):
        diff1 = torch.sum(torch.square(self.dof_pos[:, [1, 2]] - self.dof_pos[:, [10, 11]]),dim=-1)
        diff2 = torch.sum(torch.square(self.dof_pos[:, [4, 5]] - self.dof_pos[:, [7, 8]]),dim=-1)
        return 0.5 * (diff1 + diff2) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    # --- stand, stuck ---
    def _tracking_relief_scale(self):
        if not getattr(self.cfg.rewards, "tracking_relief_enabled", False):
            return torch.ones(self.num_envs, device=self.device)

        tilt = torch.norm(self.projected_gravity[:, :2], dim=1)
        start = getattr(self.cfg.rewards, "tracking_relief_tilt_start", 0.25)
        end = getattr(self.cfg.rewards, "tracking_relief_tilt_end", 0.55)
        min_scale = getattr(self.cfg.rewards, "tracking_relief_min_scale", 0.2)
        relief = torch.clamp((tilt - start) / max(end - start, 1e-6), min=0.0, max=1.0)
        return 1.0 - relief * (1.0 - min_scale)

    def _reward_stand_still(self):
        # 惩罚 (base原地不动 或 原地旋转) 时的 关节位置与默认关节位置的 偏差
        condition = torch.norm(self.commands[:, :3], dim=1) < 0.1  # commands 中XY方向线速度 < 0.1 m/s (不论角速度大小，都会受到惩罚)
        dof_deviation = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_deviation * condition.float() * self._tracking_relief_scale()

    def _reward_stand_dof_vel(self):
        # 零速站立时额外惩罚关节速度，抑制单腿来回摆动。
        condition = torch.norm(self.commands[:, :3], dim=1) < 0.1
        return torch.sum(torch.square(self.dof_vel), dim=1) * condition.float() * self._tracking_relief_scale()

    def _reward_stand_nice(self):
        # 惩罚 (base原地不动 或 原地旋转) 且 重力投影向下时 的 关节位置与默认关节位置的 偏差
        condition = (torch.norm(self.commands[:, :2], dim=1) < 0.2) * (1 - self.projected_gravity[:, 2])
        dof_deviation = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_deviation * condition.float() * self._tracking_relief_scale()

    def _reward_stuck(self):
        # 惩罚 卡住
        # 判断是否卡住：
        #   base 的 (XY方向线速度 < 0.1 m/s 且 yaw方向角速度 < 0.1 rad/s)
        small_lin_vel = torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.2
        small_ang_vel = torch.abs(self.base_ang_vel[:, 2]) < 0.2
        stuck = small_lin_vel & small_ang_vel
        #   但 commands 的 线速度 > 0.1 m/s 或 角速度 > 0.1 rad/s
        large_lin_commands = torch.norm(self.commands[:, :2], dim=1) > 0.2
        large_ang_commands = torch.abs(self.commands[:, 2]) > 0.2
        large_commands = large_lin_commands | large_ang_commands
        return stuck * large_commands

    # --- joint pose deviation ---
    def _reward_hip_action_magnitude(self):
        # 限制 action 中的 髋关节hip（0,3,6,9）动作幅度（防止 > 1.0）
        return torch.sum(torch.square(torch.maximum(torch.abs(self.actions[:, [0, 3, 6, 9]]) - 1.0,
                                                    torch.zeros_like(self.actions[:, [0, 3, 6, 9]]))), dim=1)

    def _reward_hip_pos(self):
        # 惩罚 hip关节（0,3,6,9）与默认位置的 偏差， (原地不动 或 原地旋转) 时惩罚系数为 5.0，其他为 1.0
        hip_deviation = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        #   XY方向线速度 < 0.1 m/s (不论角速度大小，都会受到惩罚) 时，惩罚力度为 5.0
        #   XY方向线速度 >= 0.1 m/s 时，惩罚力度为 1.0
        condition = torch.norm(self.commands[:, :3], dim=1) < 0.1
        multiplier = 1.0 + condition.float() * 4.0
        return hip_deviation * multiplier * self._tracking_relief_scale()
    
    def _reward_hip_pos_up(self):
        hip_deviation = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        condition = torch.norm(self.commands[:, :2], dim=1) < 0.2
        multiplier = 1.0 + condition.float() * 4.0
        return hip_deviation * multiplier * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * self._tracking_relief_scale()

    def _reward_thigh_pose(self):
        thigh_deviation = torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - self.default_dof_pos[:, [1, 4, 7, 10]]), dim=1)
        condition = torch.norm(self.commands[:, :3], dim=1) < 0.1
        multiplier = 1.0 + condition.float() * 4.0
        return thigh_deviation * multiplier * self._tracking_relief_scale()
    def _reward_thigh_pose_up(self):
        thigh_deviation = torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - self.default_dof_pos[:, [1, 4, 7, 10]]), dim=1)
        condition = torch.norm(self.commands[:, :2], dim=1) < 0.2
        multiplier = 1.0 + condition.float() * 4.0
        return thigh_deviation * multiplier * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * self._tracking_relief_scale()

    def _reward_calf_pose(self):
        calf_deviation = torch.sum(torch.abs(self.dof_pos[:, [2, 5, 8, 11]] - self.default_dof_pos[:, [2, 5, 8, 11]]), dim=1)
        condition = torch.norm(self.commands[:, :3], dim=1) < 0.1
        multiplier = 1.0 + condition.float() * 4.0
        return calf_deviation * multiplier * self._tracking_relief_scale()
    def _reward_calf_pose_up(self):
        calf_deviation = torch.sum(torch.abs(self.dof_pos[:, [2, 5, 8, 11]] - self.default_dof_pos[:, [2, 5, 8, 11]]), dim=1)
        condition = torch.norm(self.commands[:, :2], dim=1) < 0.2
        multiplier = 1.0 + condition.float() * 4.0
        return calf_deviation * multiplier * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * self._tracking_relief_scale()

    # --- 四足离地高度 ---
    def _reward_feet_clearance_base(self):
        # 惩罚 大速度下 四足抬脚距base的高度 偏离目标距离 （-0.2 m）（摔倒时不计算）
        # 当前四足相对base的 位置 和 线速度（世界坐标系）
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        # 当前四足相对base的 位置 和 线速度（body坐标系）
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)  # (num_envs, 4, 3)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])

        # 四足相对base的高度 距 目标高度 的误差（平方误差）
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.feet_height_target_base).view(self.num_envs, -1)
        condition = (torch.norm(self.commands[:, :2], dim=1) > 0.2) | (
                    torch.abs(self.commands[:, 2]) > 0.05)  # commands XY方向线速度 > 0.1m/s 或 yaw方向角速度 > 0.05rad/s 时才奖励 
        # 四足相对base的 线速度 的模
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * feet_leteral_vel, dim=1) * condition.float()
        return torch.sum(height_error, dim=1) * condition.float()

    # def _reward_foot_clearance(self):
    #     cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
    #     footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
    #     footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
    #     for i in range(len(self.feet_indices)):
    #         footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
    #         footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
    #     height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
    #     foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
    #     return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_feet_clearance_base_up(self):
        # 惩罚 大速度下 四足抬脚距base的高度 偏离目标距离 （-0.2 m）（摔倒时不计算）
        # 当前四足相对base的 位置 和 线速度（世界坐标系）
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        # 当前四足相对base的 位置 和 线速度（body坐标系）
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)  # (num_envs, 4, 3)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])

        # 四足相对base的高度 距 目标高度 的误差（平方误差）
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.feet_height_target_base).view(self.num_envs, -1)
        # 四足相对base的 线速度 的模
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * feet_leteral_vel, dim=1) * torch.clamp(-self.projected_gravity[:, 2], 0, 1)

    def _reward_feet_clearance_terrain(self):
        # 惩罚 大速度下（同时考虑线速度和角速度） 四足的抬脚高度 需接近 离地目标高度（0.15m）
        feet_heights = self._get_feet_heights()

        feet_lateral_vel = torch.norm(self.feet_vel[:, :, :2], dim=-1)
        # return torch.sum(foot_lateral_vel * torch.maximum(-feet_heights + self.cfg.rewards.feet_height_target_terrain, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(feet_lateral_vel * torch.square(feet_heights - self.cfg.rewards.feet_height_target_terrain), dim=-1)

    def _reward_feet_clearance_terrain_up(self):
        # 惩罚 大速度下 四足的抬脚高度 需接近 离地目标高度（0.15m）
        feet_heights = self._get_feet_heights()

        feet_lateral_vel = torch.norm(self.feet_vel[:, :, :2], dim=-1)
        # return torch.sum(foot_lateral_vel * torch.maximum(-feet_heights + self.cfg.rewards.feet_height_target_terrain, torch.zeros_like(foot_heights)), dim = -1)
        return torch.sum(feet_lateral_vel * torch.square(feet_heights - self.cfg.rewards.feet_height_target_terrain), dim=-1) * torch.clamp(
            -self.projected_gravity[:, 2], 0, 1)

    def _reward_feet_yaw_clearance_terrain(self):
        # 奖励 (base原地旋转) 时 脚抬起
        condition = (torch.abs(self.commands[:, 2]) > 0.05) & (torch.norm(self.commands[:, :2], dim=1) < 0.2)

        feet_heights = self._get_feet_heights()
        feet_heights = torch.clip(feet_heights,min=0,max = self.cfg.rewards.target_foot_height_yaw)
        mean_foot_height = torch.mean(feet_heights, dim=1)

        height_reward = torch.tanh(mean_foot_height + 0.15)
        return condition.float() * height_reward

#####################_reward_wtw###################################

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

    def _jump_motion_mask(self):
        if self._single_jump_flag_enabled():
            return self._single_jump_flag() & (~self.has_jumped)

        if getattr(self.cfg.commands, "single_jump_mode", False):
            return self._single_jump_active_mask()

        min_speed = getattr(self.cfg.rewards, "jump_min_command_speed", 0.2)
        min_yaw = getattr(self.cfg.rewards, "jump_min_yaw_speed", 0.05)
        return (torch.norm(self.commands[:, :2], dim=1) > min_speed) | (torch.abs(self.commands[:, 2]) > min_yaw)

    def _jump_contact_mask(self):
        threshold = getattr(self.cfg.rewards, "jump_contact_force_threshold", 5.0)
        return self.contact_forces[:, self.feet_indices, 2] > threshold

    def _jump_all_air_mask(self):
        return torch.logical_not(torch.any(self._jump_contact_mask(), dim=1))

    def _reward_jump(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self._jump_contact_mask()
        if getattr(self.cfg.rewards, "jump_sync_air_only", False):
            jump_sync = self._jump_all_air_mask()
        else:
            # Reward synchronous four-leg contact state: all stance or all flight.
            jump_sync = (contact[:, 0] == contact[:, 1]) & \
                        (contact[:, 1] == contact[:, 2]) & \
                        (contact[:, 2] == contact[:, 3])
        return jump_sync.float() * self._jump_motion_mask().float()

    def _reward_jump_air_time(self):
        """Per-step reward for keeping all feet airborne during jump commands."""
        return self._jump_all_air_mask().float() * self._jump_motion_mask().float()

    def _reward_jump_mixed_contact(self):
        contact_count = torch.sum(self._jump_contact_mask().float(), dim=1)
        mixed_contact = (contact_count > 0.0) & (contact_count < len(self.feet_indices))
        return mixed_contact.float() * self._jump_motion_mask().float()

    def _reward_jump_landing_async(self):
        """Penalize the first landing frame if all four feet do not touch down together."""
        contact = self._jump_contact_mask()
        prev_contact = self.prev_contacts
        was_all_air = torch.logical_not(torch.any(prev_contact, dim=1))
        touchdown = was_all_air & torch.any(contact, dim=1)
        contact_count = torch.sum(contact.float(), dim=1)
        missing_contact_fraction = torch.clamp(
            (float(len(self.feet_indices)) - contact_count) / float(len(self.feet_indices)),
            min=0.0,
            max=1.0,
        )
        return missing_contact_fraction * touchdown.float() * self._jump_motion_mask().float()

    def _reward_jump_hind_first_landing(self):
        """Penalize hind-feet-first touchdown during forward jumping."""
        contact = self._jump_contact_mask()
        was_all_air = torch.logical_not(torch.any(self.prev_contacts, dim=1))
        touchdown = was_all_air & torch.any(contact, dim=1)

        front_contact = torch.mean(contact[:, 0:2].float(), dim=1)
        hind_contact = torch.mean(contact[:, 2:4].float(), dim=1)
        hind_first = torch.clamp(hind_contact - front_contact, min=0.0)

        min_forward = getattr(self.cfg.rewards, "jump_forward_landing_min_command", 0.15)
        full_forward = getattr(self.cfg.rewards, "jump_forward_landing_full_command", 0.8)
        forward_weight = torch.clamp(
            (self.commands[:, 0] - min_forward) / max(full_forward - min_forward, 1e-6),
            min=0.0,
            max=1.0,
        )
        return hind_first * touchdown.float() * forward_weight * self._jump_motion_mask().float()

    def _reward_jump_no_flight(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, device=self.device)

        grace_s = getattr(self.cfg.rewards, "jump_no_flight_grace_s", 0.18)
        grace_steps = max(int(grace_s / self.dt), 1)
        late_after_trigger = self.episode_length_buf > (self.command_frame + grace_steps)
        active = self._single_jump_flag() & (~self.was_in_flight) & (~self.has_jumped) & late_after_trigger
        contact_fraction = torch.mean(self._jump_contact_mask().float(), dim=1)
        return contact_fraction * active.float()

    def _reward_jump_flight_phase_air(self):
        _, _, _, _, _, _, _, flight = self._jump_phase_masks()
        return self._jump_all_air_mask().float() * flight.float() * self._jump_motion_mask().float()

    def _reward_jump_flight_orientation(self):
        all_air = self._jump_all_air_mask().float()
        tilt_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return tilt_error * all_air * self._jump_motion_mask().float()

    def _reward_jump_flight_ang_vel_xy(self):
        all_air = self._jump_all_air_mask().float()
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * all_air * self._jump_motion_mask().float()

    def _reward_jump_flight_pitch_stable(self):
        """Suppress roll/pitch base rocking through both ascent and descent."""
        _, _, _, flight_up, flight_down, _, _, _ = self._jump_phase_masks()
        if getattr(self.cfg.rewards, "jump_flight_pitch_use_phase_mask", True):
            active = (flight_up | flight_down).float()
        else:
            active = self._jump_all_air_mask().float()

        pitch_deadband = getattr(self.cfg.rewards, "jump_flight_pitch_deadband", 0.04)
        roll_deadband = getattr(self.cfg.rewards, "jump_flight_roll_deadband", pitch_deadband)
        backward_pitch_deadband = getattr(
            self.cfg.rewards,
            "jump_flight_backward_pitch_deadband",
            pitch_deadband,
        )
        pitch_vel_deadband = getattr(self.cfg.rewards, "jump_flight_pitch_vel_deadband", 0.25)
        backward_pitch_vel_deadband = getattr(
            self.cfg.rewards,
            "jump_flight_backward_pitch_vel_deadband",
            pitch_vel_deadband,
        )
        backward_pitch_delta_deadband = getattr(
            self.cfg.rewards,
            "jump_flight_backward_pitch_delta_deadband",
            0.06,
        )
        roll_vel_deadband = getattr(self.cfg.rewards, "jump_flight_roll_vel_deadband", pitch_vel_deadband)
        pitch_weight = getattr(self.cfg.rewards, "jump_flight_pitch_weight", 1.0)
        roll_weight = getattr(self.cfg.rewards, "jump_flight_roll_weight", 0.0)
        backward_pitch_weight = getattr(self.cfg.rewards, "jump_flight_backward_pitch_weight", 0.0)
        pitch_vel_weight = getattr(self.cfg.rewards, "jump_flight_pitch_vel_weight", 0.35)
        backward_pitch_vel_weight = getattr(
            self.cfg.rewards,
            "jump_flight_backward_pitch_vel_weight",
            0.0,
        )
        backward_pitch_delta_weight = getattr(
            self.cfg.rewards,
            "jump_flight_backward_pitch_delta_weight",
            0.0,
        )
        apex_z_vel_window = getattr(
            self.cfg.rewards,
            "jump_flight_apex_z_vel_window",
            0.0,
        )
        apex_pitch_vel_extra_scale = getattr(
            self.cfg.rewards,
            "jump_flight_apex_pitch_vel_extra_scale",
            0.0,
        )
        roll_vel_weight = getattr(self.cfg.rewards, "jump_flight_roll_vel_weight", 0.0)
        backward_pitch_direction = getattr(self.cfg.rewards, "jump_flight_backward_pitch_direction", -1.0)
        backward_pitch_vel_direction = getattr(
            self.cfg.rewards,
            "jump_flight_backward_pitch_vel_direction",
            -1.0,
        )

        pitch_error = torch.clamp(torch.abs(self.projected_gravity[:, 0]) - pitch_deadband, min=0.0)
        roll_error = torch.clamp(torch.abs(self.projected_gravity[:, 1]) - roll_deadband, min=0.0)
        backward_pitch_error = torch.clamp(
            self.projected_gravity[:, 0] * backward_pitch_direction - backward_pitch_deadband,
            min=0.0,
        )
        pitch_vel_error = torch.clamp(torch.abs(self.base_ang_vel[:, 1]) - pitch_vel_deadband, min=0.0)
        backward_pitch_vel_error = torch.clamp(
            self.base_ang_vel[:, 1] * backward_pitch_vel_direction - backward_pitch_vel_deadband,
            min=0.0,
        )
        pitch_vel_delta = self.base_ang_vel[:, 1] - self.last_base_ang_vel[:, 1]
        backward_pitch_delta_error = torch.clamp(
            pitch_vel_delta * backward_pitch_vel_direction - backward_pitch_delta_deadband,
            min=0.0,
        )
        roll_vel_error = torch.clamp(torch.abs(self.base_ang_vel[:, 0]) - roll_vel_deadband, min=0.0)

        # The real apex can drift relative to the gait clock after sim-to-sim transfer.
        # Use world-frame vertical speed so pitch does not distort the apex detector.
        if apex_z_vel_window > 0.0 and apex_pitch_vel_extra_scale > 0.0:
            apex_weight = 1.0 + apex_pitch_vel_extra_scale * torch.clamp(
                1.0 - torch.abs(self.root_states[:, 9]) / apex_z_vel_window,
                min=0.0,
                max=1.0,
            )
        else:
            apex_weight = torch.ones(self.num_envs, device=self.device)

        stable_error = (
            pitch_weight * torch.square(pitch_error)
            + roll_weight * torch.square(roll_error)
            + backward_pitch_weight * torch.square(backward_pitch_error)
            + apex_weight * pitch_vel_weight * torch.square(pitch_vel_error)
            + apex_weight * backward_pitch_vel_weight * torch.square(backward_pitch_vel_error)
            + apex_weight * backward_pitch_delta_weight * torch.square(backward_pitch_delta_error)
            + roll_vel_weight * torch.square(roll_vel_error)
        )
        return stable_error * active * self._jump_motion_mask().float()

    def _reward_jump_height(self):
        """Bounded all-air reward that increases as base height reaches the configured target."""
        base_height = self._get_base_heights()
        start_height = getattr(self.cfg.rewards, "jump_height_min", self.cfg.rewards.base_height_target)
        target_height = getattr(self.cfg.rewards, "jump_height_target", start_height + 0.1)
        height_span = max(target_height - start_height, 1e-6)
        height_reward = torch.clamp((base_height - start_height) / height_span, min=0.0, max=1.0)
        return height_reward * self._jump_all_air_mask().float() * self._jump_motion_mask().float()

    def _reward_jump_z_vel(self):
        """Bounded all-air reward for upward base velocity during jump commands."""
        start_vel = getattr(self.cfg.rewards, "jump_z_vel_min", 0.0)
        target_vel = getattr(self.cfg.rewards, "jump_z_vel_target", start_vel + 0.5)
        vel_span = max(target_vel - start_vel, 1e-6)
        vel_reward = torch.clamp((self.base_lin_vel[:, 2] - start_vel) / vel_span, min=0.0, max=1.0)
        return vel_reward * self._jump_all_air_mask().float() * self._jump_motion_mask().float()

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

    def _reward_jump_takeoff_x_vel(self):
        _, _, push, _, _, _, _, _ = self._jump_phase_masks()
        start_vel = getattr(self.cfg.rewards, "jump_takeoff_x_vel_min", 0.2)
        target_vel = getattr(self.cfg.rewards, "jump_takeoff_x_vel_target", start_vel + 0.8)
        if getattr(self.cfg.rewards, "jump_takeoff_track_command_direction", False):
            command_xy = self.commands[:, :2]
            command_speed = torch.norm(command_xy, dim=1)
            command_dir = command_xy / torch.clamp(command_speed.unsqueeze(1), min=1e-6)
            takeoff_speed = torch.sum(self.base_lin_vel[:, :2] * command_dir, dim=1)
            dynamic_target = torch.clamp(command_speed, min=start_vel + 1e-3, max=target_vel)
            dynamic_start = torch.minimum(
                torch.full_like(dynamic_target, start_vel),
                0.5 * dynamic_target,
            )
            vel_reward = torch.clamp(
                (takeoff_speed - dynamic_start) / torch.clamp(dynamic_target - dynamic_start, min=1e-6),
                min=0.0,
                max=1.0,
            )
            vel_reward *= (command_speed > getattr(self.cfg.rewards, "jump_min_command_speed", 0.05)).float()
        else:
            vel_span = max(target_vel - start_vel, 1e-6)
            vel_reward = torch.clamp((self.base_lin_vel[:, 0] - start_vel) / vel_span, min=0.0, max=1.0)

        if getattr(self.cfg.rewards, "jump_takeoff_require_contact", True):
            contact_weight = torch.mean(self._jump_contact_mask().float(), dim=1)
        else:
            contact_weight = torch.ones(self.num_envs, device=self.device)
        return vel_reward * push.float() * contact_weight * self._jump_motion_mask().float()

    def _reward_jump_push_pitch(self):
        _, _, push, _, _, _, _, _ = self._jump_phase_masks()
        pitch_weight = getattr(self.cfg.rewards, "jump_push_pitch_weight", 2.0)
        pitch_ang_vel_weight = getattr(self.cfg.rewards, "jump_push_pitch_ang_vel_weight", 0.5)
        pitch_error = (
            pitch_weight * torch.square(self.projected_gravity[:, 0])
            + pitch_ang_vel_weight * torch.square(self.base_ang_vel[:, 1])
        )
        return pitch_error * push.float() * self._jump_motion_mask().float()

    def _reward_jump_push_force_balance(self):
        _, _, push, _, _, _, _, _ = self._jump_phase_masks()
        foot_forces = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)
        mean_force = torch.mean(foot_forces, dim=1, keepdim=True)
        max_force = max(getattr(self.cfg.rewards, "jump_push_balance_force_norm", 120.0), 1e-6)
        imbalance = torch.mean(torch.square((foot_forces - mean_force) / max_force), dim=1)
        return imbalance * push.float() * self._jump_motion_mask().float()

    def _reward_jump_push_front_hind_sync(self):
        _, compress, push, _, _, _, _, _ = self._jump_phase_masks()
        if self.default_dof_pos.shape[0] == self.num_envs:
            default_dof_pos = self.default_dof_pos
        else:
            default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        front_ids = getattr(self.cfg.rewards, "jump_front_hind_sync_front_ids", [1, 2, 4, 5])
        hind_ids = getattr(self.cfg.rewards, "jump_front_hind_sync_hind_ids", [7, 8, 10, 11])
        front_dev = self.dof_pos[:, front_ids] - default_dof_pos[:, front_ids]
        hind_dev = self.dof_pos[:, hind_ids] - default_dof_pos[:, hind_ids]
        sync_error = torch.mean(torch.square(front_dev - hind_dev), dim=1)
        active = compress | push
        return sync_error * active.float() * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_prepare_stance(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, device=self.device)

        prepare = (~self._single_jump_flag()) & (~self.has_jumped)
        if self.default_dof_pos.shape[0] == self.num_envs:
            default_dof_pos = self.default_dof_pos
        else:
            default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        dof_sigma = max(getattr(self.cfg.rewards, "jump_prepare_dof_sigma", 0.20), 1e-6)
        vel_sigma = max(getattr(self.cfg.rewards, "jump_prepare_vel_sigma", 0.08), 1e-6)
        dof_error = torch.mean(torch.square(self.dof_pos - default_dof_pos), dim=1)
        vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) + torch.square(self.base_ang_vel[:, 2])
        return torch.exp(-dof_error / dof_sigma - vel_error / vel_sigma) * prepare.float()

    def _reward_line_z(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, device=self.device)

        active = self._single_jump_flag() & (~self.has_jumped)
        start_vel = getattr(self.cfg.rewards, "line_z_min", 0.15)
        target_vel = getattr(self.cfg.rewards, "line_z_target", 1.45)
        vel_span = max(target_vel - start_vel, 1e-6)
        vel_reward = torch.clamp((self.base_lin_vel[:, 2] - start_vel) / vel_span, min=0.0, max=1.0)
        return vel_reward * active.float()

    def _reward_base_height_flight(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, device=self.device)

        active = self.was_in_flight & (~self.has_jumped)
        base_height = self._get_base_heights()
        start_height = getattr(self.cfg.rewards, "base_height_flight_min", getattr(self.cfg.rewards, "jump_height_min", self.cfg.rewards.base_height_target))
        target_height = getattr(self.cfg.rewards, "base_height_flight_target", getattr(self.cfg.rewards, "jump_height_target", start_height + 0.1))
        height_span = max(target_height - start_height, 1e-6)
        height_reward = torch.clamp((base_height - start_height) / height_span, min=0.0, max=1.0)
        return height_reward * active.float()

    def _reward_land_pos(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, device=self.device)

        target_x = self.commands[:, 3]
        target_y = self.commands[:, 4]
        x_error = torch.square(self.landing_forward_distance - target_x)
        y_weight = getattr(self.cfg.rewards, "land_pos_lateral_weight", 2.0)
        y_error = y_weight * torch.square(self.landing_lateral_distance - target_y)
        sigma = max(getattr(self.cfg.rewards, "land_pos_sigma", 0.06), 1e-6)
        return torch.exp(-(x_error + y_error) / sigma) * self.has_jumped.float()

    def _reward_jump_landing_stable(self):
        lin_weight = getattr(self.cfg.rewards, "jump_landing_stable_lin_weight", 1.0)
        ang_weight = getattr(self.cfg.rewards, "jump_landing_stable_ang_weight", 0.5)
        tilt_weight = getattr(self.cfg.rewards, "jump_landing_stable_tilt_weight", 2.0)
        yaw_weight = getattr(self.cfg.rewards, "jump_landing_stable_yaw_weight", 0.5)

        if self._single_jump_flag_enabled():
            active = self.has_jumped
            lin_error = lin_weight * torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
            yaw_error = yaw_weight * torch.square(self.base_ang_vel[:, 2])
        else:
            _, _, _, _, _, landing, recovery, _ = self._jump_phase_masks()
            active = (landing | recovery) & self._jump_motion_mask()
            lin_error = lin_weight * torch.sum(
                torch.square(self.base_lin_vel[:, :2] - self.commands[:, :2]),
                dim=1,
            )
            yaw_error = yaw_weight * torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])

        ang_error = ang_weight * torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        tilt_error = tilt_weight * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        sigma = max(getattr(self.cfg.rewards, "jump_landing_stable_sigma", 0.25), 1e-6)
        contact_fraction = torch.mean(self._jump_contact_mask().float(), dim=1)
        return (
            torch.exp(-(lin_error + ang_error + yaw_error + tilt_error) / sigma)
            * active.float()
            * contact_fraction
        )

    def _reward_jump_second_takeoff(self):
        if not self._single_jump_flag_enabled():
            return torch.zeros(self.num_envs, device=self.device)

        return (self.has_jumped & self._jump_all_air_mask()).float()

    def _jump_forward_distance_active_mask(self):
        if self._single_jump_flag_enabled():
            return self.has_jumped

        if getattr(self.cfg.commands, "single_jump_mode", False):
            start_time_s = getattr(
                self.cfg.rewards,
                "jump_forward_distance_start_s",
                getattr(self.cfg.commands, "single_jump_command_time_s", 1.05),
            )
            start_steps = max(int(start_time_s / self.dt), 1)
            return self.episode_length_buf >= start_steps

        _, _, _, _, _, landing, recovery, _ = self._jump_phase_masks()
        return (landing | recovery) & self._jump_motion_mask()

    def _reward_jump_forward_distance(self):
        if self._single_jump_flag_enabled():
            forward_distance = self.landing_forward_distance
            min_dist = self.commands[:, 3] - getattr(self.cfg.rewards, "jump_forward_distance_tolerance", 0.2)
            target_dist = self.commands[:, 3]
            max_dist = self.commands[:, 3] + getattr(self.cfg.rewards, "jump_forward_distance_tolerance", 0.2)
        else:
            displacement = self.root_states[:, :3] - self.episode_start_pos
            forward_distance = torch.sum(displacement * self.episode_start_forward, dim=1)
            min_dist = getattr(self.cfg.rewards, "jump_forward_distance_min", 0.8)
            target_dist = getattr(self.cfg.rewards, "jump_forward_distance_target", 1.0)
            max_dist = getattr(self.cfg.rewards, "jump_forward_distance_max", 1.2)

        sigma = max(getattr(self.cfg.rewards, "jump_forward_distance_sigma", 0.08), 1e-6)
        if not torch.is_tensor(min_dist):
            min_dist = torch.ones_like(forward_distance) * min_dist
        if not torch.is_tensor(target_dist):
            target_dist = torch.ones_like(forward_distance) * target_dist
        if not torch.is_tensor(max_dist):
            max_dist = torch.ones_like(forward_distance) * max_dist

        clamped_distance = torch.minimum(torch.maximum(forward_distance, min_dist), max_dist)
        band_reward = torch.exp(-torch.square(clamped_distance - target_dist) / sigma)
        progress_den = torch.clamp(min_dist, min=1e-6)
        overshoot_den = torch.clamp(max_dist - target_dist, min=1e-6)
        progress_scale = torch.clamp(forward_distance / progress_den, min=0.0, max=1.0)
        overshoot_penalty = torch.clamp(
            (forward_distance - max_dist) / overshoot_den,
            min=0.0,
            max=2.0,
        )

        reward = band_reward * progress_scale - overshoot_penalty
        return reward * self._jump_forward_distance_active_mask().float()

    def _reward_jump_ref_forward_vel(self):
        """Reward forward speed during push/flight without penalizing useful overshoot."""
        _, _, push, flight_up, flight_down, _, _, _ = self._jump_phase_masks()
        active = push | flight_up | flight_down

        start_vel = getattr(self.cfg.rewards, "jump_ref_forward_vel_min", 0.4)
        target_cap = getattr(self.cfg.rewards, "jump_ref_forward_vel_target", 1.8)
        target_vel = torch.clamp(self.commands[:, 0], min=start_vel, max=target_cap)
        vel_span = torch.clamp(target_vel - start_vel, min=1e-6)
        vel_reward = torch.clamp((self.base_lin_vel[:, 0] - start_vel) / vel_span, min=0.0, max=1.0)
        return vel_reward * active.float() * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _jump_phase_lerp(self, phase, start, end, start_value, end_value):
        ratio = torch.clamp((phase - start) / max(end - start, 1e-6), min=0.0, max=1.0)
        return start_value + ratio * (end_value - start_value)

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

    def _reward_jump_ref_base_height(self):
        ref_height, _, _, _ = self._get_jump_reference()
        base_height = self._get_base_heights()
        err = torch.square(base_height - ref_height)
        sigma = max(getattr(self.cfg.rewards, "jump_ref_sigma_height", 0.04), 1e-6)
        return torch.exp(-err / sigma) * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_ref_z_vel(self):
        _, ref_z_vel, _, _ = self._get_jump_reference()
        err = torch.square(self.base_lin_vel[:, 2] - ref_z_vel)
        sigma = max(getattr(self.cfg.rewards, "jump_ref_sigma_z_vel", 0.4), 1e-6)
        return torch.exp(-err / sigma) * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_ref_foot_height(self):
        _, _, ref_foot_height, ref_contact = self._get_jump_reference()
        feet_height = self._get_feet_heights()
        flight_weight = 1.0 - ref_contact
        weight_sum = torch.clamp(torch.sum(flight_weight, dim=1), min=1.0)
        err = torch.sum(torch.square(feet_height - ref_foot_height.unsqueeze(1)) * flight_weight, dim=1) / weight_sum
        sigma = max(getattr(self.cfg.rewards, "jump_ref_sigma_foot", 0.04), 1e-6)
        flight_active = torch.clamp(torch.mean(flight_weight, dim=1), min=0.0, max=1.0)
        return torch.exp(-err / sigma) * flight_active * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_ref_contact(self):
        _, _, _, ref_contact = self._get_jump_reference()
        contact = self._jump_contact_mask().float()
        return (1.0 - torch.mean(torch.abs(contact - ref_contact), dim=1)) * self._jump_motion_mask().float() * self._tracking_relief_scale()

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

    def _reward_jump_landing_knee_bend(self):
        _, _, _, _, _, landing, recovery, _ = self._jump_phase_masks()
        knee_ids = [2, 5, 8, 11]
        knee_max = getattr(self.cfg.rewards, "jump_landing_knee_max", 1.55)
        knee_excess = torch.clamp(self.dof_pos[:, knee_ids] - knee_max, min=0.0)
        if self._single_jump_flag_enabled():
            return torch.mean(torch.square(knee_excess), dim=1) * self.has_jumped.float() * self._tracking_relief_scale()

        active = landing | recovery
        return torch.mean(torch.square(knee_excess), dim=1) * active.float() * self._jump_motion_mask().float() * self._tracking_relief_scale()

    def _reward_jump_hind_knee_bend(self):
        """Keep the hind shanks off the ground without constraining the front legs."""
        hind_knee_ids = [8, 11]
        knee_max = getattr(self.cfg.rewards, "jump_hind_knee_max", 1.75)
        knee_excess = torch.clamp(
            self.dof_pos[:, hind_knee_ids] - knee_max,
            min=0.0,
        )
        return (
            torch.mean(torch.square(knee_excess), dim=1)
            * self._jump_motion_mask().float()
            * self._tracking_relief_scale()
        )

    def _reward_jump_knee_bend(self):
        """Prevent all four knees from folding into a kneeling support pose."""
        knee_ids = [2, 5, 8, 11]
        _, compress, push, _, _, _, _, _ = self._jump_phase_masks()
        moving = self._jump_motion_mask()

        normal_max = getattr(self.cfg.rewards, "jump_knee_max", 1.85)
        push_max = getattr(self.cfg.rewards, "jump_knee_push_max", 2.20)
        stand_max = getattr(self.cfg.rewards, "jump_knee_stand_max", 1.75)
        knee_max = torch.full(
            (self.num_envs,), normal_max, device=self.device
        )
        knee_max = torch.where(
            moving & (compress | push),
            torch.full_like(knee_max, push_max),
            knee_max,
        )
        knee_max = torch.where(
            ~moving,
            torch.full_like(knee_max, stand_max),
            knee_max,
        )

        knee_excess = torch.clamp(
            self.dof_pos[:, knee_ids] - knee_max.unsqueeze(1), min=0.0
        )
        return torch.mean(torch.square(knee_excess), dim=1)

    def _reward_shank_clearance(self):
        """Penalize low knee/shank origins before the shank touches terrain."""
        if len(self.shank_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        shank_z = self.rigid_body_states.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.shank_indices, 2]
        if self.cfg.terrain.measure_heights:
            ground_height = torch.mean(self.measured_heights, dim=1)
        else:
            ground_height = torch.zeros(
                self.num_envs, device=self.device
            )
        clearance = shank_z - ground_height.unsqueeze(1)
        target = getattr(self.cfg.rewards, "shank_clearance_target", 0.18)
        low_clearance = torch.clamp(target - clearance, min=0.0)
        return torch.mean(torch.square(low_clearance), dim=1)

    def _reward_jump_leg_symmetry(self):
        """Keep the four legs moving as one symmetric pronk group without forcing a specific tucked pose."""
        if getattr(self.cfg.rewards, "jump_leg_symmetry_hind_only", False):
            hind_left_ids = [7, 8]
            hind_right_ids = [10, 11]
            pos_deadband = getattr(self.cfg.rewards, "jump_leg_symmetry_pos_deadband", 0.05)
            vel_deadband = getattr(self.cfg.rewards, "jump_leg_symmetry_vel_deadband", 0.5)
            vel_weight = getattr(self.cfg.rewards, "jump_leg_symmetry_vel_weight", 0.05)

            pos_difference = torch.abs(
                self.dof_pos[:, hind_left_ids] - self.dof_pos[:, hind_right_ids]
            )
            vel_difference = torch.abs(
                self.dof_vel[:, hind_left_ids] - self.dof_vel[:, hind_right_ids]
            )
            pos_error = torch.mean(
                torch.square(torch.clamp(pos_difference - pos_deadband, min=0.0)),
                dim=1,
            )
            vel_error = torch.mean(
                torch.square(torch.clamp(vel_difference - vel_deadband, min=0.0)),
                dim=1,
            )

            if getattr(self.cfg.rewards, "jump_leg_symmetry_flight_only", False):
                active = self._jump_all_air_mask().float()
            else:
                active = torch.ones(self.num_envs, device=self.device)

            return (
                (pos_error + vel_weight * vel_error)
                * active
                * self._jump_motion_mask().float()
                * self._tracking_relief_scale()
            )

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
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / 4 * self._tracking_relief_scale()
    
    def _reward_tracking_contacts_shaped_force_exp(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        # for i in range(4):
        #     reward += - (1 - desired_contact[:, i]) * (
        #                 1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        for i in range(4):
            reward += (1 - desired_contact[:, i]) * torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        return reward / 4 * self._tracking_relief_scale()

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.feet_vel, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        return reward / 4 * self._tracking_relief_scale()
    
    def _reward_tracking_contacts_shaped_vel_exp(self):
        foot_velocities = torch.norm(self.feet_vel, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        # for i in range(4):
        #     reward += - (desired_contact[:, i] * (
        #                 1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        for i in range(4):
            reward += desired_contact[:, i] * torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)
        return reward / 4 * self._tracking_relief_scale()

    def _reward_anti_trot_diagonal_swing(self):
        contact_threshold = getattr(self.cfg.rewards, "anti_trot_contact_threshold", 1.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > contact_threshold
        swing = (~contact).float()
        diagonal_swing = swing[:, 0] * swing[:, 3] + swing[:, 1] * swing[:, 2]
        moving = ((torch.norm(self.commands[:, :2], dim=1) > 0.2) | (torch.abs(self.commands[:, 2]) > 0.1)).float()
        return diagonal_swing * moving * self._tracking_relief_scale()

    def _lateral_gait_motion_mask(self):
        return ((torch.norm(self.commands[:, :2], dim=1) > 0.2) | (torch.abs(self.commands[:, 2]) > 0.1)).float()

    def _lateral_pair_contact_mask(self):
        contact_threshold = getattr(self.cfg.rewards, "lateral_pair_contact_threshold", 1.0)
        return self.contact_forces[:, self.feet_indices, 2] > contact_threshold

    def _lateral_pair_contact_events(self):
        contact = self._lateral_pair_contact_mask()
        prev_contact = self.lateral_pair_prev_contacts if hasattr(self, "lateral_pair_prev_contacts") else self.prev_contacts
        first_contact = contact & ~prev_contact
        takeoff = (~contact) & prev_contact
        return contact, first_contact, takeoff

    def _lateral_pair_phase_windows(self):
        pair_phases = torch.stack((self.foot_indices[:, 0], self.foot_indices[:, 1]), dim=1)
        touchdown_window = getattr(self.cfg.rewards, "lateral_pair_touchdown_phase_window", 0.10)
        takeoff_window = getattr(self.cfg.rewards, "lateral_pair_takeoff_phase_window", 0.08)
        duration = getattr(self.cfg.commands, "durations", 0.5)

        touchdown_window_mask = pair_phases <= touchdown_window
        takeoff_window_mask = torch.abs(pair_phases - duration) <= takeoff_window
        return touchdown_window_mask, takeoff_window_mask

    def _reward_lateral_pair_contact_sync(self):
        contact, _, _ = self._lateral_pair_contact_events()
        desired_contact = self.desired_contact_states > 0.5

        left_sync = (
            (contact[:, 0] == contact[:, 2])
            & (contact[:, 0] == desired_contact[:, 0])
            & (contact[:, 2] == desired_contact[:, 2])
        )
        right_sync = (
            (contact[:, 1] == contact[:, 3])
            & (contact[:, 1] == desired_contact[:, 1])
            & (contact[:, 3] == desired_contact[:, 3])
        )
        pair_sync = 0.5 * (left_sync.float() + right_sync.float())
        return pair_sync * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_lateral_pair_touchdown_sync(self):
        _, first_contact, _ = self._lateral_pair_contact_events()
        desired_contact = self.desired_contact_states > 0.5
        touchdown_window, _ = self._lateral_pair_phase_windows()

        left_touchdown = first_contact[:, 0] & first_contact[:, 2] & desired_contact[:, 0] & desired_contact[:, 2]
        right_touchdown = first_contact[:, 1] & first_contact[:, 3] & desired_contact[:, 1] & desired_contact[:, 3]
        pair_touchdown = torch.stack((left_touchdown, right_touchdown), dim=1)

        # Only one touchdown reward is allowed per same-side pair in each touchdown phase window.
        self.lateral_pair_touchdown_armed = torch.where(
            touchdown_window,
            self.lateral_pair_touchdown_armed,
            torch.ones_like(self.lateral_pair_touchdown_armed),
        )
        valid_touchdown = pair_touchdown & touchdown_window & self.lateral_pair_touchdown_armed
        self.lateral_pair_touchdown_armed = torch.where(
            valid_touchdown,
            torch.zeros_like(self.lateral_pair_touchdown_armed),
            self.lateral_pair_touchdown_armed,
        )

        touchdown_sync = torch.clamp(torch.sum(valid_touchdown.float(), dim=1), max=1.0)
        return touchdown_sync * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_lateral_pair_takeoff_sync(self):
        _, _, takeoff = self._lateral_pair_contact_events()
        desired_contact = self.desired_contact_states > 0.5
        _, takeoff_window = self._lateral_pair_phase_windows()

        left_takeoff = takeoff[:, 0] & takeoff[:, 2] & (~desired_contact[:, 0]) & (~desired_contact[:, 2])
        right_takeoff = takeoff[:, 1] & takeoff[:, 3] & (~desired_contact[:, 1]) & (~desired_contact[:, 3])
        pair_takeoff = torch.stack((left_takeoff, right_takeoff), dim=1)

        # Only one takeoff reward is allowed per same-side pair in each takeoff phase window.
        self.lateral_pair_takeoff_armed = torch.where(
            takeoff_window,
            self.lateral_pair_takeoff_armed,
            torch.ones_like(self.lateral_pair_takeoff_armed),
        )
        valid_takeoff = pair_takeoff & takeoff_window & self.lateral_pair_takeoff_armed
        self.lateral_pair_takeoff_armed = torch.where(
            valid_takeoff,
            torch.zeros_like(self.lateral_pair_takeoff_armed),
            self.lateral_pair_takeoff_armed,
        )

        takeoff_sync = torch.clamp(torch.sum(valid_takeoff.float(), dim=1), max=1.0)
        return takeoff_sync * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_lateral_pair_extra_touchdown(self):
        contact, first_contact, _ = self._lateral_pair_contact_events()
        pair_phases = torch.stack((self.foot_indices[:, 0], self.foot_indices[:, 1]), dim=1)
        cycle_reset = pair_phases < self.lateral_pair_last_phase

        left_prev_contact = self.lateral_pair_prev_contacts[:, 0] | self.lateral_pair_prev_contacts[:, 2]
        right_prev_contact = self.lateral_pair_prev_contacts[:, 1] | self.lateral_pair_prev_contacts[:, 3]
        pair_prev_contact = torch.stack((left_prev_contact, right_prev_contact), dim=1)
        self.lateral_pair_touchdown_seen_in_stance = torch.where(
            cycle_reset,
            pair_prev_contact,
            self.lateral_pair_touchdown_seen_in_stance,
        )

        left_touchdown = first_contact[:, 0] | first_contact[:, 2]
        right_touchdown = first_contact[:, 1] | first_contact[:, 3]
        pair_touchdown = torch.stack((left_touchdown, right_touchdown), dim=1)
        self.lateral_pair_touchdown_cooldown = torch.clamp(self.lateral_pair_touchdown_cooldown - self.dt, min=0.0)
        cooldown_touchdown = self.lateral_pair_touchdown_cooldown > 0.0

        # A same-side pair should form one clean touchdown block in each gait cycle.
        # The cooldown also catches toe taps just before the cycle boundary.
        extra_touchdown = pair_touchdown & (self.lateral_pair_touchdown_seen_in_stance | cooldown_touchdown)
        self.lateral_pair_touchdown_seen_in_stance |= pair_touchdown
        cooldown_s = getattr(self.cfg.rewards, "lateral_pair_touchdown_cooldown_s", 0.18)
        self.lateral_pair_touchdown_cooldown = torch.where(
            pair_touchdown,
            torch.ones_like(self.lateral_pair_touchdown_cooldown) * cooldown_s,
            self.lateral_pair_touchdown_cooldown,
        )
        self.lateral_pair_last_phase[:] = pair_phases

        extra = torch.clamp(torch.sum(extra_touchdown.float(), dim=1), max=1.0)
        return extra * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_short_contact(self):
        _, _, takeoff = self._lateral_pair_contact_events()
        contact_time = self.lateral_pair_prev_contact_duration

        default_min_time = getattr(self.cfg.rewards, "short_contact_min_time_s", 0.10)
        min_time = torch.ones_like(contact_time) * default_min_time
        weights = torch.ones_like(contact_time)

        front_ids = getattr(self.cfg.rewards, "front_foot_ids", [0, 1])
        hind_ids = getattr(self.cfg.rewards, "hind_foot_ids", [2, 3])
        if len(front_ids) > 0:
            min_time[:, front_ids] = getattr(self.cfg.rewards, "front_contact_min_time_s", default_min_time)
            weights[:, front_ids] = getattr(self.cfg.rewards, "front_short_contact_weight", 1.0)
        if len(hind_ids) > 0:
            min_time[:, hind_ids] = getattr(self.cfg.rewards, "hind_contact_min_time_s", default_min_time)
            weights[:, hind_ids] = getattr(self.cfg.rewards, "hind_short_contact_weight", 1.0)

        short_takeoff = takeoff & (contact_time > 0.0) & (contact_time < min_time)
        severity = torch.clamp((min_time - contact_time) / torch.clamp(min_time, min=1e-6), min=0.0, max=1.0)
        return torch.sum(short_takeoff.float() * severity * weights, dim=1) * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_swing_contact(self):
        contact = self._lateral_pair_contact_mask()
        swing_threshold = getattr(self.cfg.rewards, "swing_contact_threshold", 0.3)
        swing = self.desired_contact_states < swing_threshold
        swing_contact = contact & swing
        return torch.sum(swing_contact.float(), dim=1) * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_lateral_pair_contact_hold(self):
        contact, _, _ = self._lateral_pair_contact_events()
        desired_contact = self.desired_contact_states > 0.5
        touchdown_window, takeoff_window = self._lateral_pair_phase_windows()
        mid_stance = desired_contact[:, [0, 1]] & desired_contact[:, [2, 3]] & (~touchdown_window) & (~takeoff_window)

        left_lost_contact = ((~contact[:, 0]) | (~contact[:, 2])) & mid_stance[:, 0]
        right_lost_contact = ((~contact[:, 1]) | (~contact[:, 3])) & mid_stance[:, 1]
        lost_contact = left_lost_contact.float() + right_lost_contact.float()
        return lost_contact * self._lateral_gait_motion_mask() * self._tracking_relief_scale()

    def _reward_feet_inward(self):
        """Penalize stance feet moving toward or across the body centerline."""
        foot_y = self._feet_pos_body_frame()[:, :, 1]
        # Foot order is FL, FR, HL, HR; positive signed_y points away from the body.
        signed_y = torch.stack((foot_y[:, 0], -foot_y[:, 1], foot_y[:, 2], -foot_y[:, 3]), dim=1)

        min_lateral_distance = getattr(self.cfg.rewards, "feet_inward_min_lateral_distance", 0.15)
        inward_error = torch.clamp(
            (min_lateral_distance - signed_y) / max(min_lateral_distance, 1e-6),
            min=0.0,
            max=2.0,
        )

        contact_threshold = getattr(self.cfg.rewards, "feet_inward_contact_threshold", 2.0)
        desired_threshold = getattr(self.cfg.rewards, "feet_inward_desired_contact_threshold", 0.35)
        actual_contact = self.contact_forces[:, self.feet_indices, 2] > contact_threshold
        expected_stance = self.desired_contact_states > desired_threshold
        stance_or_landing = actual_contact | expected_stance

        return (
            torch.sum(torch.square(inward_error) * stance_or_landing.float(), dim=1)
            * self._lateral_gait_motion_mask()
            * self._tracking_relief_scale()
        )

    def _reward_front_swing_clearance(self):
        """Keep front feet off the ground while they are still expected to swing."""
        front_ids = getattr(self.cfg.rewards, "front_swing_clearance_foot_ids", [0, 1])
        if len(front_ids) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        foot_height = self._get_feet_heights()[:, front_ids]
        desired_contact = self.desired_contact_states[:, front_ids]
        swing_threshold = getattr(self.cfg.rewards, "front_swing_clearance_contact_threshold", 0.35)
        target_height = getattr(self.cfg.rewards, "front_swing_clearance_height", 0.065)

        swing_weight = torch.clamp(
            (swing_threshold - desired_contact) / max(swing_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        clearance_error = torch.square(torch.clamp(target_height - foot_height, min=0.0))
        return (
            torch.sum(clearance_error * swing_weight, dim=1)
            * self._lateral_gait_motion_mask()
            * self._tracking_relief_scale()
        )

    def _reward_hind_swing_height_limit(self):
        """Penalize excessive hind foot height during swing to avoid high lift and hard touchdown."""
        hind_ids = getattr(self.cfg.rewards, "hind_swing_height_limit_foot_ids", [2, 3])
        if len(hind_ids) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        foot_height = self._get_feet_heights()[:, hind_ids]
        desired_contact = self.desired_contact_states[:, hind_ids]
        swing_threshold = getattr(self.cfg.rewards, "hind_swing_height_limit_contact_threshold", 0.35)
        max_height = getattr(self.cfg.rewards, "hind_swing_height_limit_height", 0.07)

        swing_weight = torch.clamp(
            (swing_threshold - desired_contact) / max(swing_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        height_excess = torch.clamp((foot_height - max_height) / max(max_height, 1e-6), min=0.0)
        return (
            torch.sum(torch.square(height_excess) * swing_weight, dim=1)
            * self._lateral_gait_motion_mask()
            * self._tracking_relief_scale()
        )

    def _reward_hind_swing_z_vel(self):
        """Softly limit hind-foot vertical speed during swing for smoother visual motion."""
        hind_ids = getattr(self.cfg.rewards, "hind_swing_z_vel_foot_ids", [2, 3])
        if len(hind_ids) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        desired_contact = self.desired_contact_states[:, hind_ids]
        swing_threshold = getattr(self.cfg.rewards, "hind_swing_z_vel_contact_threshold", 0.35)
        max_z_vel = getattr(self.cfg.rewards, "hind_swing_z_vel_max", 0.35)
        swing_weight = torch.clamp(
            (swing_threshold - desired_contact) / max(swing_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )

        z_vel_excess = torch.clamp(torch.abs(self.feet_vel[:, hind_ids, 2]) - max_z_vel, min=0.0)
        return (
            torch.sum(torch.square(z_vel_excess) * swing_weight, dim=1)
            * self._lateral_gait_motion_mask()
            * self._tracking_relief_scale()
        )

    def _reward_feet_clearance_cmd_linear(self):
        # requires: foot phase (self.foot_indices), desired_contact_states, and swing height command at index 9
        # if self.commands.shape[1] <= 9:
        #     return torch.zeros(self.num_envs, device=self.device)
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        # foot_height = self.feet_pos[:, :, 2].view(self.num_envs, -1)
        foot_height = self._get_feet_heights()
        target_height = self.cfg.rewards.target_foot_height * phases + 0.02 # offset for foot radius 2cm
        height_error = torch.square(target_height - foot_height)

        # Slow gaits give the policy more time to hold a swing leg unnecessarily high.
        # Add an asymmetric extra cost only above the reference trajectory plus a margin.
        overheight_margin = getattr(self.cfg.rewards, "feet_clearance_overheight_margin", 0.0)
        overheight_weight = getattr(self.cfg.rewards, "feet_clearance_overheight_weight", 0.0)
        overheight_error = torch.square(
            torch.clamp(foot_height - target_height - overheight_margin, min=0.0)
        )
        rew_foot_clearance = (
            height_error + overheight_weight * overheight_error
        ) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1) * self._tracking_relief_scale()

    def _reward_raibert_heuristic(self):
        # if self.commands.shape[1] <= 4:
        #     return torch.zeros(self.num_envs, device=self.device)

        cur_footsteps_translated = self.feet_pos - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.cfg.commands.num_commands >= 13:
            desired_stance_width = self.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.32
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        if self.cfg.commands.num_commands >= 14:
            desired_stance_length = self.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.37
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.ones(self.num_envs, device=self.device) * self.cfg.commands.frequencies
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
