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
import math
from os import path as osp

from legged_gym.envs.cc1.cc1_wtw_config import Cc1RoughwtwCfg, Cc1RoughwtwCfgPPO


class Cc1StairswtwCfg( Cc1RoughwtwCfg ):

    class init_state( Cc1RoughwtwCfg.init_state ):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,
            'HL_HipX_joint': 0.0,
            'FR_HipX_joint': 0.0,
            'HR_HipX_joint': 0.0,

            'FL_HipY_joint': -0.8,
            'HL_HipY_joint': -0.8,
            'FR_HipY_joint': -0.8,
            'HR_HipY_joint': -0.8,

            'FL_Knee_joint': 1.6,
            'HL_Knee_joint': 1.6,
            'FR_Knee_joint': 1.6,
            'HR_Knee_joint': 1.6,
        }

    class terrain( Cc1RoughwtwCfg.terrain ):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 15  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 10.  # [m] length of generated terrain, default: 10.0
        terrain_width = 10.  # [m] width of generated terrain, default: 10.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        # terrain_proportions = [0.55, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        terrain_proportions = [0.0, 0.05, 0.05, 0.1, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands( Cc1RoughwtwCfg.commands ):
        curriculum = True
        max_forward_curriculum = 1.5  # x_vel 限制 [-1.0, 1.5]
        max_backward_curriculum = 1.5
        max_lat_curriculum = 1.0  # y_vel 限制 [-1.0, 1.0]
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        stand_still_command = False
        Rotate_command = False
        pacing_offset = False

        frequencies = 2.5
        phases = 0.5
        offsets = 0
        bounds = 0
        durations = 0.5

        class ranges( Cc1RoughwtwCfg.commands.ranges ):
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
            heading = [-math.pi, math.pi]

    class termination:
        base_vel_violate_commands = True

        out_of_border = True

        fall_down = True

    class domain_rand( Cc1RoughwtwCfg.domain_rand ):
        randomize_payload_mass = True
        payload_mass_range = [-1.0, 5.0]

        randomize_com_displacement = True
        com_displacement_range = [-0.07, 0.07]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]

        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_restitution = False
        restitution_range = [0., 1.0]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.8, 1.2]

        randomize_kd = True
        kd_range = [0.8, 1.2]

        base_init_pos_range = dict(
            x=[-1.0, 1.0],
            y=[-1.0, 1.0],
            z=[0.0, 0.05],
        )

        base_init_rot_range = dict(
            roll=[-0.2, 0.2],
            pitch=[-0.2, 0.2],
            yaw=[-0.0, 0.0],
        )

        base_init_vel_range = dict(
            x=[-0.5, 0.5],
            y=[-0.5, 0.5],
            z=[-0.5, 0.5],
            roll=[-0.5, 0.5],
            pitch=[-0.5, 0.5],
            yaw=[-0.5, 0.5],
        )

        dof_init_pos_ratio_range = [0.8, 1.2]

        randomize_dof_vel = False
        dof_init_vel_range = [-0.1, 0.1]

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.

        delay = True

    class rewards( Cc1RoughwtwCfg.rewards ):
        class scales:
            # general
            termination = -100.  # 终止惩罚：触发终止条件时给予大负奖励
            # velocity-tracking
            tracking_lin_vel = 2.0  # 线速度跟踪奖励（对齐命令 v_x,v_y）
            tracking_ang_vel = 2.0  # 角速度跟踪奖励（对齐命令 yaw 角速度/heading）
            # root
            lin_vel_z = -2.0  # 惩罚机身竖直速度（抑制跳动/抖动）
            ang_vel_xy = -0.05  # 惩罚机身 roll/pitch 角速度（抑制翻滚）
            orientation = -2.0  # 惩罚机身姿态偏离（保持躯干水平） -0.2
            base_height = -5.0  # 惩罚机身高度偏离目标（台阶任务更强约束） -5.0
            # joint
            torques = -0.0001  # 惩罚关节力矩（节能/平滑）
            # torque_limits = -0.0  # 关节力矩超限惩罚（未启用时为 0）
            # dof_vel = -0.0  # 关节速度惩罚（可抑制高速摆动；此处关闭）
            dof_acc = -2.5e-7  # 关节加速度惩罚（平滑动作/减小冲击）
            stand_still = -0.5  # 在“站立命令”时惩罚移动（防止抖动走动） -0.1
            # hip_pos = -0.12  # 髋关节偏离默认位姿惩罚（稳姿/防外八内八） -0.12
            # thigh_pose = -0.05  # 大腿关节姿态约束惩罚（稳定腿型） -0.05
            # calf_pose = -0.03  # 小腿关节姿态约束惩罚（稳定腿型） -0.03
            # dof_pos_limits = -0.0  # 关节位置接近硬限位惩罚（此处关闭） -0.0
            # dof_vel_limits = -0.0  # 关节速度接近限位惩罚（此处关闭） -0.0
            joint_power = -3e-5  # 关节功率惩罚（|tau * qd|，能耗相关） -3e-5
            power_distribution = -10e-6 # 鼓励关节功率分布均匀（能耗平衡） -10e-6
            # feet_mirror = -0.05  # 四足步态对称/镜像约束（减小左右不一致） -0.05
            # action
            action_rate = -0.02  # 动作变化率惩罚（抑制相邻时刻动作突变） -0.05
            smoothness = -0.01  # 动作平滑项（更强约束/去抖） -0.02
            # hip_action_magnitude = -0.0  # 髋关节动作幅度惩罚（此处关闭） -0.01
            # contact
            collision = -5.0  # 非期望部位碰撞惩罚（如躯干/大腿触地） -5.0
            # feet_contact_forces = -0.00015  # 足端接触力过大惩罚（抑制“砸地”） -0.00015
            # others
            # feet_air_time = 0.25  # 鼓励足端腾空时间（促进迈步/跨越） 0.25
            # feet_air_time_variance_velocity = -10.0
            has_contact = 2.0  # 鼓励保持接触（提升稳定性，避免全腾空(base 原地不动)） 2.0
            feet_stumble = -2.0  # 绊倒/磕碰惩罚（足端撞到台阶边缘等） -2.0
            feet_slide = -0.01  # 足端打滑惩罚（接触时水平滑移）-0.01
            # feet_clearance_base = -0.0  # 足端相对机身抬脚高度奖励（此处关闭）-0.0
            # feet_clearance_terrain = -0.0  # 足端相对地形抬脚高度奖励（此处关闭）-0.0
            # feet_yaw_clearance_terrain = 2.0  # base 原地旋转时鼓励抬脚，减少“拧着走/刮地” 1.0
            stuck = -1.  # 卡住/不动惩罚（速度过小或长期无进展） -1.0
            # upward = 0.0  # 向上运动奖励（此处关闭/预留） 0.0
            # feet
            raibert_heuristic = -10.0  # Raibert启发式奖励：根据当前base速度和步态周期计算理想的足部位置，奖励与理想位置的接近程度。鼓励足部在适当位置着地以稳定运动。
            # tracking_contacts_shaped_force_exp = 0.1
            # tracking_contacts_shaped_vel_exp = 0.1
            feet_clearance_cmd_linear = -30.0

        only_positive_rewards = False  # 是否将总奖励裁剪为非负（台阶任务通常保留负奖励更稳定）
        tracking_sigma = 0.20  # 跟踪奖励的高斯 sigma（reward = exp(-err^2/sigma)），越严格”
        soft_dof_pos_limit = 0.95  # 关节位置软限位（按 URDF 上下限比例）
        soft_dof_vel_limit = 0.95  # 关节速度软限位比例
        soft_torque_limit = 0.95  # 关节力矩软限位比例
        base_height_target = 0.37  # 机身目标高度 [m]
        feet_height_target_base = -0.32  # 足端相对 base 的目标高度 [m]
        feet_height_target_terrain = 0.15  # 足端离地目标高度 [m]
        max_contact_force = 100.  # 接触力惩罚阈值 [N]（超过开始惩罚/截断）
        target_foot_height = 0.15  # feet height
        target_foot_height_yaw = 0.1  # feet height
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        cycle_time = 0.5

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Cc1StairswtwCfgPPO( Cc1RoughwtwCfgPPO ):

    class runner( Cc1RoughwtwCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        # algorithm_class_name = 'HybridPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 200  # check for potential saves every this many iterations
        experiment_name = 'stairs_wtw_cc1'
        run_name = ''
        # load and resume
        resume = True
        load_run = osp.join(logs_root, 'flat_wtw_cc1', 'Apr18_19-15-43_')
        # resume = False
        # load_run = -1
        checkpoint = -1  # -1 = last saved model