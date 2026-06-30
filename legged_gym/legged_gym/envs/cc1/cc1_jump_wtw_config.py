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
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.cc1.cc1_config import Cc1RoughCfg

class Cc1JumpwtwCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096  # 并行仿真的环境数量（需根据GPU显存调整）
        num_one_step_observations = 61  # 单步 观测向量 维度（原始传感器数据）
        num_observations = num_one_step_observations * 6    # 总 观测向量 维度（含6步历史）
        num_one_step_privileged_obs = 61 + 3 + 3 + 187  # 单步 特权观测向量 维度，（+3维线速度 + 3维随机扰动力 + 地形扫描(187))
        num_privileged_obs = num_one_step_privileged_obs * 1    # 总 特权观测向量 维度，if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12  # 动作空间维度（12个关节）
        env_spacing = 3.  # 环境之间的间距（单位：米），not used with heightfields/trimeshes
        send_timeouts = True  # 是否发送超时信号给算法，send time out information to the algorithm
        episode_length_s = 20  # 单次训练Episode的时长（秒），episode length in seconds
        observe_gait_commands = True  # 是否观察gait commands，observe gait commands in the observations

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35]  # 0.36   # 初始位置（x,y,z）单位：米
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # action = 0.0，即零动作时的目标关节角度（站立姿态）
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

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 15 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        terrain_proportions = [1.0, 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'  # 控制类型（P=位置控制，T=力矩控制）
        stiffness = {'joint': 30.0} # 关节刚度（单位：N·m/rad）
        damping = {'joint': 1.0}    # 关节阻尼（单位：N·m·s/rad）
        action_scale = 0.25  # 动作缩放因子（目标角度 = 动作 * scale + 默认角度）
        decimation = 4      # 每个policy DT 包含的 sim DT 的个数
        hip_reduction = 1 # 髋关节扭矩缩放因子（用于平衡前后腿负载）

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_forward_curriculum = 2.2  # 速度课程上限保留；本任务实际采样范围由 ranges.lin_vel_x 控制
        max_backward_curriculum = 1.5
        max_lat_curriculum = 1.0  # y_vel 限制 [-1.0, 1.0]
        max_yaw_curriculum = 1.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        stand_still_command = True  # 保留少量零速样本，让不给指令/停下时能稳定站住
        Rotate_command = True  # 先只训练前向连续跳，不混入原地旋转/侧向动作
        pacing_offset = False

        # 随机发出急停请求，但等待机器人完成当前一跳并四脚落地后才清零命令。
        sudden_stop_command = True
        sudden_stop_phase_aware = True
        sudden_stop_restore_command = True
        sudden_stop_interval_s = 3.0
        sudden_stop_env_ratio_range = [0.10, 0.15]
        sudden_stop_duration_s = [0.8, 1.5]
        sudden_stop_min_speed = 0.20
        sudden_stop_min_yaw_speed = 0.20
        sudden_stop_min_episode_time_s = 2.0

        # 连续四脚跳，周期0.5-0.6s
        frequencies = 1.8
        phases = 0
        offsets = 0
        bounds = 0
        durations = 0.5

        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-math.pi, math.pi]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CC1_0626/urdf/CC1_0626.urdf'
        name = "Cc1"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "TORSO", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        privileged_contacts_on = ["TORSO", "THIGH", "SHANK"]
        self_collisions = 1  # 1：禁用自身各部分之间的碰撞检测（提升性能）；0：启用
        flip_visual_attachments = True  # 翻转视觉模型坐标系（Y-up转Z-up），许多 .obj meshes 必须从 y-up 转到 z-up

        disable_gravity = False
        collapse_fixed_joints = False  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False  # 翻转可视化附件

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class termination:
        base_vel_violate_commands = False  # 是否在终止条件中考虑 当地形等级>3时，base速度 与 命令速度差异过大(超过2m/s)（摔倒恢复训练关闭）

        out_of_border = True  # 是否在终止条件中考虑 走出边界外

        fall_down = False  # 是否在终止条件中考虑 跌落(base的z方向线速度 < -5)

    class domain_rand:
        # startup
        randomize_payload_mass = True  # 是否随机改变 base的质量（默认质量 ±）
        payload_mass_range = [-1.5, 4.0]

        randomize_com_displacement = True  # 是否随机改变 base 的质心偏移（xyz）
        com_displacement_range = dict(
            x=[-0.07, 0.07],
            y=[-0.07, 0.07],
            z=[-0.07, 0.07],
        )

        randomize_link_mass = False  # 是否随机更改env各刚体部位（除了base）的质量（默认质量 *）
        link_mass_range = [0.9, 1.1]

        # startup and reset
        randomize_friction = True  # 是否随机化env各刚体部位的 摩擦系数
        friction_range = [0.2, 1.25]

        randomize_restitution = False  # 是否随机化env各刚体部位的 弹性系数
        restitution_range = [0., 1.0]

        # reset
        randomize_motor_strength = True  # 是否随机化env的电机强度（输出的actions *）
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True  # 是否 随机改变PD控制器的p增益（stiffness）
        kp_range = [0.9, 1.1]

        randomize_kd = True  # 是否 随机改变PD控制器的D增益（damping）
        kd_range = [0.9, 1.1]

        # 重置时随机改变base的 位置（初始位置 +），默认x,y方向为 [-1, 1]，z方向为 0，若更改则为下面的
        base_init_pos_range = dict(
            x=[-1.0, 1.0],
            y=[-1.0, 1.0],
            z=[0.0, 0.05],
        )
        # 重置时随机设置base的 方向（摔倒恢复模式都设为 [-3.14, 3.14]）
        base_init_rot_range = dict(
            roll=[-0.2, 0.2],
            pitch=[-0.2, 0.2],
            yaw=[-0.0, 0.0],
        )
        # 重置时随机设置base的 线速度、角速度，默认x,y,x,rool,pitch,roll方向为 [-0.5, 0.5]，若更改则为下面的
        base_init_vel_range = dict(
            x=[-0.5, 0.5],
            y=[-0.5, 0.5],
            z=[-0.5, 0.5],
            roll=[-0.5, 0.5],
            pitch=[-0.5, 0.5],
            yaw=[-0.5, 0.5],
        )

        dof_init_pos_ratio_range = [0.8, 1.2]  # 重置时随机改变 关节初始位置（初始关节位置 *），默认为 [0.5, 1.5]

        randomize_dof_vel = False  # 重置时设置 关节初始速度
        dof_init_vel_range = [-0.1, 0.1]  # 默认为 0.0

        # interval
        disturbance = True  # 是否给base施加一个随机扰动力（xyz方向）
        disturbance_range = [-30.0, 30.0]  # N
        disturbance_interval = 8

        push_robots = True  # 是否给base在水平方向施加一个线速度
        push_interval_s = 16  # step间隔 [s]
        max_push_vel_xy = 1.  # 施加的最大线速度 [1m/s]

        delay = True  # actions是否随机延迟一个 policy_dt

        recover_mode = False  # 是否开启摔倒恢复模式

    class rewards( LeggedRobotCfg.rewards ):
        class scales( ):
            # general
            # termination = -5.0  # THIGH/SHANK/TORSO 触地终止时给惩罚，避免跪地/摔倒局部最优

            # velocity-tracking
            tracking_lin_vel = 3.0  # 先降低前向速度压力，避免后腿单独猛蹬导致前栽
            tracking_ang_vel = 3.0  # yaw 速度跟踪；当前 yaw 命令为 0，因此主要帮助抑制起跳/落地扭身
            straight_lin_vel_y = -1.0  # 纯前进命令时惩罚侧向漂移，减少连续跳逐步走偏
            straight_yaw_vel = -1.0  # 纯前进命令时惩罚自发偏航

            # root
            lin_vel_z = -0.1  # -0.2  # z 速度惩罚刻意很弱，避免把向上起跳速度压没
            ang_vel_xy = -0.05  # -0.05  # 全周期轻度抑制 roll/pitch 角速度，主要稳定落地和起跳过渡
            orientation = -5.0  # -2.0  # 惩罚 base 姿态偏斜，优先把空中和落地姿态压稳
            stop_orientation = -3.0  # 急停零速保持阶段压住身体点头和侧倾
            stop_ang_vel_xy = -0.8  # 急停零速保持阶段抑制 roll/pitch 角速度
            base_height = -20.0  # -20.0  # 防止低趴/跪地局部最优；真正跳高仍由 jump_height / jump_z_vel / jump_takeoff_z_vel 驱动
            # base_height_vel = -4.0  # base 目标高度速度 惩罚

            # joint
            torques = -0.0001  # 关节扭矩过大 惩罚
            # torque_limits = -0.0  # 关节扭矩接近极限 惩罚
            tn_curve = -2.0  # TN 曲线约束：惩罚高速大扭矩组合，保护跳跃起跳/落地时电机不过载
            # dof_vel = -0.0  # 关节速度过大 惩罚
            dof_acc = -2.5e-7  # 关节加速度 惩罚（若步态抖动，可增大惩罚）
            stand_still = -0.5  # 零速时把关节拉回默认站姿，避免停下后继续蹦/乱摆腿
            hip_pos = -0.4  # hip关节位置与默认位置的 偏差 惩罚，(原地不动 或 原地旋转) 时惩罚系数为 5.0，其他为 1.0
            # thigh_pose = -0.1
            # calf_pose = -0.1
            # dof_pos_limits = -0.0  # 关节位置接近极限 惩罚
            # dof_vel_limits = -0.0  # 关节速度接近极限 惩罚
            joint_power = -2e-5  # 关节高功率 惩罚：降低能耗（需平衡运动效率，过高惩罚会导致动作迟缓）
            power_distribution = -10e-6 # 鼓励关节功率分布均匀（能耗平衡） -10e-6
            # feet_mirror = -0.05  # 斜对称腿的关节位置偏差 惩罚

            # action
            action_rate = -0.02  # action 一阶变化惩罚，减少电机指令突变
            smoothness = -0.01  # action 二阶平滑惩罚，抑制抖腿；过大可能让起跳不够干脆
            # hip_action_magnitude = -0.00  # action 中的 髋关节hip（0,3,6,9）动作幅度 惩罚（防止 > 1.0）

            # contact
            # collision = -2.0  # -1.0 # THIGH/SHANK/TORSO 触地强惩罚，防止膝盖/小腿当支撑点跪地
            # feet_contact_forces = -0.00015  # 四足的接触力 > 100N 惩罚

            # others
            # feet_air_time = 0.0  # 四足的空中时间接近0.5s 奖励 (原地不动时除外)
            feet_air_time_variance_velocity = -10.0  # 抑制四脚腾空时间差异，辅助四脚同步跳
            has_contact = 2.0  # 零速时奖励四脚都触地，帮助不给指令时稳定站住
            # feet_stumble = -0.0  # 四足接触到垂直表面 惩罚
            feet_slide = -0.05  # 触地脚滑动惩罚，帮助落地不打滑
            # feet_clearance_base = -2.0  # 大速度下 四足距base目标距离 惩罚
            # feet_clearance_terrain = -0.0  # 大速度下 四足离地目标高度 惩罚
            # feet_yaw_clearance_terrain = 5.0  # (base原地旋转) 时 脚抬起
            # stuck = -0.00  # base 卡住 惩罚
            # upward = 0.0  # 重力投影向下 奖励（恢复训练时开启）

            # feet
            raibert_heuristic = -10.0  # Raibert启发式奖励：根据当前base速度和步态周期计算理想的足部位置，奖励与理想位置的接近程度。鼓励足部在适当位置着地以稳定运动。
            # tracking_contacts_shaped_force = 2.0
            # tracking_contacts_shaped_vel = 2.0
            tracking_contacts_shaped_force = 1.0
            tracking_contacts_shaped_vel = 1.0
            feet_clearance_cmd_linear = -30.0  # 沿用 WTW 足端高度约束，防止无意义拖脚/乱抬脚
            jump = 2.0  # 奖励四脚处于一致接触状态：要么四脚都触地蓄力/落地，要么四脚都离地

            # 跳跃结构奖励：目标是“更高、更快、更远”的连续四脚同步前向跳。
            # jump_air_time = 0.0
            # jump_flight_phase_air = 0.8
            jump_mixed_contact = -1.5  # 惩罚 1-3 只脚接触的半同步状态，减少前后腿分裂落地/起跳
            jump_landing_async = -4.0  # 首次落地时惩罚未四脚同时接触
            jump_hind_first_landing = -0.5  # 前跳时轻量惩罚后脚先落地，补充四脚同步落地约束
            jump_landing_stable = 1.5  # 落地/恢复相保持水平、抑制晃动，同时继续跟踪 x/y/yaw 命令
            jump_height = 4.0  # 四脚都离地时奖励 base 高度达到目标，直接推动跳得更高
            jump_z_vel = 0.4  # 四脚都离地时奖励向上 z 速度，辅助提高腾空上升段
            jump_takeoff_z_vel = 2.0  # 每个蹬伸 push 相位奖励向上起跳速度，是提高爆发力的主项
            jump_takeoff_x_vel = 0.6  # 先减小前向起跳奖励，等四腿同步稳定后再逐步加回
            jump_push_pitch = -1.6  # 蹬地阶段先压住俯仰角动量，避免把后仰趋势带入空中
            # jump_push_force_balance = -0.3  # push 阶段四脚垂直力要均衡，避免只有后腿出力
            # jump_push_front_hind_sync = -2.0  # 压缩/蹬伸阶段前后腿关节变化同步，强制前腿参与蹬地
            # jump_flight_orientation = -0.0  # 四脚离地时压住机身 roll/pitch，减少空中飘和翻身趋势
            # jump_flight_ang_vel_xy = -0.5  # 已合并到 jump_flight_pitch_stable，避免重复惩罚
            jump_flight_pitch_stable = -3.5  # 参考飞行相和真实腾空期都压住 roll/pitch 晃动
            # jump_landing_knee_bend = -2.0  # 落地/恢复阶段惩罚膝关节过度弯曲，避免落地完全蹲下
            jump_landing_force_balance = -0.4  # 落地阶段惩罚四脚受力不均
            # jump_hipx_landing = -0.8  # 落地/恢复阶段压住髋外展，减少前腿外八和侧向扭身
            jump_leg_symmetry = -0.6  # 仅在真实腾空时轻量约束左右后腿，压掉右后腿最高点单独二次蹬腿


        reward_curriculum = False
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        only_positive_rewards = True  # 总奖励小于0时裁剪到0；负奖励仍会先抵消本步正奖励
        tracking_sigma = 0.20  # 跟踪奖励的高斯分布标准差 = exp(-error^2 / sigma)
        soft_dof_pos_limit = 0.95   # 关节位置软限位：关节角度超过URDF限位95%时触发惩罚。调低（如0.9）可提前约束
        soft_dof_vel_limit = 0.95   # 关节速度软限位：超过最大速度95%时惩罚。保护电机模型不过载
        soft_torque_limit = 0.95    # 关节力矩软限位：超过额定扭矩95%时惩罚。防止仿真数值发散
        # CC1 电机 TN 曲线参数。单位需要与仿真一致：扭矩 Nm，关节速度 rad/s。
        # 奖励函数采用线性包络：allowed_torque = torque_limit * (1 - abs(dof_vel) / speed_limit)。
        tn_curve_torque_limits = {"hipx": 49.0, "hipy": 49.0, "knee": 59.0}
        tn_curve_speed_limits = {"hipx": 29.0, "hipy": 29.0, "knee": 21.0}
        tn_curve_soft_ratio = 0.95  # 留 5% 安全余量，避免策略贴着极限学
        base_height_target_vel = 0.6  # base高度目标速度：base高度与目标高度的差值超过该速度时触发惩罚，鼓励平稳升降
        base_height_target = 0.42  # 机身目标高度
        feet_height_target_base = -0.25  # 足部距base的 相对距离目标（抬脚高度为0.15 以适应台阶地形）
        feet_height_target_terrain = 0.15  # 足部离地高度目标
        max_contact_force = 100.    # 四足接触力 > 100N 时触发惩罚的阈值
        target_foot_height = 0.1  # WTW 通用足端高度目标，当前不是跳高主控制量
        target_foot_height_yaw = 0.1  # 原地转向足端高度目标；本任务 yaw 命令为 0，基本不生效
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.

        # 连续跳奖励门控：只要速度/yaw 命令足够大，jump 类奖励就持续生效。
        # jump_contact_force_threshold = 5.0
        jump_min_command_speed = 0.05  # 速度命令超过该值才启用 jump 类奖励
        jump_min_yaw_speed = 0.05  # yaw 命令超过该值也可启用 jump 类奖励
        straight_min_forward_speed = 0.20
        straight_full_forward_speed = 1.0
        straight_cmd_y_threshold = 0.15
        straight_cmd_yaw_threshold = 0.15
        # jump_sync_air_only = False

        # 跳高和起跳速度目标。min 是开始给分的位置，target 是满分附近的目标。
        jump_height_min = 0.42  # base 高度超过该值后，jump_height 开始给正奖励
        jump_height_target = 0.50  # base 高度达到/超过该值时，jump_height 基本满分
        jump_z_vel_min = 0.2  # 腾空阶段 z 速度超过该值后开始奖励
        jump_z_vel_target = 1.2  # 腾空阶段 z 速度满分目标
        jump_takeoff_z_vel_min = 0.15  # push 相位向上起跳速度开始给分
        jump_takeoff_z_vel_target = 1.65  # push 相位向上起跳速度满分目标，调大可提高爆发但更容易摔
        jump_takeoff_x_vel_min = 0.2  # push 相位前向速度开始给分
        jump_takeoff_x_vel_target = 1.2  # 先稳住姿态，后续想跳更远再逐步加到 1.2~1.4
        jump_takeoff_track_command_direction = True  # 前后/左右均沿命令方向奖励，纯转向时不额外向前推
        jump_push_pitch_weight = 2.0
        jump_push_pitch_ang_vel_weight = 1.0  # 起跳时重点抑制俯仰角速度，减少空中持续后仰
        # jump_push_balance_force_norm = 120.0
        jump_flight_pitch_deadband = 0.03  # 允许约2度空中俯仰误差
        jump_flight_pitch_vel_deadband = 0.18  # 允许少量 pitch 角速度
        jump_flight_pitch_use_phase_mask = False  # 按真实四脚腾空触发
        jump_flight_pitch_weight = 1.0
        jump_flight_roll_weight = 0.8  # 空中左右歪斜惩罚，压住跳起后的左右晃
        jump_flight_roll_deadband = 0.03
        jump_flight_backward_pitch_weight = 2.5  # 额外惩罚后仰，所有跳跃方向都生效
        jump_flight_backward_pitch_deadband = 0.02
        jump_flight_backward_pitch_direction = -1.0  # projected_gravity[:,0] < 0 视为后仰
        jump_flight_pitch_vel_weight = 0.75
        jump_flight_backward_pitch_vel_weight = 1.0  # 额外压住继续后仰的 pitch 角速度
        jump_flight_backward_pitch_vel_deadband = 0.08
        jump_flight_backward_pitch_vel_direction = -1.0
        jump_flight_backward_pitch_delta_weight = 1.0  # 惩罚单个控制步内突然增加的抬头角速度
        jump_flight_backward_pitch_delta_deadband = 0.06  # 允许少量自然变化，单位为每个policy步的rad/s变化量
        jump_flight_apex_z_vel_window = 0.30  # 世界坐标z速度进入±0.30m/s时视为接近真实最高点
        jump_flight_apex_pitch_vel_extra_scale = 1.25  # 最高点将pitch角速度及突变惩罚平滑增强到最多2.25倍
        jump_flight_roll_vel_weight = 0.7  # 空中 roll 角速度惩罚，专门减小左右摇摆速度
        jump_flight_roll_vel_deadband = 0.18
        jump_leg_symmetry_hind_only = True  # 不锁死四条腿，只比较左右后腿的HipY/Knee
        jump_leg_symmetry_flight_only = True  # 支撑和落地阶段允许两条后腿独立适应接触
        jump_leg_symmetry_pos_deadband = 0.05  # 允许约0.05rad自然关节差异
        jump_leg_symmetry_vel_deadband = 0.5  # 超过0.5rad/s的左右后腿速度差才开始惩罚
        jump_leg_symmetry_vel_weight = 0.05  # 速度差只作轻量约束，避免压制正常收腿
        # jump_forward_landing_min_command = 0.15
        # jump_forward_landing_full_command = 0.8
        jump_landing_stable_lin_weight = 1.0
        jump_landing_stable_ang_weight = 0.6
        jump_landing_stable_yaw_weight = 0.5
        jump_landing_stable_tilt_weight = 2.5
        jump_landing_stable_sigma = 0.35

        # jump_landing_knee_max = 1.55  # 落地/恢复阶段允许的最大膝关节弯曲，超过后按平方惩罚
        # jump_landing_balance_force_norm = 120.0  # 四脚落地力均衡归一化，越小越严格
        # jump_takeoff_require_contact = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Cc1JumpwtwCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01  # 熵系数（鼓励探索）

        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'jump_wtw_cc1'
        run_name = ''
        # load and resume
        # resume = True  # whether to load an existing model and continue training
        # load_run = osp.join(logs_root, 'jump_cc1', 'Apr02_18-06-36_')
        resume = False
        load_run = -1
        checkpoint = -1  # -1 = last saved model
