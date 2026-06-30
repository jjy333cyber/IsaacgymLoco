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
import glob

from os import path as osp
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, MOTION_FILES_DIR

PROJECT_ROOT_DIR = MOTION_FILES_DIR.parent
MOTION_FILES = []
MOTION_FILES.extend(glob.glob(str(PROJECT_ROOT_DIR / "cc1_manzou0615/*.txt")))

class Cc1RoughwtwAmpCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096  # 并行仿真的环境数量（需根据GPU显存调整）
        num_one_step_observations = 61  # 单步观测向量维度（原始传感器数据）
        num_observations = num_one_step_observations * 6    # 总观测向量维度（含6步历史）
        # num_one_step_privileged_obs = 61 + 3 + 3 + 187  # 单步特权观测向量维度，（+3维线速度 + 3维随机扰动力 + 地形扫描(187))
        num_one_step_privileged_obs = 61 + 3 + 3
        num_privileged_obs = num_one_step_privileged_obs * 1    # 总特权观测向量维度
        num_actions = 12  # 动作空间维度（12个关节）
        env_spacing = 3.  # 环境之间的间距（单位：米），not used with heightfields/trimeshes
        send_timeouts = True  # 是否发送超时信号给算法，send time out information to the algorithm
        episode_length_s = 20  # 单次训练Episode的时长（秒），episode length in seconds
        observe_gait_commands = True  # 是否观察gait commands，observe gait commands in the observations
        using_amp = True  # AMP: return AMP states for HybridPolicyRunner

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38]  # 零速站立从目标高度开始，避免reset后先塌低
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # action = 0.0，即零动作时的目标关节角度（站立姿态）
            'FL_HipX_joint': 0.0,
            'HL_HipX_joint': 0.0,
            'FR_HipX_joint': 0.0,
            'HR_HipX_joint': 0.0,

            # 'FL_HipY_joint': -0.4,
            # 'HL_HipY_joint': -0.4,
            # 'FR_HipY_joint': -0.4,
            # 'HR_HipY_joint': -0.4,

            # 'FL_Knee_joint': 0.8,
            # 'HL_Knee_joint': 0.8,
            # 'FR_Knee_joint': 0.8,
            # 'HR_Knee_joint': 0.8,

            'FL_HipY_joint': -0.5,
            'HL_HipY_joint': -0.5,
            'FR_HipY_joint': -0.5,
            'HR_HipY_joint': -0.5,

            'FL_Knee_joint': 1.0,
            'HL_Knee_joint': 1.0,
            'FR_Knee_joint': 1.0,
            'HR_Knee_joint': 1.0,
        }

    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'plane'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 15 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
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
        # terrain_proportions = [0.4, 0.2, 0.2, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'  # 控制类型（P=位置控制，T=力矩控制）
        stiffness = {'joint': 30.0} # 关节刚度（单位：N·m/rad）
        damping = {'joint': 1.0}    # 关节阻尼（单位：N·m·s/rad）
        action_scale = 0.25  # 动作缩放因子（目标角度 = 动作 * scale + 默认角度）
        decimation = 4      # 每个policy DT 包含的 sim DT 的个数
        hip_reduction = 1   # 髋关节扭矩缩放因子（用于平衡前后腿负载）

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_forward_curriculum = 0.5
        max_backward_curriculum = 0.0
        max_lat_curriculum = 0.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        stand_still_command = True
        Rotate_command = False
        pacing_offset = False

        # 环境中加入急停命令：在训练过程中，随机生成急停命令（线速度和角速度突然变为零），迫使机器人快速适应突发情况，提高鲁棒性。急停命令的生成频率、持续时间和触发条件可以根据需要进行调整。
        sudden_stop_command = False
        sudden_stop_interval_s = 2.0
        sudden_stop_env_ratio_range = [0.10, 0.20]
        sudden_stop_duration_s = [0.4, 1.0]
        sudden_stop_min_speed = 0.25
        sudden_stop_min_yaw_speed = 0.25
        sudden_stop_min_episode_time_s = 1.0

        # walk.bvh 落脚顺序 HL -> FL -> HR -> FR；降低频率并加长支撑相，减少训练后落地频率过高。
        frequencies = 1.2
        phases = 0.5
        offsets = 0.2
        bounds = 0.0
        durations = 0.65

        class ranges( LeggedRobotCfg.commands.ranges ):
            # 第一阶段只在专家平均速度0.377m/s附近学习直线前进。
            lin_vel_x = [0.0, 0.4]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
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
        base_vel_violate_commands = False  # 是否在终止条件中考虑当地形等级>3时，base速度与命令速度差异过大(超过2m/s)（摔倒恢复训练关闭）

        out_of_border = False  # 平地 plane 不创建 self.terrain，不能使用地形边界终止

        fall_down = True  # 是否在终止条件中考虑跌落(base的z方向线速度 < -5)

    class domain_rand:
        # startup
        randomize_payload_mass = True  # 是否随机增加 base 的质量
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
        randomize_friction = True  # 是否随机化env各刚体部位的摩擦系数
        friction_range = [0.2, 1.25]

        randomize_restitution = False  # 是否随机化env各刚体部位的弹性系数
        restitution_range = [0., 1.0]

        # reset
        randomize_motor_strength = True  # 是否随机化env的电机强度（输出的actions *）
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True  # 是否 随机改变PD控制器的p增益（stiffness）
        kp_range = [0.9, 1.1]

        randomize_kd = True  # 是否 随机改变PD控制器的D增益（damping）
        kd_range = [0.9, 1.1]

        # 重置时随机改变base的位置（初始位置 +），默认x,y方向为 [-1, 1]，z方向为 0，若更改则为下面的
        base_init_pos_range = dict(
            x=[-1.0, 1.0],
            y=[-1.0, 1.0],
            z=[0.0, 0.05],
        )
        # 重置时随机设置base的方向（摔倒恢复模式都设为 [-3.14, 3.14]）
        base_init_rot_range = dict(
            roll=[-0.2, 0.2],
            pitch=[-0.2, 0.2],
            yaw=[-0.0, 0.0],
        )
        # 重置时随机设置base的线速度、角速度，默认x,y,x,rool,pitch,roll方向为 [-0.5, 0.5]，若更改则为下面的
        base_init_vel_range = dict(
            x=[-0.2, 0.2],
            y=[-0.2, 0.2],
            z=[-0.15, 0.15],
            roll=[-0.2, 0.2],
            pitch=[-0.2, 0.2],
            yaw=[-0.2, 0.2],
        )

        dof_init_pos_ratio_range = [0.8, 1.2]  # 重置时随机改变关节初始位置（初始关节位置 *），默认为 [0.5, 1.5]

        randomize_dof_vel = False  # 重置时设置关节初始速度
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
        class scales:
            # general
            termination = -100.  # 仿真终止时的惩罚：未启用。设为负值（如-10.0）可在跌倒时给予额外惩罚
            
            # velocity-tracking
            tracking_lin_vel = 3.0  # 速度跟踪略降，避免为了追速度把前身压低
            tracking_ang_vel = 0.5  # commands 中yaw方向的角速度跟踪奖励

            # 新加
            straight_lin_vel_y = -2.0  # 前进直行命令时抑制横向漂移
            straight_yaw_vel = -2.0  # 前进直行命令时抑制偏航漂移
            forward_backward_orientation = -5.0  # 前进/后退都压住俯仰失衡

            # root
            lin_vel_z = -2.0  # base 的 Z 轴线速度惩罚：防止机身跳跃
            ang_vel_xy = -0.15  # 移动时进一步抑制roll/pitch角速度，减少左右摆和前后点头
            orientation = -5.0  # 全局机身水平约束，AMP专家本身几乎是平的
            base_height = -20.0  # 加强全局高度约束，防止低趴局部最优
            moving_base_height = -15.0  # 运动时额外防止机身下沉，只惩罚低于目标高度
            stand_base_height = -18.0  # 零速站立专门惩罚低于目标高度，避免原地低趴
            moving_orientation_flat = -5.0  # 移动时显式压住pitch/roll，保证基座是平的

            # 新加
            stop_orientation = -5.0  # 零速/急停时额外压住身体前倾和侧倾，重点解决停下沉头
            stop_ang_vel_xy = -1.0  # 零速/急停时抑制roll/pitch角速度，减少刹停点头

            # joint
            torques = -0.0001  # 关节扭矩过大惩罚
            # torque_limits = -0.0  # 关节扭矩接近极限惩罚
            # dof_vel = -0.0  # 关节速度过大惩罚
            dof_acc = -2.5e-7  # 关节加速度惩罚（若步态抖动，可增大惩罚）
            stand_still = -0.5  # (base原地不动或原地旋转) 时的关节位置与默认关节位置的偏差惩罚
            stand_dof_vel = -0.03  # 零速时惩罚关节速度，避免站立时单腿持续摆动
            hip_pos = -0.4  # 适度限制HipX偏离默认位置，辅助抑制pace单侧支撑时四腿向内夹
            # thigh_pose = -0.01
            # calf_pose = -0.02
            # dof_pos_limits = -0.0  # 关节位置接近极限惩罚
            # dof_vel_limits = -0.0  # 关节速度接近极限惩罚
            joint_power = -2e-5  # 关节高功率惩罚：降低能耗（需平衡运动效率，过高惩罚会导致动作迟缓）
            # feet_mirror = -0.05  # 斜对称腿的关节位置偏差惩罚
            
            # action
            action_rate = -0.02  # action变化惩罚
            smoothness = -0.01  # action二阶平滑性惩罚（复杂地形，可适当降低）
            # hip_action_magnitude = -0.00  # action 中的髋关节hip（0,3,6,9）动作幅度惩罚（防止 > 1.0）
            
            # contact
            collision = -10.0  # 指定关节的碰撞 惩罚。检测超过 max_contact_force (100N) 的接触，设为负值（如-0.1）可防硬件过载
            feet_contact_forces = -0.00015  # 四足的接触力 > 100N 惩罚
            
            # others
            # feet_air_time = 1.0  # 四足的空中时间接近 0.5s 奖励 (原地不动时除外)
            # feet_air_time_variance_velocity = -0.0
            has_contact = 4.0  # 零速只保留适度触地奖励，避免为了四脚贴地而低趴
            # feet_stumble = -0.0  # 四足接触到垂直表面惩罚
            feet_slide = -0.08  # -0.05  # 脚接触地面具有相对base的速度惩罚
            # feet_clearance_base = -0.2  # 四足距base目标距离惩罚
            # feet_clearance_terrain = -0.0  # 大速度下 四足离地目标高度惩罚
            # feet_yaw_clearance_terrain = 2.0  # (base原地旋转) 时脚抬起
            stuck = -3.  # base 卡住惩罚
            # upward = 0.0  # 重力投影向下奖励（恢复训练时开启）
            
            # feet
            raibert_heuristic = 0.0  # -10.0  # Raibert启发式奖励：根据当前base速度和步态周期计算理想的足部位置，奖励与理想位置的接近程度。鼓励足部在适当位置着地以稳定运动。
            # tracking_contacts_shaped_force = 1.0
            # tracking_contacts_shaped_vel = 1.0
            tracking_contacts_shaped_force_exp = 0.5  # AMP行走先放软时钟相位约束，减少腿被拉直/强制高抬
            tracking_contacts_shaped_vel_exp = 0.5  # 支撑脚低滑动仍保留，但不压过专家动作
            feet_clearance_cmd_linear = -0.5  # 降低足端高度约束权重，避免前腿抬得过高

            anti_trot_diagonal_swing = -0.8  # 惩罚对角腿同时摆动，避免AMP行走塌成trot
            swing_contact = -2.0  # 明确摆动期提前触地惩罚，降低一步内点地频率


        reward_curriculum = False
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        only_positive_rewards = False  # 需要让零速摆腿、过高站姿等负奖励真正生效
        tracking_sigma = 0.20  # 跟踪奖励的高斯分布标准差 = exp(-error^2 / sigma)
        soft_dof_pos_limit = 0.95   # 关节位置软限位：关节角度超过URDF限位95%时触发惩罚。调低（如0.9）可提前约束
        soft_dof_vel_limit = 0.95   # 关节速度软限位：超过最大速度95%时惩罚。保护电机模型不过载
        soft_torque_limit = 0.95    # 关节力矩软限位：超过额定扭矩95%时惩罚。防止仿真数值发散
        base_height_target = 0.42   # 零速目标高度抬高，避免原地站立时机身压得过低
        stand_base_height_target = 0.44  # 零速站立时目标
        moving_base_height_target = 0.44  # 移动时保持高而平，避免前身沉下去
        moving_height_min_lin_cmd = 0.05
        moving_height_full_lin_cmd = 0.35
        moving_height_min_yaw_cmd = 0.05
        moving_height_full_yaw_cmd = 0.35
        feet_height_target_base = -0.28  # 足部距base的 相对距离目标（抬脚高度为0.15 以适应台阶地形）
        feet_height_target_terrain = 0.04  # 足部离地高度目标，避免前腿显式抬得过高
        max_contact_force = 100.    # 四足接触力 > 100N 时触发惩罚的阈值
        target_foot_height = 0.05  # 降低显式摆腿高度，避免前腿越训越高
        target_foot_height_yaw = 0.05  # feet height
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.

        # 后退时额外抑制身体后仰/侧翻：当命令的线速度为负（后退）且超过一定阈值时，增加对身体姿态的惩罚。
        # backward_orientation_min_speed = 0.05
        # backward_orientation_full_speed = 1.0
        forward_pitch_deadband = 0.015  # 更早惩罚低头前倾，目标是基座尽量水平
        forward_pitch_direction = 1.0  # 若实际日志里前倾时 projected_gravity[:,0] 为负，则改成 -1.0
        forward_nose_up_weight = 0.1  # 轻量压住后沉/抬头，避免再用大权重把策略推成沉头
        moving_orientation_pitch_deadband = 0.010
        moving_orientation_roll_deadband = 0.012
        moving_orientation_pitch_weight = 3.5
        moving_orientation_roll_weight = 2.5

        # 新加零速/急停奖励：当线速度和角速度都接近零时，额外奖励身体保持稳定（不前倾/后仰/侧翻）。可提高刹停时的稳定性和安全性。
        zero_command_lin_vel_threshold = 0.20
        zero_command_yaw_vel_threshold = 0.20
        stop_pitch_weight = 4.0
        stop_roll_weight = 1.2

        # 新加前进直行约束：只在 cmd_x 足够大、cmd_y/cmd_yaw 接近 0 时生效。
        straight_min_forward_speed = 0.20
        straight_full_forward_speed = 1.0
        straight_cmd_y_threshold = 0.15
        straight_cmd_yaw_threshold = 0.15

        # pace 防蹭地/防二次触地参数
        anti_trot_contact_threshold = 1.0
        lateral_pair_contact_threshold = 2.0
        swing_contact_threshold = 0.3

        lateral_pair_touchdown_cooldown_s = 0.20
        short_contact_min_time_s = 0.10
        front_contact_min_time_s = 0.12
        hind_contact_min_time_s = 0.12
        front_short_contact_weight = 1.3
        hind_short_contact_weight = 1.2
        lateral_pair_touchdown_phase_window = 0.10
        lateral_pair_takeoff_phase_window = 0.08

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
class Cc1RoughwtwAmpCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'HybridPolicyRunner'

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
        desired_kl = 0.005
        max_grad_norm = 1.
        amp_replay_buffer_size = 1000000

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HybridPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'flat_wtw_amp_cc1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        # resume = True
        # load_run = osp.join(logs_root, 'flat_cc1', 'Apr03_14-57-33_')
        checkpoint = -1  # -1 = last saved model

        amp_reward_coef = 0.02
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]
        min_normalized_std = [0.05, 0.02, 0.05] * 4
