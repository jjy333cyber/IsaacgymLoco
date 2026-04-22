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

from legged_gym.envs.lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO


class Lite3StairsCfg( Lite3RoughCfg ):
    """Lite3 台阶地形任务配置（基于 Lite3RoughCfg 只覆盖少量参数）。"""

    class init_state( Lite3RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.35]  # 初始位置 x,y,z [m]
        default_joint_angles = {  # 零动作时的目标关节角 [rad]
            'FL_HipX_joint': 0.0,  # 左前髋 X
            'HL_HipX_joint': 0.0,  # 左后髋 X
            'FR_HipX_joint': 0.0,  # 右前髋 X
            'HR_HipX_joint': 0.0,  # 右后髋 X

            'FL_HipY_joint': -0.8,  # 左前髋 Y
            'HL_HipY_joint': -0.8,  # 左后髋 Y
            'FR_HipY_joint': -0.8,  # 右前髋 Y
            'HR_HipY_joint': -0.8,  # 右后髋 Y

            'FL_Knee_joint': 1.6,  # 左前膝
            'HL_Knee_joint': 1.6,  # 左后膝
            'FR_Knee_joint': 1.6,  # 右前膝
            'HR_Knee_joint': 1.6,  # 右后膝
        }

    class terrain( Lite3RoughCfg.terrain ):
        mesh_type = 'trimesh'  # 地形网格类型：none/plane/heightfield/trimesh
        horizontal_scale = 0.1  # 水平方向分辨率 [m]
        vertical_scale = 0.005  # 垂直方向分辨率 [m]
        border_size = 15  # 场地边界缓冲区 [m]
        curriculum = True  # 是否启用地形课程学习
        static_friction = 1.0  # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.  # 弹性系数
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 10.  # 单块地形长度 [m]
        terrain_width = 10.  # 单块地形宽度 [m]
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        terrain_proportions = [0.0, 0.0, 0.1, 0.1, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands( Lite3RoughCfg.commands ):
        curriculum = True  # 是否对命令范围做课程学习
        max_forward_curriculum = 1.5  # 前进速度上限（课程上限）
        max_backward_curriculum = 1.0  # 倒退速度上限（课程上限）
        max_lat_curriculum = 1.0  # 侧向速度上限（课程上限）
        num_commands = 4  # 命令维度：x速度、y速度、yaw角速度、heading
        resampling_time = 10.  # 命令重采样间隔 [s]
        heading_command = True  # 是否使用 heading 命令（由 heading 误差计算 yaw 角速度）

        class ranges( Lite3RoughCfg.commands.ranges ):
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-math.pi, math.pi]

    class termination:
        base_vel_violate_commands = True  # 是否用“机身速度偏离命令过大”作为终止条件
        out_of_border = True  # 是否走出边界即终止
        fall_down = True  # 是否摔倒/跌落即终止

    class domain_rand( Lite3RoughCfg.domain_rand ):
        randomize_payload_mass = True  # 是否随机附加载荷质量
        payload_mass_range = [0.0, 3.0]  # 附加载荷质量范围 [kg]

        randomize_com_displacement = True  # 是否随机质心偏移
        com_displacement_range = [-0.05, 0.05]  # 质心偏移范围 [m]

        randomize_link_mass = False  # 是否随机各 link 质量
        link_mass_range = [0.9, 1.1]  # link 质量缩放范围

        randomize_friction = True  # 是否随机摩擦系数
        friction_range = [0.2, 1.25]  # 摩擦系数范围

        randomize_restitution = False  # 是否随机弹性系数
        restitution_range = [0., 1.0]  # 弹性系数范围

        randomize_motor_strength = True  # 是否随机电机强度（动作输出缩放）
        motor_strength_range = [0.9, 1.1]  # 电机强度缩放范围

        randomize_kp = True  # 是否随机 PD 的 Kp
        kp_range = [0.8, 1.2]  # Kp 缩放范围

        randomize_kd = True  # 是否随机 PD 的 Kd
        kd_range = [0.8, 1.2]  # Kd 缩放范围

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

        dof_init_pos_ratio_range = [0.8, 1.2]  # 关节初始位置缩放范围

        randomize_dof_vel = False  # 是否随机关节初始速度
        dof_init_vel_range = [-0.1, 0.1]  # 关节初始速度范围

        disturbance = True  # 是否施加随机外力扰动
        disturbance_range = [-30.0, 30.0]  # 扰动力范围 [N]
        disturbance_interval = 8  # 扰动施加间隔（单位：秒/或按实现定义）

        push_robots = True  # 是否周期性推机器人（速度脉冲）
        push_interval_s = 16  # 推动间隔 [s]
        max_push_vel_xy = 1.  # 最大水平推送速度 [m/s]

        delay = True  # 是否对动作引入随机延迟

    class rewards( Lite3RoughCfg.rewards ):
        class scales:
            # general
            termination = -100.  # 终止惩罚：触发终止条件时给予大负奖励
            # velocity-tracking
            tracking_lin_vel = 2.0  # 线速度跟踪奖励（对齐命令 v_x,v_y）
            tracking_ang_vel = 2.0  # 角速度跟踪奖励（对齐命令 yaw 角速度/heading）
            # root
            lin_vel_z = -2.0  # 惩罚机身竖直速度（抑制跳动/抖动）
            ang_vel_xy = -0.05  # 惩罚机身 roll/pitch 角速度（抑制翻滚）
            orientation = -0.5  # 惩罚机身姿态偏离（保持躯干水平） -0.2
            base_height = -5.0  # 惩罚机身高度偏离目标（台阶任务更强约束） -5.0
            # joint
            torques = -0.0001  # 惩罚关节力矩（节能/平滑）
            torque_limits = -0.0  # 关节力矩超限惩罚（未启用时为 0）
            dof_vel = -0.0  # 关节速度惩罚（可抑制高速摆动；此处关闭）
            dof_acc = -2.5e-7  # 关节加速度惩罚（平滑动作/减小冲击）
            stand_still = -0.5  # 在“站立命令”时惩罚移动（防止抖动走动） -0.1
            hip_pos = -0.2  # 髋关节偏离默认位姿惩罚（稳姿/防外八内八） -0.12
            thigh_pose = -0.1  # 大腿关节姿态约束惩罚（稳定腿型） -0.05
            calf_pose = -0.1  # 小腿关节姿态约束惩罚（稳定腿型） -0.03
            dof_pos_limits = -0.0  # 关节位置接近硬限位惩罚（此处关闭） -0.0
            dof_vel_limits = -0.0  # 关节速度接近限位惩罚（此处关闭） -0.0
            joint_power = -3e-5  # 关节功率惩罚（|tau * qd|，能耗相关） -3e-5
            # feet_mirror = -0.1  # 四足步态对称/镜像约束（减小左右不一致） -0.05
            # action
            action_rate = -0.05  # 动作变化率惩罚（抑制相邻时刻动作突变） -0.05
            smoothness = -0.02  # 动作平滑项（更强约束/去抖） -0.02
            hip_action_magnitude = -0.01  # 髋关节动作幅度惩罚（此处关闭） -0.0
            # contact
            collision = -5.0  # 非期望部位碰撞惩罚（如躯干/大腿触地） -5.0
            feet_contact_forces = -0.0005  # 足端接触力过大惩罚（抑制“砸地”） -0.00015
            # others
            feet_air_time = 0.5  # 鼓励足端腾空时间（促进迈步/跨越） 0.25
            has_contact = 2.0  # 鼓励保持接触（提升稳定性，避免全腾空） 2.0
            feet_stumble = -2.0  # 绊倒/磕碰惩罚（足端撞到台阶边缘等） -2.0
            feet_slide = -0.02  # 足端打滑惩罚（接触时水平滑移）-0.01
            feet_clearance_base = -0.0  # 足端相对机身抬脚高度奖励（此处关闭）-0.0
            feet_clearance_terrain = -0.0  # 足端相对地形抬脚高度奖励（此处关闭）-0.0
            feet_yaw_clearance_terrain = 2.0  # base 原地旋转时鼓励抬脚，减少“拧着走/刮地” 1.0
            stuck = -1.  # 卡住/不动惩罚（速度过小或长期无进展） -1.0
            upward = 0.0  # 向上运动奖励（此处关闭/预留） 0.0
            # feet
            trot = 0.8
            feet_clearance_swing = 0.4  # feet clearance can increase for more   
            # default_hip_pos = -0.2
            # default_pos = -0.1

        only_positive_rewards = False  # 是否将总奖励裁剪为非负（台阶任务通常保留负奖励更稳定）
        tracking_sigma = 0.20  # 跟踪奖励的高斯 sigma（reward = exp(-err^2/sigma)），越小越“严格”
        soft_dof_pos_limit = 0.95  # 关节位置软限位（按 URDF 上下限比例）
        soft_dof_vel_limit = 0.95  # 关节速度软限位比例
        soft_torque_limit = 0.95  # 关节力矩软限位比例
        base_height_target = 0.47  # 机身目标高度 [m]
        feet_height_target_base = -0.32  # 足端相对 base 的目标高度 [m]
        feet_height_target_terrain = 0.15  # 足端离地目标高度 [m]
        max_contact_force = 100.  # 接触力惩罚阈值 [N]（超过开始惩罚/截断）
        target_foot_height = 0.1  # feet height
        cycle_time = 0.5


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")  # 训练日志根目录


class Lite3StairsCfgPPO( Lite3RoughCfgPPO ):

    class runner( Lite3RoughCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        # algorithm_class_name = 'HybridPPO'
        num_steps_per_env = 100  # 每次迭代每个环境采样步数
        max_iterations = 40000  # 最大迭代次数（策略更新次数）

        # logging
        save_interval = 200  # 保存间隔（每 N 次迭代检查保存）
        experiment_name = 'stairs_lite3'  # 实验名称（日志目录名）
        run_name = ''  # 运行名称（可为空）
        # load and resume
        resume = True  # 是否从历史 run 恢复
        # load_run = osp.join(logs_root, 'flat_lite3', 'Mar09_09-47-04_')  # 指定恢复的 run 目录
        load_run = osp.join(logs_root, 'stairs_lite3', 'Mar11_09-16-10_')  # 指定恢复的 run 目录
        checkpoint = -1  # checkpoint 索引（-1 表示最新）
