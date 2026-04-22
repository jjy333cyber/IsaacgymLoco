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
import math  # 数学库
from os import path as osp
from legged_gym.envs.lite3.lite3_config import Lite3RoughCfg, Lite3RoughCfgPPO

class Lite3RoughRecoverCfg( Lite3RoughCfg ):
    """Lite3 倒地恢复（recover）任务配置。"""

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
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [flat, rough, smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles, stepping_stones, pit, gap]
        terrain_proportions = [0.5, 0.5]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands( Lite3RoughCfg.commands ):
        curriculum = True
        max_forward_curriculum = 2.0  # x_vel 限制 [-1.0, 2.0]
        max_backward_curriculum = 1.0
        max_lat_curriculum = 1.0  # y_vel 限制 [-1.0, 1.0]
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error

        class ranges( Lite3RoughCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-math.pi, math.pi]

    class asset( Lite3RoughCfg.asset ):
        penalize_contacts_on = ["THIGH", "SHANK", "TORSO"]  # 触地惩罚部位
        terminate_after_contacts_on = []  # 倒地恢复任务：不因躯干触地直接终止
        self_collisions = 0  # 恢复任务可开启自碰撞（0=启用）

    class termination( Lite3RoughCfg.termination ):
        base_vel_violate_commands = False

    class domain_rand ( Lite3RoughCfg.domain_rand ):
        base_init_rot_range = dict(
            roll=[-3.14, 3.14],
            pitch=[-3.14, 3.14],
            yaw=[-3.14, 3.14],
        )

        recover_mode = True  # 开启倒地恢复模式

    class rewards( Lite3RoughCfg.rewards ):
        class scales:
            # general
            termination = -0.0
            # velocity-tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            # root
            lin_vel_z_up = -2.0
            ang_vel_xy_up = -0.05
            orientation_up = -2.0  # base 非水平姿态 惩罚
            base_height_up = -5.0
            # joint
            torques = -0.0002
            torque_limits = -0.0
            dof_vel = -0.0
            dof_acc = -2.5e-7
            stand_nice = -0.1  # (base原地不动 或 原地旋转) 且 重力投影向下时 的 关节位置与默认关节位置的 偏差 惩罚
            hip_pos_up = -0.3
            thigh_pose_up = -0.05
            calf_pose_up = -0.05
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            joint_power = -2e-5
            feet_mirror_up = -0.05
            # action
            action_rate = -0.02
            smoothness = -0.01
            hip_action_magnitude = -0.01
            # contact
            collision_up = -0.0
            feet_contact_forces = -0.00015
            # others
            feet_air_time = 0.25
            has_contact = 0.3  # 摔倒恢复训练时可开启
            feet_stumble_up = -0.0
            feet_slide_up = -0.01
            feet_clearance_base_up = -0.1
            feet_clearance_terrain_up = -0.0
            feet_yaw_clearance_terrain = 1.0  # (base原地旋转) 时 脚抬起
            stuck = -0.05
            upward = 1.0  # 摔倒恢复训练时可开启

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.43
        feet_height_target_base = -0.27
        feet_height_target_terrain = 0.15
        max_contact_force = 100.  # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Lite3RoughRecoverCfgPPO( Lite3RoughCfgPPO ):

    class runner( Lite3RoughCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100  # per iteration
        max_iterations = 2000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'recover_lite3'
        run_name = ''
        # load and resume
        # resume = True
        # load_run = osp.join(logs_root, 'flat_lite3', 'Jul23_11-53-29_init0.1')  # -1 = last run
        checkpoint = -1  # -1 = last saved model