import math
from os import path as osp
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.cc1.cc1_config import Cc1RoughCfg


class Cc1Jump1wtwCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # 并行仿真的环境数量（需根据GPU显存调整）
        num_one_step_observations = 64  # 原 cc1_jump_wtw 为 61；新增 jump_target_x/y 和 jump_flag 三个 command 输入
        num_observations = num_one_step_observations * 6  # 总观测向量维度（含6步历史）
        num_one_step_privileged_obs = 64 + 3 + 3 + 187  # 单步特权观测：（obs + 线速度 + 随机扰动力 + 地形扫描）
        num_privileged_obs = num_one_step_privileged_obs * 1
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 2.0
        observe_gait_commands = False  # jump1 使用显式 jump_flag 状态机，不再依赖周期 gait clock

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.36]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_HipX_joint': 0.0,
            'HL_HipX_joint': 0.0,
            'FR_HipX_joint': 0.0,
            'HR_HipX_joint': 0.0,

            'FL_HipY_joint': -0.5,
            'HL_HipY_joint': -0.5,
            'FR_HipY_joint': -0.5,
            'HR_HipY_joint': -0.5,

            'FL_Knee_joint': 1.0,
            'HL_Knee_joint': 1.0,
            'FR_Knee_joint': 1.0,
            'HR_Knee_joint': 1.0,
        }

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 15
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        terrain_proportions = [1.0, 0, 0, 0]
        slope_treshold = 0.75

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 30.0}
        damping = {'joint': 1.0}
        action_scale = 0.25
        decimation = 4
        hip_reduction = 1

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_forward_curriculum = 2.2
        max_backward_curriculum = 1.5
        max_lat_curriculum = 1.0
        num_commands = 6  # [lin_vel_x, lin_vel_y, yaw_vel, jump_target_x, jump_target_y, jump_flag]
        resampling_time = 10.
        heading_command = False
        stand_still_command = False
        Rotate_command = False
        pacing_offset = False

        # jump1 单次跳状态机。保留旧 3 个速度/yaw command，额外新增 3 个 jump command。
        single_jump_mode = False
        single_jump_flag_mode = True
        observe_jump_commands = True
        single_jump_zero_velocity_commands = True
        single_jump_command_frame_range = [50, 60]
        jump_target_x_range = [1.8, 2.2]
        jump_target_y_range = [0.0, 0.0]

        # 训练辅助：jump_flag 触发后、还没腾空/落地前，早期给一次向前/向上的速度脉冲，随后衰减。
        push_towards_goal = True
        jump_assist_start_prob = 1.0
        jump_assist_end_prob = 0.0
        jump_assist_decay_steps = 1500000
        jump_assist_x_vel_range = [1.0, 1.8]
        jump_assist_z_vel_range = [1.3, 2.1]

        frequencies = 1.0
        phases = 0
        offsets = 0
        bounds = 0
        durations = 0.5

        class ranges(LeggedRobotCfg.commands.ranges):
            # jump1 起跳前速度命令保持 0，目标落点由 jump_target_x/y 给出。
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-math.pi, math.pi]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/CC1_modified/urdf/CC1_0313.urdf'
        name = "Cc1"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "TORSO", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        privileged_contacts_on = ["TORSO", "THIGH", "SHANK"]
        self_collisions = 1
        flip_visual_attachments = True

        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class termination:
        base_vel_violate_commands = False
        out_of_border = True
        fall_down = False

    class domain_rand:
        # startup
        randomize_payload_mass = True
        payload_mass_range = [-1.5, 2.0]

        randomize_com_displacement = True
        com_displacement_range = dict(
            x=[-0.07, 0.07],
            y=[-0.07, 0.07],
            z=[-0.07, 0.07],
        )

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]

        # startup and reset
        randomize_friction = True
        friction_range = [0.2, 1.25]

        randomize_restitution = False
        restitution_range = [0., 1.0]

        # reset
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        randomize_kp = True
        kp_range = [0.9, 1.1]

        randomize_kd = True
        kd_range = [0.9, 1.1]

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

        # interval
        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        push_robots = False
        push_interval_s = 16
        max_push_vel_xy = 1.

        delay = True
        recover_mode = False

    class rewards(LeggedRobotCfg.rewards):
        class scales():
            # general
            # termination = -0.0

            # velocity-tracking
            tracking_lin_vel = 1.0  # jump1 中起跳后按 jump_target / time_s 推导前向速度目标
            tracking_ang_vel = 2.0

            # root
            lin_vel_z = -0.02
            ang_vel_xy = -0.1
            orientation = -2.0
            base_height = -3.0
            # base_height_vel = -4.0

            # joint
            torques = -0.0001
            # torque_limits = -0.0
            # dof_vel = -0.0
            dof_acc = -2.5e-7
            stand_still = -0.5
            # hip_pos = -0.4
            # thigh_pose = -0.1
            # calf_pose = -0.1
            # dof_pos_limits = -0.0
            # dof_vel_limits = -0.0
            joint_power = -2e-5
            power_distribution = -10e-6
            # feet_mirror = -0.05

            # action
            action_rate = -0.02
            smoothness = -0.01
            # hip_action_magnitude = -0.00

            # contact
            # collision = -1.0
            # feet_contact_forces = -0.00015

            # others
            # feet_air_time = 0.0
            feet_air_time_variance_velocity = 0.0
            # has_contact = 10.0
            # feet_stumble = -0.0
            feet_slide = -0.1
            # feet_clearance_base = -2.0
            # feet_clearance_terrain = -0.0
            # feet_yaw_clearance_terrain = 5.0
            # stuck = -0.00
            # upward = 0.0

            # feet
            raibert_heuristic = 0.0
            # tracking_contacts_shaped_force = 2.0
            # tracking_contacts_shaped_vel = 2.0
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0
            feet_clearance_cmd_linear = 0.0
            jump = 2.0

            # jump1 状态机核心奖励。
            jump_prepare_stance = 0.35
            line_z = 7.0
            base_height_flight = 6.0
            land_pos = 5.0
            jump_landing_stable = 1.2
            jump_second_takeoff = -6.0
            jump_no_flight = -5.0

            # 少量保留四脚同步/半接触/落地不过度蹲的约束。
            # jump_air_time = 0.0
            # jump_flight_phase_air = 0.8
            jump_mixed_contact = -3.0
            jump_height = 2.0
            jump_z_vel = 1.0
            jump_takeoff_z_vel = 3.5
            jump_takeoff_x_vel = 2.2
            jump_forward_distance = 0.0
            jump_landing_knee_bend = -1.0


        reward_curriculum = False
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]

        only_positive_rewards = False
        tracking_sigma = 0.20
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target_vel = 0.6
        base_height_target = 0.42
        feet_height_target_base = -0.25
        feet_height_target_terrain = 0.15
        max_contact_force = 100.
        target_foot_height = 0.1
        target_foot_height_yaw = 0.1
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        # cycle_time = 0.333

        # jump1 状态机参数。
        jump_tracking_lin_vel_time_s = 0.70
        jump_tracking_lin_vel_max = 3.5
        line_z_min = 0.15
        line_z_target = 2.2
        base_height_flight_min = 0.43
        base_height_flight_target = 0.68
        land_pos_sigma = 0.12
        land_pos_lateral_weight = 2.0
        jump_no_flight_grace_s = 0.18
        jump_prepare_dof_sigma = 0.20
        jump_prepare_vel_sigma = 0.08
        jump_landing_stable_sigma = 0.25
        jump_landing_stable_lin_weight = 1.0
        jump_landing_stable_ang_weight = 0.5
        jump_landing_stable_tilt_weight = 2.0

        # 跳跃奖励门控。single_jump_flag_mode 下由 jump_flag / has_jumped 控制。
        # jump_contact_force_threshold = 5.0
        jump_min_command_speed = 0.05
        jump_min_yaw_speed = 0.05
        jump_sync_air_only = True

        # 跳高和起跳速度目标。保留原配置，便于后续切回或对照调参。
        jump_height_min = 0.43
        jump_height_target = 0.68
        jump_z_vel_min = 0.2
        jump_z_vel_target = 1.8
        jump_takeoff_z_vel_min = 0.15
        jump_takeoff_z_vel_target = 2.2
        jump_takeoff_x_vel_min = 0.2
        jump_takeoff_x_vel_target = 3.0

        # 原固定跳远奖励参数保留；jump1 默认使用 land_pos，不打开 jump_forward_distance。
        jump_forward_distance_min = 1.8
        jump_forward_distance_target = 2.0
        jump_forward_distance_max = 2.2
        jump_forward_distance_sigma = 0.12
        jump_forward_distance_start_s = 1.05

        jump_landing_knee_max = 1.55
        # jump_takeoff_require_contact = True
        # jump_ref_forward_vel_min = 0.5
        # jump_ref_forward_vel_target = 1.8

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            jump_distance = 1.0
            jump_flag = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")


class Cc1Jump1wtwCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100
        max_iterations = 10000

        # logging
        save_interval = 100
        experiment_name = 'jump1_wtw_cc1'
        run_name = ''
        # load and resume
        # resume = True
        # load_run = osp.join(logs_root, 'jump_cc1', 'Apr02_18-06-36_')
        resume = False
        load_run = -1
        checkpoint = -1
