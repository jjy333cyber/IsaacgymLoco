import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from pathlib import Path

# 中文说明：本文件是“关键点动作（keypoint .txt）→ 机器人关节动作（mocap .txt）”的重定向/映射配置。
# 主要用于 `datasets/retarget_kp_motions.py`，控制：
# - 使用哪个机器人 URDF
# - 参考关键点坐标到仿真坐标的尺度/偏移/初始姿态
# - 末端（脚）/髋关节在 URDF 中的 link id
# - IK 的默认关节姿态、阻尼、以及脚端高度偏移
# - 要处理哪些 mocap 片段（输入文件、帧区间、权重）

MOTION_FILES_DIR = str(Path(__file__).parent)
# 是否打开 PyBullet 可视化窗口展示重定向过程（True: GUI; False: DIRECT）
VISUALIZE_RETARGETING = True

# 目标机器人 URDF 路径（用于 IK、关节 id、link 名称等）
# URDF_FILENAME = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3_gym/urdf/Lite3.urdf"
URDF_FILENAME = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/Lite3/Lite3_urdf/urdf/Lite3.urdf"

# 重定向输出动作保存目录（生成的 mocap 动作 .txt 会写入该目录）
OUTPUT_DIR = f"{MOTION_FILES_DIR}/mocap_motions_lite3"

# 参考关键点坐标整体缩放系数（把数据集的单位/尺度映射到仿真尺度）
REF_POS_SCALE = 0.825
# 仿真中机器人初始基座位置（PyBullet loadURDF 的 basePosition）
INIT_POS = np.array([0, 0, 0.35])
# 仿真中机器人初始基座朝向四元数（x,y,z,w）
INIT_ROT = np.array([0, 0, 0, 1.0])

# 机器人四个脚端在 URDF 中对应的 link id（用于 IK 末端目标：calculateInverseKinematics2）
# 顺序需与下方 SIM_HIP_JOINT_IDS / SIM_TOE_OFFSET_LOCAL / FOOT_NAME 一致：
# [前左 FL, 前右 FR, 后左 RL(HL), 后右 RR(HR)]
# SIM_TOE_JOINT_IDS = [6, 11, 16, 21]
SIM_TOE_JOINT_IDS = [3, 7, 11, 15]
# 机器人四个髋关节（或髋相关 link）的 link id（用于获取髋部世界坐标作为脚端目标的参考点）
# SIM_HIP_JOINT_IDS = [2, 7, 12, 17]
SIM_HIP_JOINT_IDS = [1, 5, 9, 13]
# 根（基座）位置的额外偏移（在 retarget_root_pose 结果上叠加，使模型在地面附近更合理）
SIM_ROOT_OFFSET = np.array([0, 0, -0.00])
# 脚端目标的局部偏移（在 heading 坐标系下旋转到世界系后叠加到脚端目标）
# 用于微调脚相对髋的位置，补偿 URDF 脚端参考点与期望接触点的不一致。
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.08, 0.0]),
    np.array([0, 0.08, 0.0]),
    np.array([0, -0.065, 0.0]),
    np.array([0, 0.065, 0.0])
]
# 抬脚/脚端高度偏移（会在处理参考 toe 关键点的 z 轴时加上，避免脚穿地）
TOE_HEIGHT_OFFSET = 0.02

# IK 的默认关节姿态（rest pose），用于引导 IK 解到“合理的膝盖弯曲方向/姿态”
# 这里 12 维对应 4 条腿 × 每条腿 3 个关节。
DEFAULT_JOINT_POSE = np.array([0, -0.8, 1.6, 
                               0, -0.8, 1.6, 
                               0, -0.8, 1.6, 
                               0, -0.8, 1.6])
# IK 关节阻尼（jointDamping），与关节数量一致（12 维）；
# 阻尼用于稳定 IK 求解，抑制不必要的大幅摆动。
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

# 前向方向补偿（用于从 pelvis→neck 的方向估计 forward_dir 后再加一个偏移）
# 一般保持为 0；若参考数据 forward 方向有系统性偏差可在此修正。
FORWARD_DIR_OFFSET = np.array([0, 0, 0])

# URDF 中四个脚端 link 的名称（用于从 URDF 构建正向运动学链）
FR_FOOT_NAME = "FR_FOOT"
FL_FOOT_NAME = "FL_FOOT"
HR_FOOT_NAME = "HR_FOOT"
HL_FOOT_NAME = "HL_FOOT"

# 要处理的动作片段列表。
# 每个元素格式为：
# [
#   输出动作名(output_name),
#   输入关键点文件路径(input_joint_pos_txt),
#   起始帧(frame_start),
#   结束帧(frame_end, 含义见 retarget_kp_motions.py：会传入 end+1 做切片),
#   动作权重(weight，用于后续训练/采样时的权重；不影响 IK 计算)
# ]
MOCAP_MOTIONS = [
    # Output motion name, input file, frame start, frame end, motion weight.
    [
        "pace0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 162, 201, 1
    ],
    [
        "pace1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 201, 400, 1
    ],
    [
        "pace2",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk00_joint_pos.txt", 400, 600, 1
    ],
    [
        "trot0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 448, 481, 1
    ],
    [
        "trot1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk03_joint_pos.txt", 400, 600, 1
    ],
    [
        "trot2",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run04_joint_pos.txt", 480, 663, 1
    ],
    [
        "canter0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 430, 480, 1
    ],
    [
        "canter1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 380, 430, 1
    ],
    [
        "canter2",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_run00_joint_pos.txt", 480, 566, 1
    ],
    [
        "right_turn0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 1085, 1124, 1.5
    ],
    [
        "right_turn1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 560, 670, 1.5
    ],
    [
        "left_turn0",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 2404, 2450, 1.5
    ],
    [
        "left_turn1",
        f"{MOTION_FILES_DIR}/keypoint_datasets/ai4animation/dog_walk09_joint_pos.txt", 120, 220, 1.5
    ]
]