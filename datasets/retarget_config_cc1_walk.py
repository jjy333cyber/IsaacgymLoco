"""Retarget configuration for walk.bvh converted CC1 keypoints."""

from pathlib import Path

import numpy as np
from pybullet_utils import transformations

from legged_gym import LEGGED_GYM_ROOT_DIR


MOTION_FILES_DIR = str(Path(__file__).parent)
VISUALIZE_RETARGETING = False

URDF_FILENAME = (
    f"{LEGGED_GYM_ROOT_DIR}/resources/robots/CC1_0626/urdf/CC1_0626.urdf"
)
OUTPUT_DIR = f"{MOTION_FILES_DIR}/mocap_motions_cc1_walk_raw"

FRAME_DURATION = 0.02
REF_POS_SCALE = 1.0
# walk.bvh is Y-up and moves along +Z. These rotations map +Z to CC1 +X.
REF_COORD_ROT = transformations.quaternion_from_euler(0.5 * np.pi, 0, 0)
REF_ROOT_ROT = transformations.quaternion_from_euler(0, 0, 0.5 * np.pi)

INIT_POS = np.array([0, 0, 0.42])
INIT_ROT = np.array([0, 0, 0, 1.0])
SIM_TOE_JOINT_IDS = [3, 7, 11, 15]
SIM_HIP_JOINT_IDS = [1, 5, 9, 13]
SIM_ROOT_OFFSET = np.array([0, 0, 0.02])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.02, 0.0]),
    np.array([0, 0.02, 0.0]),
    np.array([0, -0.02, 0.0]),
    np.array([0, 0.02, 0.0]),
]
TOE_HEIGHT_OFFSET = 0.02

# DEFAULT_JOINT_POSE = np.array([
#     0.0, -0.4, 0.8,
#     0.0, -0.4, 0.8,
#     0.0, -0.4, 0.8,
#     0.0, -0.4, 0.8,
# ])
DEFAULT_JOINT_POSE = np.array([
    0.0, -0.5, 1.0,
    0.0, -0.5, 1.0,
    0.0, -0.5, 1.0,
    0.0, -0.5, 1.0,
])
JOINT_DAMPING = [
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
]
FORWARD_DIR_OFFSET = np.array([0, 0, 0])

FR_FOOT_NAME = "FR_FOOT"
FL_FOOT_NAME = "FL_FOOT"
HR_FOOT_NAME = "HR_FOOT"
HL_FOOT_NAME = "HL_FOOT"

OUTPUT_JOINT_ORDER = "FL_FR_HL_HR"
MOCAP_MOTIONS = [[
    "walk",
    f"{MOTION_FILES_DIR}/keypoint_datasets/cc1_walk/walk_joint_pos.txt",
    0,
    None,
    1.0,
]]
