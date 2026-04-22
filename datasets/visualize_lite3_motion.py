#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd

POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12

# JOINT_NAMES = [
#     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
#     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
# ]

JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]



def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_motion(path: Path):
    motion = json.loads(path.read_text())
    frames = np.asarray(motion["Frames"], dtype=np.float32)
    frame_duration = float(motion.get("FrameDuration", 0.02))
    return frames, frame_duration


def reorder_legs(data: np.ndarray, order: str) -> np.ndarray:
    if order == "fl_fr_rl_rr":
        return data
    if order == "fr_fl_rr_rl":
        legs = np.split(data, 4)
        return np.concatenate([legs[1], legs[0], legs[3], legs[2]])
    raise ValueError(f"Unsupported order: {order}")


def build_joint_map(robot_id: int):
    name_to_idx = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        name_to_idx[name] = i
    return name_to_idx


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Lite3 motion in PyBullet."
    )
    parser.add_argument(
        "motion_file",
        type=Path,
        help="Path to a motion .txt/.json file, or a directory containing .txt files",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=(
            repo_root()
            / "legged_gym/resources/robots/Lite3/Lite3_urdf/urdf/Lite3.urdf"
        ),
    )
    parser.add_argument(
        "--order",
        choices=["fl_fr_rl_rr", "fr_fl_rr_rl"],
        default="fl_fr_rl_rr",
        help="Leg ordering in the dataset (default matches retarget script)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed multiplier"
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    motion_path: Path = args.motion_file
    if motion_path.is_dir():
        motion_files = sorted(
            [p for p in motion_path.iterdir() if p.is_file() and p.suffix == ".txt"]
        )
        if len(motion_files) == 0:
            raise SystemExit(f"No .txt motion files found in directory: {motion_path}")
    else:
        motion_files = [motion_path]

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        str(args.urdf), [0, 0, 0.3], [0, 0, 0, 1], useFixedBase=False
    )
    name_to_idx = build_joint_map(robot)

    while True:
        for motion_file in motion_files:
            frames, frame_duration = load_motion(motion_file)
            start = max(0, args.start)
            end = (
                frames.shape[0]
                if args.end is None
                else min(frames.shape[0], args.end)
            )
            if start >= end:
                print(
                    f"[skip] {motion_file} (invalid frame range: "
                    f"start={start}, end={end}, total={frames.shape[0]})"
                )
                continue

            print(f"[play] {motion_file}  (frames={frames.shape[0]}, dt={frame_duration})")
            dt = frame_duration / max(args.speed, 1e-6)

            for i in range(start, end):
                frame = frames[i]
                root_pos = frame[0:POS_SIZE]
                root_rot = frame[POS_SIZE:POS_SIZE + ROT_SIZE]
                joint_pos = frame[
                    POS_SIZE + ROT_SIZE : POS_SIZE + ROT_SIZE + JOINT_POS_SIZE
                ]
                joint_pos = reorder_legs(joint_pos, args.order)

                p.resetBasePositionAndOrientation(robot, root_pos, root_rot)
                for name, q in zip(JOINT_NAMES, joint_pos):
                    j = name_to_idx.get(name)
                    if j is not None:
                        p.resetJointState(robot, j, q)

                base_pos = p.getBasePositionAndOrientation(robot)[0]
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.2,
                    cameraYaw=50,
                    cameraPitch=-25,
                    cameraTargetPosition=base_pos,
                )

                p.stepSimulation()
                time.sleep(dt)

        if not args.loop:
            break

    p.disconnect()


if __name__ == "__main__":
    main()
