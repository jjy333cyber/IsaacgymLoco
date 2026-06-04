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
TAR_TOE_POS_LOCAL_SIZE = 12

JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
FOOT_LINK_NAMES = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
FOOT_COLORS = [
    [0.95, 0.15, 0.10],
    [0.10, 0.35, 0.95],
    [0.15, 0.70, 0.25],
    [0.95, 0.65, 0.10],
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cc1_urdf() -> Path:
    return repo_root() / "legged_gym/resources/robots/CC1_modified/urdf/CC1_0603.urdf"


def load_motion(path: Path):
    motion = json.loads(path.read_text())
    frames = np.asarray(motion["Frames"], dtype=np.float32)
    frame_duration = float(motion.get("FrameDuration", 0.02))
    return frames, frame_duration


def iter_motion_files(path: Path):
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in (".txt", ".json"))
        if not files:
            raise SystemExit(f"No motion files found in directory: {path}")
        return files
    return [path]


def reorder_legs(data: np.ndarray, order: str) -> np.ndarray:
    if order == "fl_fr_hl_hr":
        return data
    if order == "fl_fr_rl_rr":
        return data
    if order == "fr_fl_hr_hl":
        legs = np.split(data, 4)
        return np.concatenate([legs[1], legs[0], legs[3], legs[2]])
    raise ValueError(f"Unsupported order: {order}")


def build_name_to_idx(robot_id: int, link: bool = False):
    name_to_idx = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        idx_name = info[12] if link else info[1]
        name_to_idx[idx_name.decode("utf-8")] = i
    return name_to_idx


def update_foot_markers(robot_id: int, foot_link_ids, marker_ids):
    new_ids = []
    for foot_id, marker_id, color in zip(foot_link_ids, marker_ids, FOOT_COLORS):
        if foot_id is None:
            new_ids.append(marker_id)
            continue
        pos = np.asarray(p.getLinkState(robot_id, foot_id, computeForwardKinematics=True)[4])
        line_from = pos - np.array([0.0, 0.0, 0.035])
        line_to = pos + np.array([0.0, 0.0, 0.035])
        kwargs = dict(
            lineFromXYZ=line_from,
            lineToXYZ=line_to,
            lineColorRGB=color,
            lineWidth=4,
        )
        if marker_id is not None:
            kwargs["replaceItemUniqueId"] = marker_id
        new_ids.append(p.addUserDebugLine(**kwargs))
    return new_ids


def main():
    parser = argparse.ArgumentParser(description="Visualize CC1 AMP motion in PyBullet.")
    parser.add_argument("motion_file", type=Path, help="Motion file or directory containing motion files")
    parser.add_argument("--urdf", type=Path, default=default_cc1_urdf())
    parser.add_argument(
        "--order",
        choices=["fl_fr_hl_hr", "fl_fr_rl_rr", "fr_fl_hr_hl"],
        default="fl_fr_hl_hr",
        help="Leg ordering in the dataset. CC1 retarget output is fl_fr_hl_hr.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index exclusive")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--fixed-base", action="store_true", help="Load robot with fixed base")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Add z offset to root position during playback")
    parser.add_argument("--draw-feet", action="store_true", help="Draw colored foot markers")
    parser.add_argument("--camera-distance", type=float, default=1.2)
    parser.add_argument("--camera-yaw", type=float, default=50.0)
    parser.add_argument("--camera-pitch", type=float, default=-25.0)
    args = parser.parse_args()

    motion_files = iter_motion_files(args.motion_file)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        str(args.urdf),
        [0, 0, 0.3],
        [0, 0, 0, 1],
        useFixedBase=args.fixed_base,
        flags=p.URDF_MAINTAIN_LINK_ORDER,
    )

    joint_name_to_idx = build_name_to_idx(robot, link=False)
    missing_joints = [name for name in JOINT_NAMES if name not in joint_name_to_idx]
    if missing_joints:
        print(f"[warn] Missing joints in URDF: {missing_joints}")

    link_name_to_idx = build_name_to_idx(robot, link=True)
    foot_link_ids = [link_name_to_idx.get(name) for name in FOOT_LINK_NAMES]
    missing_feet = [name for name, idx in zip(FOOT_LINK_NAMES, foot_link_ids) if idx is None]
    if missing_feet:
        print(f"[warn] Missing foot links in URDF: {missing_feet}")

    foot_marker_ids = [None] * 4
    base_text_id = None

    try:
        while True:
            for motion_file in motion_files:
                frames, frame_duration = load_motion(motion_file)
                start = max(0, args.start)
                end = frames.shape[0] if args.end is None else min(frames.shape[0], args.end)
                if start >= end:
                    print(
                        f"[skip] {motion_file} "
                        f"(invalid frame range: start={start}, end={end}, total={frames.shape[0]})"
                    )
                    continue

                print(f"[play] {motion_file} (frames={frames.shape[0]}, dt={frame_duration})")
                dt = frame_duration / max(args.speed, 1e-6)

                for i in range(start, end):
                    frame = frames[i]
                    root_pos = frame[0:POS_SIZE].copy()
                    root_pos[2] += args.z_offset
                    root_rot = frame[POS_SIZE:POS_SIZE + ROT_SIZE]
                    joint_pos = frame[POS_SIZE + ROT_SIZE:POS_SIZE + ROT_SIZE + JOINT_POS_SIZE]
                    joint_pos = reorder_legs(joint_pos, args.order)

                    p.resetBasePositionAndOrientation(robot, root_pos, root_rot)
                    for name, q in zip(JOINT_NAMES, joint_pos):
                        joint_id = joint_name_to_idx.get(name)
                        if joint_id is not None:
                            p.resetJointState(robot, joint_id, float(q))

                    if args.draw_feet:
                        foot_marker_ids = update_foot_markers(robot, foot_link_ids, foot_marker_ids)

                    base_pos = p.getBasePositionAndOrientation(robot)[0]
                    text_kwargs = dict(
                        text=f"base_z={base_pos[2]:.3f}  frame={i}",
                        textPosition=[base_pos[0], base_pos[1], base_pos[2] + 0.18],
                        textColorRGB=[0, 0, 0],
                        textSize=1.1,
                    )
                    if base_text_id is not None:
                        text_kwargs["replaceItemUniqueId"] = base_text_id
                    base_text_id = p.addUserDebugText(**text_kwargs)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=args.camera_distance,
                        cameraYaw=args.camera_yaw,
                        cameraPitch=args.camera_pitch,
                        cameraTargetPosition=base_pos,
                    )
                    p.stepSimulation()
                    time.sleep(dt)

            if not args.loop:
                break
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
