#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data as pd

POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12
TAR_TOE_POS_LOCAL_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
TAR_TOE_VEL_LOCAL_SIZE = 12

ROOT_POS_START = 0
ROOT_ROT_START = ROOT_POS_START + POS_SIZE
JOINT_POS_START = ROOT_ROT_START + ROT_SIZE
TAR_TOE_POS_START = JOINT_POS_START + JOINT_POS_SIZE
LINEAR_VEL_START = TAR_TOE_POS_START + TAR_TOE_POS_LOCAL_SIZE
ANGULAR_VEL_START = LINEAR_VEL_START + LINEAR_VEL_SIZE
JOINT_VEL_START = ANGULAR_VEL_START + ANGULAR_VEL_SIZE
TAR_TOE_VEL_START = JOINT_VEL_START + JOINT_VEL_SIZE
FRAME_SIZE = TAR_TOE_VEL_START + TAR_TOE_VEL_LOCAL_SIZE

LEG_NAMES = ["FL", "FR", "HL", "HR"]
JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cc1_urdf() -> Path:
    return repo_root() / "legged_gym/resources/robots/CC1_modified/urdf/CC1_0603.urdf"


def load_motion(path: Path):
    motion = json.loads(path.read_text())
    frames = np.asarray(motion["Frames"], dtype=np.float32)
    frame_duration = float(motion.get("FrameDuration", 0.02))
    weight = float(motion.get("MotionWeight", 1.0))
    return frames, frame_duration, weight


def iter_motion_files(paths):
    files = []
    for pth in paths:
        if pth.is_dir():
            files.extend(sorted(p for p in pth.iterdir() if p.suffix in (".txt", ".json")))
        else:
            files.append(pth)
    return files


def reorder_legs(data: np.ndarray, order: str) -> np.ndarray:
    if order == "fl_fr_hl_hr":
        return data
    if order == "fl_fr_rl_rr":
        return data
    if order == "fr_fl_hr_hl":
        legs = np.split(data, 4, axis=-1)
        return np.concatenate([legs[1], legs[0], legs[3], legs[2]], axis=-1)
    raise ValueError(f"Unsupported order: {order}")


def load_joint_limits(urdf_path: Path):
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pd.getDataPath())
        robot = p.loadURDF(str(urdf_path), useFixedBase=True, flags=p.URDF_MAINTAIN_LINK_ORDER)
        limits = {}
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            name = info[1].decode("utf-8")
            jtype = info[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                limits[name] = (info[8], info[9])
        return limits
    finally:
        p.disconnect(cid)


def summarize(values):
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
    }


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def toe_world_positions(root_pos: np.ndarray, root_rot: np.ndarray, toe_local: np.ndarray) -> np.ndarray:
    toe_local = toe_local.reshape(toe_local.shape[0], 4, 3)
    return root_pos[:, None, :] + quat_rotate(root_rot[:, None, :], toe_local)


def print_stats(prefix: str, stats: dict):
    print(
        f"  {prefix}: min={stats['min']:.3f} mean={stats['mean']:.3f} "
        f"p95={stats['p95']:.3f} max={stats['max']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Quantitative checks for CC1 AMP motion files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Motion files or directories")
    parser.add_argument("--urdf", type=Path, default=default_cc1_urdf())
    parser.add_argument(
        "--order",
        choices=["fl_fr_hl_hr", "fl_fr_rl_rr", "fr_fl_hr_hl"],
        default="fl_fr_hl_hr",
        help="Leg ordering in the dataset. CC1 retarget output is fl_fr_hl_hr.",
    )
    parser.add_argument("--tol", type=float, default=1e-3, help="Joint limit tolerance")
    parser.add_argument(
        "--foot-ground-threshold",
        type=float,
        default=0.015,
        help="World z threshold used to report low-foot ratio.",
    )
    args = parser.parse_args()

    files = iter_motion_files(args.paths)
    if not files:
        raise SystemExit("No motion files found.")

    limits = load_joint_limits(args.urdf)
    missing = [name for name in JOINT_NAMES if name not in limits]
    if missing:
        print(f"[warn] Missing joint limits in URDF for: {missing}")
    joint_limits = [limits.get(name, (-np.inf, np.inf)) for name in JOINT_NAMES]

    for path in files:
        frames, frame_duration, weight = load_motion(path)
        if frames.ndim != 2:
            print(f"{path}: invalid frame shape {frames.shape}")
            continue

        if frames.shape[1] != FRAME_SIZE:
            print(f"{path}: unexpected frame width {frames.shape[1]} (expected {FRAME_SIZE})")

        finite_ratio = float(np.mean(np.isfinite(frames)))

        root_pos = frames[:, ROOT_POS_START:ROOT_ROT_START]
        root_rot = frames[:, ROOT_ROT_START:JOINT_POS_START]
        joint_pos = frames[:, JOINT_POS_START:TAR_TOE_POS_START]
        toe_pos = frames[:, TAR_TOE_POS_START:LINEAR_VEL_START]
        lin_vel = frames[:, LINEAR_VEL_START:ANGULAR_VEL_START]
        ang_vel = frames[:, ANGULAR_VEL_START:JOINT_VEL_START]
        joint_vel = frames[:, JOINT_VEL_START:TAR_TOE_VEL_START]

        joint_pos = reorder_legs(joint_pos, args.order)
        toe_pos = reorder_legs(toe_pos, args.order)
        joint_vel = reorder_legs(joint_vel, args.order)

        quat_norm = np.linalg.norm(root_rot, axis=1)
        quat_dev = np.abs(quat_norm - 1.0)

        joint_low = np.array([jl for jl, _ in joint_limits], dtype=np.float32)
        joint_high = np.array([jh for _, jh in joint_limits], dtype=np.float32)
        below = joint_pos < (joint_low - args.tol)
        above = joint_pos > (joint_high + args.tol)
        limit_viol = np.logical_or(below, above)
        limit_viol_ratio = float(np.mean(limit_viol))
        if np.any(limit_viol):
            viol_amount = np.maximum(joint_low - joint_pos, joint_pos - joint_high)
            max_viol = float(np.max(viol_amount[limit_viol]))
        else:
            max_viol = 0.0

        root_speed = np.linalg.norm(lin_vel, axis=1)
        ang_speed = np.linalg.norm(ang_vel, axis=1)
        joint_speed = np.linalg.norm(joint_vel.reshape(joint_vel.shape[0], 4, 3), axis=2).reshape(-1)

        toe_local = toe_pos.reshape(-1, 4, 3)
        toe_world = toe_world_positions(root_pos, root_rot, toe_pos)
        toe_z_local = toe_local[:, :, 2]
        toe_z_world = toe_world[:, :, 2]
        low_foot_ratio = float(np.mean(toe_z_world < args.foot_ground_threshold))

        duration = max(frames.shape[0] - 1, 0) * frame_duration

        print(f"\n{path}")
        print(f"  frames: {frames.shape[0]} | duration: {duration:.3f}s | dt: {frame_duration:.4f}s | weight: {weight}")
        print(f"  finite_ratio: {finite_ratio:.4f} | quat_norm_dev max: {float(np.max(quat_dev)):.4e}")
        print(f"  base_z: min={root_pos[:, 2].min():.3f} mean={root_pos[:, 2].mean():.3f} max={root_pos[:, 2].max():.3f}")
        print(f"  joint_limit_viol: {limit_viol_ratio * 100:.2f}% | max_violation(rad): {max_viol:.4f}")
        print(f"  low_foot_ratio(world_z<{args.foot_ground_threshold:.3f}): {low_foot_ratio * 100:.2f}%")

        print_stats("root_speed(m/s)", summarize(root_speed))
        print_stats("ang_speed(rad/s)", summarize(ang_speed))
        print_stats("joint_speed(rad/s)", summarize(joint_speed))
        print_stats("toe_z_local(m)", summarize(toe_z_local.reshape(-1)))
        print_stats("toe_z_world(m)", summarize(toe_z_world.reshape(-1)))

        for i, leg in enumerate(LEG_NAMES):
            stats = summarize(toe_z_world[:, i])
            print(
                f"  {leg}_toe_z_world: min={stats['min']:.3f} "
                f"mean={stats['mean']:.3f} p95={stats['p95']:.3f} max={stats['max']:.3f}"
            )


if __name__ == "__main__":
    main()
