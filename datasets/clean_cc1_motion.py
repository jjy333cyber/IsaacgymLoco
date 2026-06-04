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
    data = json.loads(path.read_text())
    frames = np.asarray(data["Frames"], dtype=np.float32)
    return data, frames


def save_motion(dst_path: Path, template: dict, frames: np.ndarray):
    out = dict(template)
    out["Frames"] = frames.tolist()
    dst_path.write_text(json.dumps(out, indent=2))


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


def split_segments(good_mask: np.ndarray):
    segments = []
    start = None
    for i, good in enumerate(good_mask):
        if good and start is None:
            start = i
        elif not good and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(good_mask)))
    return segments


def build_bad_mask(frames, args, joint_limits):
    if frames.ndim != 2 or frames.shape[1] < FRAME_SIZE:
        return np.ones((frames.shape[0],), dtype=bool), {"bad_shape": np.ones((frames.shape[0],), dtype=bool)}

    masks = {}
    masks["non_finite"] = ~np.isfinite(frames).all(axis=1)

    root_pos = frames[:, ROOT_POS_START:ROOT_ROT_START]
    root_rot = frames[:, ROOT_ROT_START:JOINT_POS_START]
    joint_pos = frames[:, JOINT_POS_START:TAR_TOE_POS_START]
    lin = frames[:, LINEAR_VEL_START:ANGULAR_VEL_START]
    ang = frames[:, ANGULAR_VEL_START:JOINT_VEL_START]
    joint_vel = frames[:, JOINT_VEL_START:TAR_TOE_VEL_START]

    joint_pos = reorder_legs(joint_pos, args.order)
    joint_vel = reorder_legs(joint_vel, args.order)

    masks["lin_vel"] = np.max(np.abs(lin), axis=1) > args.max_lin
    masks["ang_vel"] = np.max(np.abs(ang), axis=1) > args.max_ang
    masks["joint_vel"] = np.max(np.abs(joint_vel), axis=1) > args.max_joint_vel

    quat_norm = np.linalg.norm(root_rot, axis=1)
    masks["quat_norm"] = np.abs(quat_norm - 1.0) > args.quat_tol

    if args.min_base_z is not None:
        masks["min_base_z"] = root_pos[:, 2] < args.min_base_z
    if args.max_base_z is not None:
        masks["max_base_z"] = root_pos[:, 2] > args.max_base_z

    if joint_limits is not None:
        joint_low = np.array([jl for jl, _ in joint_limits], dtype=np.float32)
        joint_high = np.array([jh for _, jh in joint_limits], dtype=np.float32)
        below = joint_pos < (joint_low - args.joint_limit_tol)
        above = joint_pos > (joint_high + args.joint_limit_tol)
        masks["joint_limits"] = np.logical_or(below, above).any(axis=1)

    bad = np.zeros((frames.shape[0],), dtype=bool)
    for mask in masks.values():
        bad |= mask
    return bad, masks


def main():
    parser = argparse.ArgumentParser(description="Clean CC1 AMP motion dataset.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--urdf", type=Path, default=default_cc1_urdf())
    parser.add_argument(
        "--order",
        choices=["fl_fr_hl_hr", "fl_fr_rl_rr", "fr_fl_hr_hl"],
        default="fl_fr_hl_hr",
        help="Leg ordering in the dataset. CC1 retarget output is fl_fr_hl_hr.",
    )
    parser.add_argument("--max-lin", type=float, default=2.0, help="Max abs base linear velocity (m/s)")
    parser.add_argument("--max-ang", type=float, default=3.0, help="Max abs base angular velocity (rad/s)")
    parser.add_argument("--max-joint-vel", type=float, default=15.0, help="Max abs joint velocity (rad/s)")
    parser.add_argument("--quat-tol", type=float, default=1e-3, help="Quaternion norm tolerance")
    parser.add_argument("--joint-limit-tol", type=float, default=1e-3, help="Joint limit tolerance")
    parser.add_argument("--ignore-joint-limits", action="store_true", help="Disable URDF joint limit checks")
    parser.add_argument("--min-base-z", type=float, default=0.12, help="Drop frames below this base z; use -1 to disable")
    parser.add_argument("--max-base-z", type=float, default=0.80, help="Drop frames above this base z; use -1 to disable")
    parser.add_argument("--min-frames", type=int, default=60, help="Minimum frames per kept segment")
    parser.add_argument("--drop-bad-ratio", type=float, default=0.2, help="Drop file if bad ratio exceeds this")
    args = parser.parse_args()

    if args.min_base_z is not None and args.min_base_z < 0:
        args.min_base_z = None
    if args.max_base_z is not None and args.max_base_z < 0:
        args.max_base_z = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix in (".txt", ".json"))
    if not files:
        raise SystemExit(f"No motion files found in {args.input_dir}")

    joint_limits = None
    if not args.ignore_joint_limits:
        limits = load_joint_limits(args.urdf)
        missing = [name for name in JOINT_NAMES if name not in limits]
        if missing:
            print(f"[warn] Missing joint limits in URDF for: {missing}")
        joint_limits = [limits.get(name, (-np.inf, np.inf)) for name in JOINT_NAMES]

    for path in files:
        try:
            template, frames = load_motion(path)
        except Exception as exc:
            print(f"{path.name}: skipped (parse_error={exc})")
            continue

        bad_mask, reason_masks = build_bad_mask(frames, args, joint_limits)
        good_mask = ~bad_mask
        bad_ratio = 1.0 - float(np.mean(good_mask))
        reason_text = ", ".join(
            f"{name}={float(np.mean(mask)) * 100:.1f}%"
            for name, mask in reason_masks.items()
            if np.any(mask)
        )
        reason_text = reason_text if reason_text else "no_bad_frames"

        if bad_ratio > args.drop_bad_ratio:
            print(f"{path.name}: dropped (bad_ratio={bad_ratio:.3f}; {reason_text})")
            continue

        segments = split_segments(good_mask)
        segments = [seg for seg in segments if (seg[1] - seg[0]) >= args.min_frames]
        if not segments:
            print(f"{path.name}: no valid segments (bad_ratio={bad_ratio:.3f}; {reason_text})")
            continue

        stem = path.stem
        if len(segments) == 1:
            out_path = args.output_dir / f"{stem}.txt"
            s, e = segments[0]
            save_motion(out_path, template, frames[s:e])
            print(f"{path.name}: kept {e - s} frames -> {out_path.name} ({reason_text})")
        else:
            for i, (s, e) in enumerate(segments):
                out_path = args.output_dir / f"{stem}_c{i + 1}.txt"
                save_motion(out_path, template, frames[s:e])
                print(f"{path.name}: kept {e - s} frames -> {out_path.name} ({reason_text})")


if __name__ == "__main__":
    main()
