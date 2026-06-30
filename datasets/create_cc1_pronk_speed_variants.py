"""Create speed-scaled AMP motion variants for the CC1 pronk jump clip.

The output keeps the original FrameDuration and resamples the pose channels in
time. Velocity channels are recomputed afterwards, so AMP sees physically
consistent faster/slower motion instead of only a metadata change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
JOINT_POS = slice(7, 19)
TOE_LOCAL = slice(19, 31)
BASE_LIN_VEL = slice(31, 34)
BASE_ANG_VEL = slice(34, 37)
JOINT_VEL = slice(37, 49)
TOE_LOCAL_VEL = slice(49, 61)
FRAME_SIZE = 61

DEFAULT_SPEEDS = (0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)


def _output_motion(
    frames: np.ndarray,
    out_filename: Path,
    motion_weight: float,
    frame_duration: float,
    joint_order: str | None = None,
) -> None:
    with out_filename.open("w", encoding="utf-8") as f:
        f.write("{\n")
        f.write("\"LoopMode\": \"Wrap\",\n")
        f.write("\"FrameDuration\": " + str(frame_duration) + ",\n")
        f.write("\"EnableCycleOffsetPosition\": true,\n")
        f.write("\"EnableCycleOffsetRotation\": true,\n")
        f.write("\"MotionWeight\": " + str(motion_weight) + ",\n")
        if joint_order is not None:
            f.write("\"JointOrder\": \"" + str(joint_order) + "\",\n")
        f.write("\n")
        f.write("\"Frames\":\n")
        f.write("[")
        for i, curr_frame in enumerate(frames):
            if i != 0:
                f.write(",")
            f.write("\n  [")
            for j, curr_val in enumerate(curr_frame):
                if j != 0:
                    f.write(", ")
                f.write("%.5f" % curr_val)
            f.write("]")
        f.write("\n]\n}")


def _speed_label(speed: float) -> str:
    if abs(speed - round(speed)) < 1e-9:
        return f"{int(round(speed))}x"
    return f"{speed:g}x"


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    if quat.ndim == 1:
        return -quat if quat[3] < 0.0 else quat
    quat[quat[:, 3] < 0.0] *= -1.0
    return quat


def _quat_to_matrix_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat(q0.astype(np.float64, copy=True))
    q1 = _normalize_quat(q1.astype(np.float64, copy=True))
    dots = np.sum(q0 * q1, axis=1)

    flip = dots < 0.0
    q1[flip] *= -1.0
    dots[flip] *= -1.0
    dots = np.clip(dots, -1.0, 1.0)

    alpha = alpha[:, None]
    out = np.empty_like(q0)

    close = dots > 0.9995
    if np.any(close):
        out[close] = q0[close] + alpha[close] * (q1[close] - q0[close])

    far = ~close
    if np.any(far):
        theta_0 = np.arccos(dots[far])[:, None]
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha[far]
        scale_0 = np.sin(theta_0 - theta) / sin_theta_0
        scale_1 = np.sin(theta) / sin_theta_0
        out[far] = scale_0 * q0[far] + scale_1 * q1[far]

    return _normalize_quat(out)


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return vel


def _quaternion_local_angular_velocity(quats: np.ndarray, dt: float) -> np.ndarray:
    ang_vel = np.zeros((quats.shape[0], 3), dtype=np.float64)
    if quats.shape[0] <= 1:
        return ang_vel

    for i in range(quats.shape[0]):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, quats.shape[0] - 1)
        denom = (next_i - prev_i) * dt
        if denom <= 0.0:
            continue

        q_delta = _quat_multiply_xyzw(_quat_inverse_xyzw(quats[prev_i]), quats[next_i])
        q_delta = _normalize_quat(q_delta)

        vec = q_delta[:3]
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-12:
            continue

        angle = 2.0 * np.arctan2(vec_norm, q_delta[3])
        if angle > np.pi:
            angle -= 2.0 * np.pi
        ang_vel[i] = vec / vec_norm * angle / denom
    return ang_vel


def _load_motion(path: Path) -> tuple[dict, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        motion = json.load(f)
    frames = np.asarray(motion["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != FRAME_SIZE:
        raise ValueError(f"{path} must be a {FRAME_SIZE}-column AMP motion, got {frames.shape}")
    return motion, frames


def _interpolate_linear(values: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(dst_times, src_times, values[:, i]) for i in range(values.shape[1])])


def _interpolate_quat(quats: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    idx0 = np.searchsorted(src_times, dst_times, side="right") - 1
    idx0 = np.clip(idx0, 0, len(src_times) - 2)
    idx1 = idx0 + 1
    denom = np.maximum(src_times[idx1] - src_times[idx0], 1e-12)
    alpha = (dst_times - src_times[idx0]) / denom
    return _quat_slerp_xyzw(quats[idx0], quats[idx1], alpha)


def _resample_motion(frames: np.ndarray, frame_duration: float, speed: float) -> np.ndarray:
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed}")

    src_duration = (frames.shape[0] - 1) * frame_duration
    dst_duration = src_duration / speed
    dst_count = max(int(round(dst_duration / frame_duration)) + 1, 2)

    src_times = np.linspace(0.0, src_duration, frames.shape[0])
    dst_times = np.linspace(0.0, dst_duration, dst_count)
    sample_times = np.clip(dst_times * speed, 0.0, src_duration)

    root_pos = _interpolate_linear(frames[:, ROOT_POS], src_times, sample_times)
    root_quat = _interpolate_quat(frames[:, ROOT_QUAT], src_times, sample_times)
    joint_pos = _interpolate_linear(frames[:, JOINT_POS], src_times, sample_times)
    toe_local = _interpolate_linear(frames[:, TOE_LOCAL], src_times, sample_times)

    root_rot_mats = np.asarray([_quat_to_matrix_xyzw(q) for q in root_quat])
    lin_vel_world = _finite_difference(root_pos, frame_duration)
    base_lin_vel = np.einsum("nij,nj->ni", np.transpose(root_rot_mats, (0, 2, 1)), lin_vel_world)
    base_ang_vel = _quaternion_local_angular_velocity(root_quat, frame_duration)
    joint_vel = _finite_difference(joint_pos, frame_duration)
    toe_local_vel = _finite_difference(toe_local, frame_duration)

    out = np.zeros((dst_count, FRAME_SIZE), dtype=np.float64)
    out[:, ROOT_POS] = root_pos
    out[:, ROOT_QUAT] = root_quat
    out[:, JOINT_POS] = joint_pos
    out[:, TOE_LOCAL] = toe_local
    out[:, BASE_LIN_VEL] = base_lin_vel
    out[:, BASE_ANG_VEL] = base_ang_vel
    out[:, JOINT_VEL] = joint_vel
    out[:, TOE_LOCAL_VEL] = toe_local_vel
    return out


def create_variants(input_path: Path, output_dir: Path, speeds: list[float]) -> None:
    motion, frames = _load_motion(input_path)
    frame_duration = float(motion["FrameDuration"])
    motion_weight = float(motion.get("MotionWeight", 1.0))
    joint_order = motion.get("JointOrder")

    output_dir.mkdir(parents=True, exist_ok=True)
    for speed in speeds:
        scaled_frames = _resample_motion(frames, frame_duration, speed)
        out_path = output_dir / f"{input_path.stem}_{_speed_label(speed)}{input_path.suffix}"
        _output_motion(
            scaled_frames,
            out_path,
            motion_weight=motion_weight,
            frame_duration=frame_duration,
            joint_order=joint_order,
        )
        duration = (scaled_frames.shape[0] - 1) * frame_duration
        print(f"[speed {speed:g}x] {out_path} frames={scaled_frames.shape[0]} duration={duration:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/mocap_motions_cc1_pronk/jump.txt"),
        help="Source CC1 pronk AMP motion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the input motion directory.",
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPEEDS),
        help="Speed multipliers to generate.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.input.parent
    create_variants(args.input, output_dir, args.speeds)


if __name__ == "__main__":
    main()
