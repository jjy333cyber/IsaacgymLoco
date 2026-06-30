#!/usr/bin/env python3
"""Convert a CC1-authored BVH directly into AMP motion format.

This script is for BVH files that already describe the CC1 animation rig.
It does not solve IK and does not retarget animal keypoints.  Instead it
extracts the root motion and local leg rotations, maps them to the 12 CC1
actuated joints, computes velocities, and writes the 61-value AMP frame format.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from bvh_to_cc1_keypoints import axis_rotation, parse_bvh


POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12
TAR_TOE_POS_LOCAL_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
TAR_TOE_VEL_LOCAL_SIZE = 12
FRAME_SIZE = (
    POS_SIZE
    + ROT_SIZE
    + JOINT_POS_SIZE
    + TAR_TOE_POS_LOCAL_SIZE
    + LINEAR_VEL_SIZE
    + ANGULAR_VEL_SIZE
    + JOINT_VEL_SIZE
    + TAR_TOE_VEL_LOCAL_SIZE
)

JOINT_NAMES = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
]
FOOT_LINK_NAMES = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

# Source BVH coordinates used by the current CC1 rig:
#   source +Z = forward, source +X = left, source +Y = up.
# IsaacGym/URDF convention:
#   target +X = forward, target +Y = left, target +Z = up.
SOURCE_TO_ISAAC = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

CC1_IK_RIG_MAP = {
    "FL": {
        "hipx": "IKLegFrontL01_jnt",
        "hipy": "IKLegFrontL02_jnt",
        "knee": "IKLegFrontL03_jnt",
    },
    "FR": {
        "hipx": "IKLegFrontR01_jnt",
        "hipy": "IKLegFrontR02_jnt",
        "knee": "IKLegFrontR03_jnt",
    },
    "HL": {
        "hipx": "IKLegBackL01_jnt",
        "hipy": "IKLegBackL02_jnt",
        "knee": "IKLegBackL03_jnt",
    },
    "HR": {
        "hipx": "IKLegBackR01_jnt",
        "hipy": "IKLegBackR02_jnt",
        "knee": "IKLegBackR03_jnt",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cc1_urdf() -> Path:
    return repo_root() / "legged_gym/resources/robots/CC1_modified/urdf/CC1_0603.urdf"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "mocap_motions_cc1_walk"


def strip_namespace(name: str) -> str:
    return name.split(":")[-1]


def build_name_lookup(nodes):
    lookup = {}
    for i, node in enumerate(nodes):
        lookup[node.name] = i
        lookup[strip_namespace(node.name)] = i
    return lookup


def find_node(name_lookup, name: str) -> int:
    if name in name_lookup:
        return name_lookup[name]
    short = strip_namespace(name)
    if short in name_lookup:
        return name_lookup[short]
    raise ValueError(f"BVH is missing node {name!r}")


def channel_value(nodes, frame, node_index: int, channel_name: str, default=0.0):
    node = nodes[node_index]
    for i, channel in enumerate(node.channels):
        if channel == channel_name:
            return frame[node.channel_start + i]
    return default


def root_position_source(nodes, motion):
    root = nodes[0]
    positions = np.zeros((motion.shape[0], 3), dtype=np.float64)
    for frame_index, frame in enumerate(motion):
        values = {}
        for i, channel in enumerate(root.channels):
            if channel.endswith("position"):
                values[channel[0]] = frame[root.channel_start + i]
        positions[frame_index] = [
            values.get("X", root.offset[0]),
            values.get("Y", root.offset[1]),
            values.get("Z", root.offset[2]),
        ]
    return positions


def local_rotation_matrix(nodes, frame, node_index: int):
    node = nodes[node_index]
    rot = np.eye(3, dtype=np.float64)
    for i, channel in enumerate(node.channels):
        if channel.endswith("rotation"):
            rot = rot @ axis_rotation(channel[0], frame[node.channel_start + i])
    return rot


def matrix_to_quat_xyzw(rot):
    trace = np.trace(rot)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def quat_to_matrix_xyzw(quat):
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


def quat_inverse_xyzw(quat):
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quat_multiply_xyzw(a, b):
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


def finite_difference(values, dt):
    vel = np.zeros_like(values)
    if values.shape[0] <= 1:
        return vel
    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return vel


def quaternion_local_angular_velocity(quats, dt):
    ang_vel = np.zeros((quats.shape[0], 3), dtype=np.float64)
    if quats.shape[0] <= 1:
        return ang_vel

    for i in range(quats.shape[0]):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, quats.shape[0] - 1)
        denom = (next_i - prev_i) * dt
        if denom <= 0.0:
            continue

        q_delta = quat_multiply_xyzw(quat_inverse_xyzw(quats[prev_i]), quats[next_i])
        q_delta /= max(np.linalg.norm(q_delta), 1e-12)
        if q_delta[3] < 0.0:
            q_delta *= -1.0

        vec = q_delta[:3]
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-12:
            continue
        angle = 2.0 * np.arctan2(vec_norm, q_delta[3])
        if angle > np.pi:
            angle -= 2.0 * np.pi
        ang_vel[i] = vec / vec_norm * angle / denom
    return ang_vel


def source_vec_to_isaac(vec):
    return vec @ SOURCE_TO_ISAAC.T


def source_rot_to_isaac(rot):
    return SOURCE_TO_ISAAC @ rot @ SOURCE_TO_ISAAC.T


def build_root_pose(nodes, motion, source_scale, base_height):
    root_src = root_position_source(nodes, motion)
    root_delta = (root_src - root_src[0]) * source_scale
    root_pos = source_vec_to_isaac(root_delta)
    root_pos[:, 2] += base_height

    root_quat = np.zeros((motion.shape[0], ROT_SIZE), dtype=np.float64)
    for frame_index, frame in enumerate(motion):
        root_rot_src = local_rotation_matrix(nodes, frame, 0)
        root_quat[frame_index] = matrix_to_quat_xyzw(source_rot_to_isaac(root_rot_src))
    return root_pos, root_quat


def detect_preset(name_lookup, requested):
    if requested != "auto":
        return requested
    if "IKLegFrontL02_jnt" in name_lookup and "IKLegBackR03_jnt" in name_lookup:
        return "cc1_ik_rig"
    if all(name in name_lookup for name in JOINT_NAMES):
        return "urdf_joints"
    raise ValueError(
        "Could not auto-detect BVH preset. Expected CC1 IK rig nodes "
        "or URDF joint names. Use --preset explicitly if needed."
    )


def extract_rotation_delta(nodes, motion, node_index, axis, reference_frame):
    channel = f"{axis.upper()}rotation"
    values = np.array(
        [channel_value(nodes, frame, node_index, channel, 0.0) for frame in motion],
        dtype=np.float64,
    )
    return values - values[reference_frame]


def build_joint_positions(nodes, motion, args):
    name_lookup = build_name_lookup(nodes)
    preset = detect_preset(name_lookup, args.preset)

    joint_pos = np.zeros((motion.shape[0], JOINT_POS_SIZE), dtype=np.float64)
    defaults = {
        "hipx": args.default_hipx,
        "hipy": args.default_hipy,
        "knee": args.default_knee,
    }

    if preset == "cc1_ik_rig":
        leg_order = ["FL", "FR", "HL", "HR"]
        for leg_index, leg in enumerate(leg_order):
            spec = CC1_IK_RIG_MAP[leg]
            base = leg_index * 3

            hipx_node = find_node(name_lookup, spec["hipx"])
            hipy_node = find_node(name_lookup, spec["hipy"])
            knee_node = find_node(name_lookup, spec["knee"])

            hipx_delta = extract_rotation_delta(
                nodes, motion, hipx_node, args.hipx_axis, args.reference_frame
            )
            hipy_delta = extract_rotation_delta(
                nodes, motion, hipy_node, args.hipy_axis, args.reference_frame
            )
            knee_delta = extract_rotation_delta(
                nodes, motion, knee_node, args.knee_axis, args.reference_frame
            )

            joint_pos[:, base + 0] = defaults["hipx"] + np.deg2rad(
                args.hipx_sign * args.hipx_scale * hipx_delta
            )
            joint_pos[:, base + 1] = defaults["hipy"] + np.deg2rad(
                args.hipy_sign * args.hipy_scale * hipy_delta
            )
            joint_pos[:, base + 2] = defaults["knee"] + np.deg2rad(
                args.knee_sign * args.knee_scale * knee_delta
            )
        return joint_pos, preset

    if preset == "urdf_joints":
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            node_index = find_node(name_lookup, joint_name)
            axis = args.urdf_joint_axis
            delta = extract_rotation_delta(
                nodes, motion, node_index, axis, args.reference_frame
            )
            if joint_name.endswith("HipX_joint"):
                default = defaults["hipx"]
                sign = args.hipx_sign
                scale = args.hipx_scale
            elif joint_name.endswith("HipY_joint"):
                default = defaults["hipy"]
                sign = args.hipy_sign
                scale = args.hipy_scale
            else:
                default = defaults["knee"]
                sign = args.knee_sign
                scale = args.knee_scale
            joint_pos[:, joint_index] = default + np.deg2rad(sign * scale * delta)
        return joint_pos, preset

    raise ValueError(f"Unsupported preset: {preset}")


def compute_toe_local_with_pybullet(root_pos, root_quat, joint_pos, urdf_path):
    try:
        import pybullet as p
        import pybullet_data as pd
    except ImportError as exc:
        raise RuntimeError(
            "pybullet is required to compute CC1 toe positions. "
            "Install pybullet or pass --no-fk-toes."
        ) from exc

    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pd.getDataPath())
        robot = p.loadURDF(
            str(urdf_path),
            useFixedBase=False,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )

        joint_ids = {}
        link_ids = {}
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            joint_ids[info[1].decode("utf-8")] = i
            link_ids[info[12].decode("utf-8")] = i

        missing_joints = [name for name in JOINT_NAMES if name not in joint_ids]
        missing_feet = [name for name in FOOT_LINK_NAMES if name not in link_ids]
        if missing_joints or missing_feet:
            raise ValueError(
                f"URDF mismatch. Missing joints={missing_joints}, "
                f"missing feet={missing_feet}"
            )

        toe_local = np.zeros((root_pos.shape[0], TAR_TOE_POS_LOCAL_SIZE), dtype=np.float64)
        foot_bottom_z = np.zeros(root_pos.shape[0], dtype=np.float64)
        for frame_index in range(root_pos.shape[0]):
            p.resetBasePositionAndOrientation(
                robot,
                root_pos[frame_index].tolist(),
                root_quat[frame_index].tolist(),
            )
            for joint_index, joint_name in enumerate(JOINT_NAMES):
                p.resetJointState(robot, joint_ids[joint_name], joint_pos[frame_index, joint_index])

            root_rot = quat_to_matrix_xyzw(root_quat[frame_index])
            foot_positions = []
            foot_aabb_mins = []
            for foot_name in FOOT_LINK_NAMES:
                foot_link_id = link_ids[foot_name]
                link_state = p.getLinkState(
                    robot,
                    foot_link_id,
                    computeForwardKinematics=True,
                )
                foot_world = np.asarray(link_state[4], dtype=np.float64)
                foot_local = root_rot.T @ (foot_world - root_pos[frame_index])
                foot_positions.append(foot_local)
                foot_aabb_mins.append(p.getAABB(robot, foot_link_id)[0][2])
            toe_local[frame_index] = np.asarray(foot_positions).reshape(-1)
            foot_bottom_z[frame_index] = np.min(foot_aabb_mins)
        return toe_local, foot_bottom_z
    finally:
        p.disconnect(cid)


def write_motion(path, frames, frame_duration, motion_weight):
    path.parent.mkdir(parents=True, exist_ok=True)
    motion = {
        "LoopMode": "Wrap",
        "FrameDuration": float(frame_duration),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": float(motion_weight),
        "JointOrder": "FL_FR_HL_HR",
        "Frames": np.round(frames, 5).tolist(),
    }
    path.write_text(json.dumps(motion, indent=2))


def resolve_output_path(input_bvh: Path, output: Path):
    if output.suffix.lower() in (".txt", ".json"):
        return output
    return output / f"{input_bvh.stem}.txt"


def print_summary(out_path, frames, dt, preset, foot_bottom_z=None):
    root_pos = frames[:, 0:3]
    joint_pos = frames[:, 7:19]
    toe_local = frames[:, 19:31].reshape(frames.shape[0], 4, 3)
    root_rot = np.asarray([quat_to_matrix_xyzw(q) for q in frames[:, 3:7]])
    toe_world = root_pos[:, None, :] + np.einsum("nij,nkj->nki", root_rot, toe_local)
    print(f"[bvh_to_cc1_amp] preset={preset}")
    print(f"[bvh_to_cc1_amp] wrote {out_path}")
    print(f"[bvh_to_cc1_amp] frames={frames.shape[0]} dt={dt:.6f} cols={frames.shape[1]}")
    print(
        "[bvh_to_cc1_amp] base_z "
        f"min={root_pos[:, 2].min():.3f} "
        f"mean={root_pos[:, 2].mean():.3f} "
        f"max={root_pos[:, 2].max():.3f}"
    )
    print(
        "[bvh_to_cc1_amp] toe_world_z "
        f"min={toe_world[:, :, 2].min():.3f} "
        f"mean={toe_world[:, :, 2].mean():.3f} "
        f"max={toe_world[:, :, 2].max():.3f}"
    )
    if foot_bottom_z is not None:
        print(
            "[bvh_to_cc1_amp] foot_bottom_z "
            f"min={foot_bottom_z.min():.3f} "
            f"mean={foot_bottom_z.mean():.3f} "
            f"max={foot_bottom_z.max():.3f}"
        )
    for i, name in enumerate(JOINT_NAMES):
        values = joint_pos[:, i]
        print(
            f"  {name:15s} min={values.min(): .3f} "
            f"mean={values.mean(): .3f} max={values.max(): .3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CC1-authored BVH directly to AMP motion txt."
    )
    parser.add_argument("input_bvh", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=default_output_dir(),
        help="Output .txt/.json file or output directory.",
    )
    parser.add_argument("--urdf", type=Path, default=default_cc1_urdf())
    parser.add_argument("--source-scale", type=float, default=0.01)
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.45,
        help="Base height for frame 0. Source vertical deltas are preserved.",
    )
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--preset",
        choices=["auto", "cc1_ik_rig", "urdf_joints"],
        default="auto",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="BVH frame used as zero pose before adding CC1 default joint angles.",
    )
    parser.add_argument("--default-hipx", type=float, default=0.0)
    parser.add_argument("--default-hipy", type=float, default=-0.25)
    parser.add_argument("--default-knee", type=float, default=0.60)
    parser.add_argument("--hipx-axis", choices=["X", "Y", "Z"], default="Z")
    parser.add_argument("--hipy-axis", choices=["X", "Y", "Z"], default="X")
    parser.add_argument("--knee-axis", choices=["X", "Y", "Z"], default="X")
    parser.add_argument("--urdf-joint-axis", choices=["X", "Y", "Z"], default="X")
    parser.add_argument("--hipx-sign", type=float, default=1.0)
    parser.add_argument("--hipy-sign", type=float, default=-1.0)
    parser.add_argument("--knee-sign", type=float, default=-1.0)
    parser.add_argument("--hipx-scale", type=float, default=1.0)
    parser.add_argument("--hipy-scale", type=float, default=0.45)
    parser.add_argument("--knee-scale", type=float, default=0.45)
    parser.add_argument(
        "--no-fk-toes",
        action="store_true",
        help="Write zero toe local positions/velocities instead of using PyBullet FK.",
    )
    parser.add_argument(
        "--no-ground-align",
        action="store_true",
        help="Do not shift root z so the lowest FK toe point touches the ground.",
    )
    parser.add_argument(
        "--ground-align-mode",
        choices=["frame", "global", "none"],
        default="frame",
        help=(
            "How to align feet to the ground. frame keeps every frame's lowest "
            "toe on the ground, which is best for high-stance walk visualization."
        ),
    )
    parser.add_argument(
        "--ground-clearance",
        type=float,
        default=0.0,
        help="Lowest foot collision bottom z after ground alignment.",
    )
    args = parser.parse_args()

    nodes, motion, dt = parse_bvh(args.input_bvh)
    start = max(args.start, 0)
    end = motion.shape[0] if args.end is None else min(args.end, motion.shape[0])
    if end <= start:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")
    if not (start <= args.reference_frame < end):
        raise ValueError(
            f"--reference-frame must be inside the selected range [{start}, {end})"
        )
    motion = motion[start:end]
    args.reference_frame -= start

    root_pos, root_quat = build_root_pose(
        nodes,
        motion,
        source_scale=args.source_scale,
        base_height=args.base_height,
    )
    joint_pos, preset = build_joint_positions(nodes, motion, args)
    joint_vel = finite_difference(joint_pos, dt)

    root_rot_mats = np.asarray([quat_to_matrix_xyzw(q) for q in root_quat])
    if args.no_fk_toes:
        toe_local = np.zeros((motion.shape[0], TAR_TOE_POS_LOCAL_SIZE), dtype=np.float64)
        foot_bottom_z = None
    else:
        toe_local, foot_bottom_z = compute_toe_local_with_pybullet(
            root_pos,
            root_quat,
            joint_pos,
            args.urdf,
        )

    ground_align_mode = "none" if args.no_ground_align else args.ground_align_mode
    if ground_align_mode != "none" and not args.no_fk_toes:
        if ground_align_mode == "frame":
            z_shift = args.ground_clearance - foot_bottom_z
        elif ground_align_mode == "global":
            z_shift = np.full_like(foot_bottom_z, args.ground_clearance - np.min(foot_bottom_z))
        else:
            z_shift = np.zeros_like(foot_bottom_z)
        root_pos[:, 2] += z_shift
        foot_bottom_z = foot_bottom_z + z_shift

    lin_vel_world = finite_difference(root_pos, dt)
    base_lin_vel = np.einsum("nij,nj->ni", np.transpose(root_rot_mats, (0, 2, 1)), lin_vel_world)
    base_ang_vel = quaternion_local_angular_velocity(root_quat, dt)
    toe_vel_local = finite_difference(toe_local, dt)

    frames = np.hstack(
        [
            root_pos,
            root_quat,
            joint_pos,
            toe_local,
            base_lin_vel,
            base_ang_vel,
            joint_vel,
            toe_vel_local,
        ]
    )
    if frames.shape[1] != FRAME_SIZE:
        raise RuntimeError(f"Expected {FRAME_SIZE} columns, got {frames.shape[1]}")

    out_path = resolve_output_path(args.input_bvh, args.output)
    write_motion(out_path, frames, dt, args.motion_weight)
    print_summary(out_path, frames, dt, preset, foot_bottom_z=foot_bottom_z)


if __name__ == "__main__":
    main()
