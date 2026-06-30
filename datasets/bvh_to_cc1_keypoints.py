#!/usr/bin/env python3
"""Convert the quadruped jump BVH into keypoints used by CC1 retargeting."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BvhNode:
    name: str
    parent: int
    offset: np.ndarray
    channels: list
    channel_start: int


class TokenStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def pop(self, expected=None):
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of BVH hierarchy")
        value = self.tokens[self.index]
        self.index += 1
        if expected is not None and value != expected:
            raise ValueError(f"Expected {expected!r}, got {value!r}")
        return value

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None


def parse_bvh(path: Path):
    lines = path.read_text().splitlines()
    motion_line = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    tokens = re.findall(r"\{|\}|[^\s{}]+", "\n".join(lines[:motion_line]))
    stream = TokenStream(tokens)
    stream.pop("HIERARCHY")

    nodes = []
    total_channels = 0

    def skip_end_site():
        stream.pop("End")
        stream.pop("Site")
        stream.pop("{")
        stream.pop("OFFSET")
        for _ in range(3):
            float(stream.pop())
        stream.pop("}")

    def parse_node(parent):
        nonlocal total_channels
        kind = stream.pop()
        if kind not in ("ROOT", "JOINT"):
            raise ValueError(f"Expected ROOT or JOINT, got {kind!r}")
        name = stream.pop()
        stream.pop("{")
        stream.pop("OFFSET")
        offset = np.array([float(stream.pop()) for _ in range(3)], dtype=np.float64)

        channels = []
        if stream.peek() == "CHANNELS":
            stream.pop("CHANNELS")
            count = int(stream.pop())
            channels = [stream.pop() for _ in range(count)]

        node_index = len(nodes)
        nodes.append(BvhNode(name, parent, offset, channels, total_channels))
        total_channels += len(channels)

        while stream.peek() != "}":
            if stream.peek() == "End":
                skip_end_site()
            else:
                parse_node(node_index)
        stream.pop("}")

    parse_node(-1)

    frames_line = lines[motion_line + 1].strip()
    frame_time_line = lines[motion_line + 2].strip()
    frame_count = int(frames_line.split(":", 1)[1])
    frame_time = float(frame_time_line.split(":", 1)[1])
    motion_rows = [
        np.fromstring(line, sep=" ", dtype=np.float64)
        for line in lines[motion_line + 3:]
        if line.strip()
    ]
    motion = np.vstack(motion_rows)

    if motion.shape != (frame_count, total_channels):
        raise ValueError(
            f"BVH motion shape is {motion.shape}, expected "
            f"({frame_count}, {total_channels})"
        )
    return nodes, motion, frame_time


def axis_rotation(axis, angle_deg):
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported rotation axis: {axis}")


def forward_kinematics(nodes, motion, translation_mode):
    positions = np.zeros((motion.shape[0], len(nodes), 3), dtype=np.float64)

    for frame_index, frame in enumerate(motion):
        world_rotations = [None] * len(nodes)
        for node_index, node in enumerate(nodes):
            local_pos = node.offset.copy()
            local_rot = np.eye(3, dtype=np.float64)
            position_values = {}

            for channel_offset, channel in enumerate(node.channels):
                value = frame[node.channel_start + channel_offset]
                if channel.endswith("position"):
                    position_values[channel[0]] = value
                elif channel.endswith("rotation"):
                    local_rot = local_rot @ axis_rotation(channel[0], value)

            for axis_index, axis in enumerate("XYZ"):
                channel_value = position_values.get(axis)
                if channel_value is None:
                    continue
                if translation_mode == "channels":
                    local_pos[axis_index] = channel_value
                else:
                    local_pos[axis_index] += channel_value

            if node.parent < 0:
                positions[frame_index, node_index] = local_pos
                world_rotations[node_index] = local_rot
            else:
                parent_rot = world_rotations[node.parent]
                parent_pos = positions[frame_index, node.parent]
                positions[frame_index, node_index] = parent_pos + parent_rot @ local_pos
                world_rotations[node_index] = parent_rot @ local_rot

    return positions


def build_keypoints(node_positions, node_names):
    required = {
        "fl_hip": "IKLegFrontL02_jnt",
        "fl_knee": "IKLegFrontL03_jnt",
        "fl_toe": "IKLegFrontL04_jnt",
        "fr_hip": "IKLegFrontR02_jnt",
        "fr_knee": "IKLegFrontR03_jnt",
        "fr_toe": "IKLegFrontR04_jnt",
        "hl_hip": "IKLegBackL02_jnt",
        "hl_knee": "IKLegBackL03_jnt",
        "hl_toe": "IKLegBackL04_jnt",
        "hr_hip": "IKLegBackR02_jnt",
        "hr_knee": "IKLegBackR03_jnt",
        "hr_toe": "IKLegBackR04_jnt",
    }
    missing = [name for name in required.values() if name not in node_names]
    if missing:
        raise ValueError(f"BVH is missing required joints: {missing}")

    p = {key: node_positions[:, node_names[name], :] for key, name in required.items()}
    keypoints = np.zeros((node_positions.shape[0], 24, 3), dtype=np.float64)

    def fill_chain(indices, hip, knee, toe):
        keypoints[:, indices[0]] = hip
        if len(indices) == 5:
            keypoints[:, indices[1]] = 0.5 * (hip + knee)
            keypoints[:, indices[2]] = knee
            keypoints[:, indices[3]] = 0.5 * (knee + toe)
        else:
            keypoints[:, indices[1]] = knee
            keypoints[:, indices[2]] = 0.5 * (knee + toe)
        keypoints[:, indices[-1]] = toe

    fill_chain([6, 7, 8, 9, 10], p["fl_hip"], p["fl_knee"], p["fl_toe"])
    fill_chain([11, 12, 13, 14, 15], p["fr_hip"], p["fr_knee"], p["fr_toe"])
    fill_chain([16, 17, 18, 19], p["hl_hip"], p["hl_knee"], p["hl_toe"])
    fill_chain([20, 21, 22, 23], p["hr_hip"], p["hr_knee"], p["hr_toe"])

    keypoints[:, 0] = 0.5 * (p["hl_hip"] + p["hr_hip"])
    keypoints[:, 3] = 0.5 * (p["fl_hip"] + p["fr_hip"])
    keypoints[:, 1] = keypoints[:, 0] + (keypoints[:, 3] - keypoints[:, 0]) / 3.0
    keypoints[:, 2] = keypoints[:, 0] + 2.0 * (keypoints[:, 3] - keypoints[:, 0]) / 3.0
    keypoints[:, 4] = keypoints[:, 3]
    keypoints[:, 5] = keypoints[:, 3]
    return keypoints


def symmetrize_keypoints(keypoints):
    result = keypoints.copy()
    pairs = [
        (6, 11), (7, 12), (8, 13), (9, 14), (10, 15),
        (16, 20), (17, 21), (18, 22), (19, 23),
    ]
    for left, right in pairs:
        midpoint = 0.5 * (result[:, left] + result[:, right])
        half_width = 0.5 * np.abs(result[:, left, 0] - result[:, right, 0])
        result[:, left, 0] = midpoint[:, 0] + half_width
        result[:, right, 0] = midpoint[:, 0] - half_width
        result[:, left, 1:] = midpoint[:, 1:]
        result[:, right, 1:] = midpoint[:, 1:]

    result[:, 0] = 0.5 * (result[:, 16] + result[:, 20])
    result[:, 3] = 0.5 * (result[:, 6] + result[:, 11])
    result[:, 1] = result[:, 0] + (result[:, 3] - result[:, 0]) / 3.0
    result[:, 2] = result[:, 0] + 2.0 * (result[:, 3] - result[:, 0]) / 3.0
    result[:, 4] = result[:, 3]
    result[:, 5] = result[:, 3]
    return result


def find_cycle_peaks(signal, frame_time):
    candidates = [
        i for i in range(1, len(signal) - 1)
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]
    ]
    min_distance = max(int(round(0.5 / frame_time)), 1)
    peaks = []
    for candidate in candidates:
        if signal[candidate] < np.percentile(signal, 60):
            continue
        if not peaks or candidate - peaks[-1] >= min_distance:
            peaks.append(candidate)
        elif signal[candidate] > signal[peaks[-1]]:
            peaks[-1] = candidate
    return peaks


def resample_positions(values, source_duration, target_dt):
    interval_count = max(int(round(source_duration / target_dt)), 1)
    source_times = np.linspace(0.0, source_duration, values.shape[0])
    sample_times = np.linspace(0.0, source_duration, interval_count + 1)
    result = np.empty((len(sample_times), values.shape[1], values.shape[2]), dtype=np.float64)
    for joint in range(values.shape[1]):
        for axis in range(3):
            result[:, joint, axis] = np.interp(
                sample_times, source_times, values[:, joint, axis]
            )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the provided quadruped jump BVH to CC1 retarget keypoints. "
            "Asymmetric motion is preserved by default."
        )
    )
    parser.add_argument("input_bvh", type=Path)
    parser.add_argument("output_txt", type=Path)
    parser.add_argument("--source-scale", type=float, default=0.01, help="BVH units to meters")
    parser.add_argument("--target-dt", type=float, default=0.02)
    parser.add_argument(
        "--translation-mode",
        choices=["channels", "offset_plus_channels"],
        default="channels",
        help="This BVH stores each local OFFSET again in its position channels.",
    )
    parser.add_argument("--no-trim-cycles", action="store_true")
    parser.add_argument(
        "--symmetrize",
        dest="symmetrize",
        action="store_true",
        help="Force left/right legs to be mirrored. Disabled by default to preserve jump.bvh style.",
    )
    parser.add_argument(
        "--no-symmetrize",
        dest="symmetrize",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(symmetrize=False)
    args = parser.parse_args()

    nodes, motion, source_dt = parse_bvh(args.input_bvh)
    node_positions = forward_kinematics(nodes, motion, args.translation_mode)
    node_positions *= args.source_scale
    node_names = {node.name: i for i, node in enumerate(nodes)}
    keypoints = build_keypoints(node_positions, node_names)

    if args.symmetrize:
        keypoints = symmetrize_keypoints(keypoints)

    toe_ids = [10, 15, 19, 23]
    ground_height = float(np.median(np.percentile(keypoints[:, toe_ids, 1], 2, axis=0)))
    keypoints[:, :, 1] -= ground_height
    root_center = 0.5 * (keypoints[:, 0] + keypoints[:, 3])
    keypoints[:, :, 0] -= root_center[0, 0]
    keypoints[:, :, 2] -= root_center[0, 2]

    source_start = 0
    source_end = keypoints.shape[0] - 1
    if not args.no_trim_cycles:
        root_height = root_center[:, 1] - ground_height
        peaks = find_cycle_peaks(root_height, source_dt)
        if len(peaks) >= 2:
            source_start, source_end = peaks[0], peaks[-1]
            keypoints = keypoints[source_start:source_end + 1]
        else:
            print("[warn] Could not find two complete height-cycle peaks; keeping all frames.")

    source_duration = (keypoints.shape[0] - 1) * source_dt
    keypoints = resample_positions(keypoints, source_duration, args.target_dt)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_txt,
        keypoints.reshape(keypoints.shape[0], -1),
        delimiter=",",
        fmt="%.7f",
    )

    root_height = 0.5 * (keypoints[:, 0, 1] + keypoints[:, 3, 1])
    forward_distance = keypoints[-1, 3, 2] - keypoints[0, 3, 2]
    print(f"Wrote {args.output_txt}")
    print(
        f"  source frames: {motion.shape[0]} at {1.0 / source_dt:.2f} Hz | "
        f"kept source frames: {source_start}..{source_end}"
    )
    print(
        f"  output frames: {keypoints.shape[0]} at {1.0 / args.target_dt:.2f} Hz | "
        f"duration: {(keypoints.shape[0] - 1) * args.target_dt:.3f}s"
    )
    print(
        f"  root height: {root_height.min():.3f}..{root_height.max():.3f} m | "
        f"forward distance: {forward_distance:.3f} m | "
        f"symmetrized: {args.symmetrize}"
    )


if __name__ == "__main__":
    main()
