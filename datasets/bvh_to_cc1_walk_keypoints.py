#!/usr/bin/env python3
"""Convert a quadruped walk BVH into cyclic CC1 retarget keypoints."""

import argparse
from pathlib import Path

import numpy as np

from bvh_to_cc1_keypoints import (
    build_keypoints,
    forward_kinematics,
    parse_bvh,
    resample_positions,
)


LEG_TO_TOE_ID = {
    "FL": 10,
    "FR": 15,
    "HL": 19,
    "HR": 23,
}


def build_short_name_index(nodes):
    """Strip Maya namespaces while keeping the original FK node indices."""
    node_names = {}
    for index, node in enumerate(nodes):
        short_name = node.name.rsplit("|", 1)[-1].rsplit(":", 1)[-1]
        if short_name in node_names:
            raise ValueError(
                f"BVH contains duplicate joint name after namespace removal: "
                f"{short_name}"
            )
        node_names[short_name] = index
    return node_names


def find_touchdowns(contact):
    return [
        i for i in range(1, len(contact))
        if contact[i] and not contact[i - 1]
    ]


def find_active_start(root_center, frame_time, min_speed):
    horizontal_delta = np.diff(root_center[:, [0, 2]], axis=0)
    speed = np.linalg.norm(horizontal_delta, axis=1) / frame_time
    active = np.flatnonzero(speed > min_speed)
    return int(active[0] + 1) if len(active) else 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert a lateral-sequence quadruped walk BVH to CC1 keypoints."
    )
    parser.add_argument("input_bvh", type=Path)
    parser.add_argument("output_txt", type=Path)
    parser.add_argument("--source-scale", type=float, default=0.01)
    parser.add_argument("--target-dt", type=float, default=0.02)
    parser.add_argument(
        "--translation-mode",
        choices=["channels", "offset_plus_channels"],
        default="channels",
    )
    parser.add_argument(
        "--reference-leg",
        choices=list(LEG_TO_TOE_ID),
        default="HL",
        help="Clip from one touchdown of this leg to a later matching touchdown.",
    )
    parser.add_argument("--min-cycles", type=int, default=3)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--contact-height", type=float, default=0.015)
    parser.add_argument("--active-speed", type=float, default=0.02)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    args = parser.parse_args()

    nodes, motion, source_dt = parse_bvh(args.input_bvh)
    node_positions = forward_kinematics(
        nodes, motion, args.translation_mode
    )
    node_positions *= args.source_scale
    node_names = build_short_name_index(nodes)
    keypoints = build_keypoints(node_positions, node_names)

    toe_ids = list(LEG_TO_TOE_ID.values())
    ground_height = float(
        np.median(np.percentile(keypoints[:, toe_ids, 1], 2, axis=0))
    )
    keypoints[:, :, 1] -= ground_height

    root_center = 0.5 * (keypoints[:, 0] + keypoints[:, 3])
    keypoints[:, :, 0] -= root_center[0, 0]
    keypoints[:, :, 2] -= root_center[0, 2]
    root_center = 0.5 * (keypoints[:, 0] + keypoints[:, 3])

    contacts = {
        leg: keypoints[:, toe_id, 1] <= args.contact_height
        for leg, toe_id in LEG_TO_TOE_ID.items()
    }
    touchdowns = {
        leg: find_touchdowns(contact)
        for leg, contact in contacts.items()
    }

    active_start = find_active_start(
        root_center, source_dt, args.active_speed
    )
    stable_start = active_start + int(round(args.settle_time / source_dt))

    if args.start_frame is not None or args.end_frame is not None:
        source_start = args.start_frame or 0
        source_end = (
            args.end_frame
            if args.end_frame is not None
            else keypoints.shape[0] - 1
        )
    else:
        stable_touchdowns = [
            frame for frame in touchdowns[args.reference_leg]
            if frame >= stable_start
        ]
        required = args.min_cycles + 1
        if len(stable_touchdowns) < required:
            raise ValueError(
                f"{args.reference_leg} has only {len(stable_touchdowns)} "
                f"stable touchdowns; need {required}. "
                "Use --start-frame/--end-frame to select a clip manually."
            )
        source_start = stable_touchdowns[0]
        source_end = stable_touchdowns[args.min_cycles]

    if not 0 <= source_start < source_end < keypoints.shape[0]:
        raise ValueError(
            f"Invalid source range {source_start}..{source_end} "
            f"for {keypoints.shape[0]} frames"
        )

    source_period_frames = (
        source_end - source_start
    ) / args.min_cycles
    source_period = source_period_frames * source_dt
    source_frequency = 1.0 / source_period

    phase_by_leg = {}
    duty_by_leg = {}
    for leg, contact in contacts.items():
        next_touchdowns = [
            frame for frame in touchdowns[leg]
            if frame >= source_start
        ]
        if next_touchdowns:
            phase_by_leg[leg] = (
                (next_touchdowns[0] - source_start) / source_period_frames
            ) % 1.0
        duty_by_leg[leg] = float(
            np.mean(contact[source_start:source_end])
        )

    selected = keypoints[source_start:source_end + 1]
    source_duration = (selected.shape[0] - 1) * source_dt
    output = resample_positions(
        selected, source_duration, args.target_dt
    )

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_txt,
        output.reshape(output.shape[0], -1),
        delimiter=",",
        fmt="%.7f",
    )

    forward_distance = (
        root_center[source_end, 2] - root_center[source_start, 2]
    )
    mean_speed = forward_distance / source_duration
    mean_duration = float(np.mean(list(duty_by_leg.values())))

    print(f"Wrote {args.output_txt}")
    print(
        f"  source: {motion.shape[0]} frames at {1.0 / source_dt:.2f} Hz"
    )
    print(
        f"  clip: {source_start}..{source_end} "
        f"({args.min_cycles} cycles, {source_duration:.3f}s)"
    )
    print(
        f"  output: {output.shape[0]} frames at "
        f"{1.0 / args.target_dt:.2f} Hz"
    )
    print(
        f"  forward distance: {forward_distance:.3f}m | "
        f"mean speed: {mean_speed:.3f}m/s"
    )
    print(
        f"  gait frequency: {source_frequency:.3f}Hz | "
        f"mean duty: {mean_duration:.3f}"
    )
    print(
        "  touchdown phases: "
        + ", ".join(
            f"{leg}={phase_by_leg.get(leg, float('nan')):.3f}"
            for leg in ("HL", "FL", "HR", "FR")
        )
    )
    print(
        "  WTW recommendation: "
        f"frequencies={source_frequency:.3f}, "
        f"phases={phase_by_leg.get('HR', 0.5):.3f}, "
        f"offsets={phase_by_leg.get('FR', 0.8):.3f}, "
        f"bounds=0.0, durations={mean_duration:.3f}"
    )


if __name__ == "__main__":
    main()
