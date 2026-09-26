#!/usr/bin/env python3
"""Recompute one gripper trajectory and atomically overwrite its NPZ in place.

Unlike ``adjust_0723_gripper_trajectory.py``, this command does not copy the
retarget output tree. Only the explicitly selected ``episode_XXXXXX.npz`` is
rewritten; every other episode NPZ remains untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import adjust_0723_gripper_trajectory as batch_adjuster

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_safety import refuse_protected_dataset_write


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--episode-idx", type=int, required=True)
    parser.add_argument(
        "--pose-field",
        choices=("target_eef_wxyz", "achieved_eef_wxyz", "human_eef_fixed_spine3_xyzw"),
        default="target_eef_wxyz",
    )
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--minimum-prominence-m", type=float, default=0.02)
    parser.add_argument("--minimum-distance-frames", type=int, default=15)
    parser.add_argument("--return-confirmation-frames", type=int, default=3)
    parser.add_argument("--close-ramp-frames", type=int, default=10)
    parser.add_argument("--open-ramp-frames", type=int, default=10)
    parser.add_argument("--object-width-m", type=float, default=0.06)
    parser.add_argument("--grasp-compression-m", type=float, default=0.0)
    parser.add_argument("--gripper-min-width-m", type=float, default=0.035)
    parser.add_argument("--gripper-max-width-m", type=float, default=0.120)
    parser.add_argument("--grasp-command", type=float, default=None)
    parser.add_argument("--close-start-frame", type=int, default=None)
    parser.add_argument("--close-end-frame", type=int, default=None)
    parser.add_argument("--open-start-frame", type=int, default=None)
    parser.add_argument("--open-end-frame", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the recomputed boundaries and values without overwriting anything",
    )
    args = parser.parse_args(argv)

    # Reuse the batch script's validation so both entry points accept exactly
    # the same trajectory/calibration ranges.
    validation_argv = [
        "--input-root",
        str(args.input_root),
        "--episode-idx",
        str(args.episode_idx),
        "--pose-field",
        args.pose_field,
        "--smooth-window",
        str(args.smooth_window),
        "--minimum-prominence-m",
        str(args.minimum_prominence_m),
        "--minimum-distance-frames",
        str(args.minimum_distance_frames),
        "--return-confirmation-frames",
        str(args.return_confirmation_frames),
        "--close-ramp-frames",
        str(args.close_ramp_frames),
        "--open-ramp-frames",
        str(args.open_ramp_frames),
        "--object-width-m",
        str(args.object_width_m),
        "--grasp-compression-m",
        str(args.grasp_compression_m),
        "--gripper-min-width-m",
        str(args.gripper_min_width_m),
        "--gripper-max-width-m",
        str(args.gripper_max_width_m),
    ]
    for option, value in (
        ("--grasp-command", args.grasp_command),
        ("--close-start-frame", args.close_start_frame),
        ("--close-end-frame", args.close_end_frame),
        ("--open-start-frame", args.open_start_frame),
        ("--open-end-frame", args.open_end_frame),
    ):
        if value is not None:
            validation_argv.extend((option, str(value)))
    if args.dry_run:
        validation_argv.append("--dry-run")
    return batch_adjuster.parse_args(validation_argv)


def update_summary(root: Path, adjustment_report: dict[str, Any]) -> None:
    summary_path = root / "retarget_summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_index = int(adjustment_report["episode_index"])
    previous = summary.get("gripper_in_place_adjustments", [])
    if not isinstance(previous, list):
        previous = []
    entries = [
        item
        for item in previous
        if not isinstance(item, dict) or item.get("episode_index") != episode_index
    ]
    entries.append(
        {
            "episode_index": episode_index,
            "method": adjustment_report["method"],
            "grasp_target": adjustment_report["grasp_target"],
            "boundaries": adjustment_report["boundaries"],
            "report": str(root / f"episode_{episode_index:06d}_gripper_adjustment.json"),
        }
    )
    entries.sort(key=lambda item: int(item.get("episode_index", -1)))
    summary["gripper_in_place_adjustments"] = entries
    batch_adjuster.write_json(summary_path, summary)


def adjust_in_place(args: argparse.Namespace) -> Path | None:
    root = refuse_protected_dataset_write(
        args.input_root,
        purpose="adjust a gripper trajectory in place",
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Input retarget root does not exist: {root}")

    episode_path = root / f"episode_{args.episode_idx:06d}.npz"
    if not episode_path.is_file():
        raise FileNotFoundError(f"Retarget episode does not exist: {episode_path}")

    print("Single-episode in-place gripper adjustment")
    print(f"  target:             {episode_path}")
    print("  other episode NPZs: untouched")
    adjustment = batch_adjuster.prepare_episode_adjustment(
        args,
        input_root=root,
        output_root=root,
        episode_index=args.episode_idx,
    )
    adjustment.report["original_files_modified"] = True
    adjustment.report["write_mode"] = "atomic_in_place_single_episode"

    if args.dry_run:
        print("Dry run completed; no files were written.")
        return None

    # NamedTemporaryFile + os.replace in the shared helper guarantees that a
    # failed serialization cannot leave a partially written episode archive.
    batch_adjuster.write_npz_atomic(episode_path, adjustment.payload)
    report_path = root / f"episode_{args.episode_idx:06d}_gripper_adjustment.json"
    batch_adjuster.write_json(report_path, adjustment.report)
    batch_adjuster.update_episode_report(root, args.episode_idx, adjustment.report)
    update_summary(root, adjustment.report)

    with np.load(episode_path, allow_pickle=False) as archive:
        actual = np.asarray(archive["action_effector"], dtype=np.float32)
    expected = np.asarray(adjustment.payload["action_effector"], dtype=np.float32)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise batch_adjuster.GripperAdjustmentError(
            f"Post-write verification failed for {episode_path}"
        )

    print(f"Overwritten atomically: {episode_path}")
    print(f"Saved report:           {report_path}")
    print("All other episode NPZ files remain unchanged.")
    return episode_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adjust_in_place(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        batch_adjuster.GripperAdjustmentError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
