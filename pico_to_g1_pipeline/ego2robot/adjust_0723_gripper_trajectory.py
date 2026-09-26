#!/usr/bin/env python3
"""Adjust gripper trajectories in a fixed-SPINE3 0723 retarget output tree.

The source directory is never modified.  The full retarget output tree is
copied to a new sibling directory, then every episode NPZ is updated by
default.  Pass ``--episode-idx`` to process only one episode.

* left gripper: always open (1.0)
* right gripper: ramp 1->grasp command before the first height minimum
* right gripper: hold the grasp command while the hand lifts and descends again
* right gripper: keep holding through placement
* right gripper: ramp grasp command->1 once return-to-initial motion starts

The grasp command is calculated from the object width and the physical gripper
opening range.  The A2D SDK maps 35..120 mm to [0, 1], so a 60 mm object maps
to approximately 0.294 instead of fully closing to 0.

The two height minima are detected from ``target_eef_wxyz`` by default.  The
second minimum marks placement; opening begins only after consecutive frames
confirm that the right hand is moving back toward its initial position.  All
four ramp boundary frames can be overridden after using ``--dry-run``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))
from pipeline_safety import refuse_protected_dataset_write

DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "outputs/fixed_spine3_to_g1_0723_complete"
)
RIGHT_Z_COLUMN = 9


class GripperAdjustmentError(RuntimeError):
    """Raised when a safe, unambiguous adjustment cannot be generated."""


@dataclass
class EpisodeAdjustment:
    episode_index: int
    episode_name: str
    payload: dict[str, np.ndarray]
    report: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <input-root>_gripper_adjusted",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="Process only this episode; default: process every episode_*.npz",
    )
    parser.add_argument(
        "--pose-field",
        choices=("target_eef_wxyz", "achieved_eef_wxyz", "human_eef_fixed_spine3_xyzw"),
        default="target_eef_wxyz",
    )
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--minimum-prominence-m", type=float, default=0.02)
    parser.add_argument("--minimum-distance-frames", type=int, default=15)
    parser.add_argument(
        "--return-confirmation-frames",
        type=int,
        default=3,
        help="Consecutive distance-reduction steps required to confirm return (default: 3)",
    )
    parser.add_argument("--close-ramp-frames", type=int, default=10)
    parser.add_argument("--open-ramp-frames", type=int, default=10)
    parser.add_argument(
        "--object-width-m",
        type=float,
        default=0.06,
        help="Object width at the grasp point in metres (default: 0.06)",
    )
    parser.add_argument(
        "--grasp-compression-m",
        type=float,
        default=0.0,
        help=(
            "Amount subtracted from object width for grasp contact; keep 0 in "
            "rigid simulation to avoid penetration (default: 0)"
        ),
    )
    parser.add_argument(
        "--gripper-min-width-m",
        type=float,
        default=0.035,
        help="Physical opening represented by command 0 (A2D default: 0.035)",
    )
    parser.add_argument(
        "--gripper-max-width-m",
        type=float,
        default=0.120,
        help="Physical opening represented by command 1 (A2D default: 0.120)",
    )
    parser.add_argument(
        "--grasp-command",
        type=float,
        default=None,
        help="Direct [0,1] grasp command override; bypasses physical-width conversion",
    )
    parser.add_argument("--close-start-frame", type=int, default=None)
    parser.add_argument("--close-end-frame", type=int, default=None)
    parser.add_argument("--open-start-frame", type=int, default=None)
    parser.add_argument("--open-end-frame", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace output-root if it exists; the input root is still never modified",
    )
    args = parser.parse_args(argv)

    if args.episode_idx is not None and args.episode_idx < 0:
        parser.error("--episode-idx must be non-negative")
    if args.smooth_window < 3 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window must be an odd integer >= 3")
    if args.minimum_prominence_m <= 0:
        parser.error("--minimum-prominence-m must be positive")
    for name in (
        "minimum_distance_frames",
        "return_confirmation_frames",
        "close_ramp_frames",
        "open_ramp_frames",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "close_start_frame",
        "close_end_frame",
        "open_start_frame",
        "open_end_frame",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not np.isfinite(args.object_width_m) or args.object_width_m <= 0:
        parser.error("--object-width-m must be finite and positive")
    if not np.isfinite(args.grasp_compression_m) or args.grasp_compression_m < 0:
        parser.error("--grasp-compression-m must be finite and non-negative")
    if args.grasp_compression_m >= args.object_width_m:
        parser.error("--grasp-compression-m must be smaller than --object-width-m")
    if (
        not np.isfinite(args.gripper_min_width_m)
        or not np.isfinite(args.gripper_max_width_m)
        or args.gripper_min_width_m < 0
        or args.gripper_max_width_m <= args.gripper_min_width_m
    ):
        parser.error(
            "gripper widths must satisfy 0 <= --gripper-min-width-m "
            "< --gripper-max-width-m"
        )
    if args.grasp_command is not None and (
        not np.isfinite(args.grasp_command) or not 0.0 <= args.grasp_command <= 1.0
    ):
        parser.error("--grasp-command must be finite and within [0,1]")
    return args


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Retarget episode does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_pose_field(payload: dict[str, np.ndarray], field: str) -> np.ndarray:
    if field not in payload:
        raise GripperAdjustmentError(f"NPZ has no pose field {field!r}")
    poses = np.asarray(payload[field], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 14:
        raise GripperAdjustmentError(f"{field} has shape {poses.shape}, expected (N, 14)")
    if len(poses) < 3 or not np.isfinite(poses).all():
        raise GripperAdjustmentError(f"{field} is too short or contains non-finite values")
    return poses


def effective_smooth_window(requested: int, frame_count: int) -> int:
    maximum = frame_count if frame_count % 2 else frame_count - 1
    window = min(requested, maximum)
    if window < 3:
        raise GripperAdjustmentError("Trajectory is too short for height smoothing")
    return window


def detect_height_minima(
    right_z: np.ndarray,
    *,
    smooth_window: int,
    minimum_prominence_m: float,
    minimum_distance_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = effective_smooth_window(smooth_window, len(right_z))
    smoothed = savgol_filter(right_z, window, 2, mode="interp")
    minima, properties = find_peaks(
        -smoothed,
        prominence=minimum_prominence_m,
        distance=minimum_distance_frames,
    )
    if len(minima) < 2:
        raise GripperAdjustmentError(
            f"Detected only {len(minima)} right-hand height minima; need at least 2. "
            "Adjust --minimum-prominence-m/--minimum-distance-frames or provide "
            "--close-end-frame and --open-end-frame."
        )
    # This adjustment is specifically defined by the first two descend/lift/
    # descend minima in chronological order.
    selected = minima[:2].astype(np.int64)
    prominences = np.asarray(properties["prominences"][:2], dtype=np.float64)
    return selected, prominences, smoothed


def detect_return_start(
    right_positions: np.ndarray,
    *,
    placement_frame: int,
    confirmation_frames: int,
) -> tuple[int, np.ndarray]:
    """Detect return onset from decreasing distance to the initial position.

    Searching starts at the placement frame, so noisy motion before the object
    reaches the destination can never trigger an early release.
    """
    positions = np.asarray(right_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise GripperAdjustmentError(
            f"right_positions has shape {positions.shape}, expected (N, 3)"
        )
    distances = np.linalg.norm(positions - positions[0], axis=1)
    final_candidate = len(positions) - confirmation_frames - 1
    if placement_frame > final_candidate:
        raise GripperAdjustmentError(
            "Not enough frames after placement to confirm return-to-initial motion"
        )

    # Requiring every step in the confirmation window to reduce distance is
    # intentionally conservative: opening late is safer than releasing the
    # object before placement has completed.
    for candidate in range(placement_frame, final_candidate + 1):
        changes = np.diff(distances[candidate : candidate + confirmation_frames + 1])
        if np.all(changes < 0.0):
            return candidate, distances
    raise GripperAdjustmentError(
        "Could not confirm return-to-initial motion after the second height minimum. "
        "Adjust --return-confirmation-frames or provide --open-start-frame."
    )


def resolve_boundaries(
    args: argparse.Namespace,
    frame_count: int,
    detected_minima: np.ndarray,
    detected_return_start: int,
) -> tuple[int, int, int, int]:
    close_end = (
        args.close_end_frame if args.close_end_frame is not None else int(detected_minima[0])
    )
    open_start = (
        args.open_start_frame
        if args.open_start_frame is not None
        else detected_return_start
    )
    open_end = (
        args.open_end_frame
        if args.open_end_frame is not None
        else open_start + args.open_ramp_frames
    )
    close_start = (
        args.close_start_frame
        if args.close_start_frame is not None
        else close_end - args.close_ramp_frames
    )
    boundaries = (close_start, close_end, open_start, open_end)
    if not 0 <= close_start < close_end <= open_start < open_end < frame_count:
        raise GripperAdjustmentError(
            "Gripper boundaries must satisfy "
            f"0 <= close_start < close_end <= open_start < open_end < {frame_count}; "
            f"got {boundaries}"
        )
    return boundaries


def resolve_grasp_command(args: argparse.Namespace) -> tuple[float, float]:
    """Return normalized command and corresponding target physical opening."""
    target_width = args.object_width_m - args.grasp_compression_m
    if args.grasp_command is not None:
        command_value = float(args.grasp_command)
        target_width = args.gripper_min_width_m + command_value * (
            args.gripper_max_width_m - args.gripper_min_width_m
        )
        return command_value, target_width

    if not args.gripper_min_width_m <= target_width <= args.gripper_max_width_m:
        raise GripperAdjustmentError(
            "Requested grasp opening is outside the calibrated gripper range: "
            f"target={target_width:.6f} m, range="
            f"[{args.gripper_min_width_m:.6f}, {args.gripper_max_width_m:.6f}] m. "
            "Adjust the width calibration or use --grasp-command explicitly."
        )
    command_value = (target_width - args.gripper_min_width_m) / (
        args.gripper_max_width_m - args.gripper_min_width_m
    )
    return float(command_value), float(target_width)


def build_gripper_trajectory(
    frame_count: int,
    close_start: int,
    close_end: int,
    open_start: int,
    open_end: int,
    grasp_command: float,
) -> np.ndarray:
    if not np.isfinite(grasp_command) or not 0.0 <= grasp_command <= 1.0:
        raise GripperAdjustmentError(
            f"grasp_command must be finite and within [0,1], got {grasp_command}"
        )
    gripper = np.ones((frame_count, 2), dtype=np.float32)
    gripper[close_start : close_end + 1, 1] = np.linspace(
        1.0, grasp_command, close_end - close_start + 1, dtype=np.float32
    )
    gripper[close_end : open_start + 1, 1] = grasp_command
    gripper[open_start : open_end + 1, 1] = np.linspace(
        grasp_command, 1.0, open_end - open_start + 1, dtype=np.float32
    )
    gripper[open_end:, 1] = 1.0
    return gripper


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary_name: str | None = None
    original_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o664
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".npz",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, **payload)
        os.replace(temporary_name, path)
        path.chmod(original_mode)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_episode_report(
    output_root: Path,
    episode_index: int,
    adjustment_report: dict[str, Any],
) -> None:
    episode_report_path = output_root / f"episode_{episode_index:06d}_report.json"
    if episode_report_path.is_file():
        report = json.loads(episode_report_path.read_text(encoding="utf-8"))
        report["gripper"] = adjustment_report["gripper"]
        report["gripper_adjustment_report"] = str(
            output_root / f"episode_{episode_index:06d}_gripper_adjustment.json"
        )
        report["gripper_adjustment"] = {
            "method": adjustment_report["method"],
            "boundaries": adjustment_report["boundaries"],
            "grasp_target": adjustment_report["grasp_target"],
            "return_detection": adjustment_report["return_detection"],
        }
        write_json(episode_report_path, report)


def update_summary_report(
    output_root: Path,
    adjustments: list[EpisodeAdjustment],
) -> None:
    summary_path = output_root / "retarget_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["gripper_adjustment"] = {
            "schema": "retargeted_gripper_batch_adjustment.v1",
            "method": adjustments[0].report["method"],
            "episode_count": len(adjustments),
            "episode_indices": [item.episode_index for item in adjustments],
            "grasp_target": adjustments[0].report["grasp_target"],
            "reports": [
                str(
                    output_root
                    / f"episode_{item.episode_index:06d}_gripper_adjustment.json"
                )
                for item in adjustments
            ],
        }
        write_json(summary_path, summary)


def discover_episode_indices(input_root: Path, requested: int | None) -> list[int]:
    if requested is not None:
        path = input_root / f"episode_{requested:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Retarget episode does not exist: {path}")
        return [requested]

    indices: list[int] = []
    for path in sorted(input_root.glob("episode_*.npz")):
        suffix = path.stem.removeprefix("episode_")
        if len(suffix) == 6 and suffix.isdigit():
            indices.append(int(suffix))
    if not indices:
        raise FileNotFoundError(f"No episode_*.npz files found in {input_root}")
    if len(indices) != len(set(indices)):
        raise GripperAdjustmentError("Duplicate episode indices were discovered")
    return indices


def prepare_episode_adjustment(
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    episode_index: int,
) -> EpisodeAdjustment:
    episode_name = f"episode_{episode_index:06d}.npz"
    source_path = input_root / episode_name
    payload = load_npz(source_path)
    poses = validate_pose_field(payload, args.pose_field)
    right_z = poses[:, RIGHT_Z_COLUMN]
    detected, prominences, smoothed_z = detect_height_minima(
        right_z,
        smooth_window=args.smooth_window,
        minimum_prominence_m=args.minimum_prominence_m,
        minimum_distance_frames=args.minimum_distance_frames,
    )
    right_positions = poses[:, 7:10]
    if args.open_start_frame is None:
        return_start, initial_position_distances = detect_return_start(
            right_positions,
            placement_frame=int(detected[1]),
            confirmation_frames=args.return_confirmation_frames,
        )
        return_start_source = "automatic_consecutive_distance_reduction"
    else:
        return_start = args.open_start_frame
        initial_position_distances = np.linalg.norm(
            right_positions - right_positions[0], axis=1
        )
        return_start_source = "--open-start-frame override"
    close_start, close_end, open_start, open_end = resolve_boundaries(
        args, len(poses), detected, return_start
    )
    grasp_command, target_opening_width_m = resolve_grasp_command(args)
    gripper = build_gripper_trajectory(
        len(poses), close_start, close_end, open_start, open_end, grasp_command
    )

    report: dict[str, Any] = {
        "schema": "retargeted_gripper_height_adjustment.v3",
        "method": "placement_then_return_triggered_width_calibrated_gripper",
        "source_root": str(input_root),
        "output_root": str(output_root),
        "episode_index": episode_index,
        "frames": len(poses),
        "pose_field": args.pose_field,
        "right_height_column": RIGHT_Z_COLUMN,
        "detected_minima": [
            {
                "frame": int(frame),
                "right_z_m": float(right_z[frame]),
                "smoothed_right_z_m": float(smoothed_z[frame]),
                "prominence_m": float(prominence),
            }
            for frame, prominence in zip(detected, prominences, strict=True)
        ],
        "return_detection": {
            "placement_frame": int(detected[1]),
            "return_start_frame": int(return_start),
            "return_start_source": return_start_source,
            "confirmation_frames": args.return_confirmation_frames,
            "distance_to_initial_position_m": float(
                initial_position_distances[return_start]
            ),
            "confirmed_distances_m": initial_position_distances[
                return_start : return_start + args.return_confirmation_frames + 1
            ].astype(float).tolist(),
        },
        "boundaries": {
            "close_start_frame": close_start,
            "close_end_frame": close_end,
            "open_start_frame": open_start,
            "open_end_frame": open_end,
        },
        "grasp_target": {
            "object_width_m": args.object_width_m,
            "grasp_compression_m": args.grasp_compression_m,
            "target_opening_width_m": target_opening_width_m,
            "gripper_min_width_m": args.gripper_min_width_m,
            "gripper_max_width_m": args.gripper_max_width_m,
            "command": grasp_command,
            "command_source": (
                "--grasp-command override"
                if args.grasp_command is not None
                else "linear physical-width calibration"
            ),
        },
        "gripper": {
            "fields": ["hand_status", "action_effector"],
            "order": ["left", "right"],
            "meaning": "0=minimum calibrated opening, 1=maximum calibrated opening",
            "left": "constant 1",
            "right": f"1 -> {grasp_command:.6f} -> hold -> 1 -> hold 1",
            "minimum": gripper.min(axis=0).astype(float).tolist(),
            "maximum": gripper.max(axis=0).astype(float).tolist(),
            "valid": "action_effector_valid is true for every adjusted command",
        },
        "original_files_modified": False,
    }

    print(f"Episode {episode_index:06d} ({len(poses)} frames)")
    print(
        "  detected minima:    "
        + ", ".join(f"frame {int(i)} (z={right_z[i]:.5f} m)" for i in detected)
    )
    print(f"  placement frame:    {int(detected[1])}")
    print(
        f"  return starts:      frame {return_start} "
        f"(distance to initial={initial_position_distances[return_start]:.5f} m)"
    )
    print(f"  object width:       {args.object_width_m:.5f} m")
    print(f"  target opening:     {target_opening_width_m:.5f} m")
    print(f"  grasp command:      {grasp_command:.6f}")
    print(
        f"  close ramp:         frames {close_start}..{close_end} "
        f"(1 -> {grasp_command:.6f})"
    )
    print(
        f"  grasp hold:         frames {close_end}..{open_start} "
        f"({grasp_command:.6f})"
    )
    print(
        f"  open ramp:          frames {open_start}..{open_end} "
        f"({grasp_command:.6f} -> 1)"
    )
    print(f"  open hold:          frames {open_end}..{len(poses) - 1} (1)")

    payload["hand_status"] = gripper.copy()
    payload["action_effector"] = gripper.copy()
    if "hand_status_valid" not in payload:
        payload["hand_status_valid"] = np.ones_like(gripper, dtype=np.bool_)
    else:
        hand_valid = np.asarray(payload["hand_status_valid"], dtype=np.bool_)
        if hand_valid.shape != gripper.shape:
            raise GripperAdjustmentError(
                f"hand_status_valid has shape {hand_valid.shape}, expected {gripper.shape}"
            )
    payload["action_effector_valid"] = np.ones_like(gripper, dtype=np.bool_)
    return EpisodeAdjustment(
        episode_index=episode_index,
        episode_name=episode_name,
        payload=payload,
        report=report,
    )


def adjust(args: argparse.Namespace) -> Path | None:
    input_root = args.input_root.expanduser().resolve()
    output_root = (
        refuse_protected_dataset_write(
            args.output_root,
            purpose="write adjusted gripper output",
        )
        if args.output_root is not None
        else refuse_protected_dataset_write(
            input_root.with_name(f"{input_root.name}_gripper_adjusted"),
            purpose="write adjusted gripper output",
        )
    )
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input retarget root does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents or output_root in input_root.parents:
        raise GripperAdjustmentError("Input and output must be separate sibling directory trees")

    episode_indices = discover_episode_indices(input_root, args.episode_idx)
    selection = "all episodes" if args.episode_idx is None else f"episode {args.episode_idx}"
    print("0723 retargeted gripper adjustment")
    print(f"  input:              {input_root}")
    print(f"  output:             {output_root}")
    print(f"  selection:          {selection} ({len(episode_indices)} episode(s))")

    # Prepare every episode before copying or writing anything. If automatic
    # detection fails for one trajectory, no partial output tree is created.
    adjustments = [
        prepare_episode_adjustment(args, input_root, output_root, episode_index)
        for episode_index in episode_indices
    ]
    if args.dry_run:
        print(f"Dry run completed for {len(adjustments)} episode(s); no files were written.")
        return None

    if output_root.exists():
        if not args.overwrite_output:
            raise GripperAdjustmentError(
                f"Output already exists: {output_root}; use --overwrite-output to replace it"
            )
        shutil.rmtree(output_root)
    shutil.copytree(input_root, output_root)

    for item in adjustments:
        destination_path = output_root / item.episode_name
        write_npz_atomic(destination_path, item.payload)
        adjustment_path = (
            output_root / f"episode_{item.episode_index:06d}_gripper_adjustment.json"
        )
        write_json(adjustment_path, item.report)
        update_episode_report(output_root, item.episode_index, item.report)
    update_summary_report(output_root, adjustments)
    print(f"Saved {len(adjustments)} adjusted episode(s) to: {output_root}")
    print("Original input tree remains unchanged.")
    return output_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adjust(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, GripperAdjustmentError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
