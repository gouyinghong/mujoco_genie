#!/usr/bin/env python3
"""Retarget hands expressed in the episode-0/frame-0 SPINE3 frame to G1.

The source split LeRobot dataset stores each hand pose relative to the SPINE3
frame of the same tracking frame and stores that frame's world pose in
``spine3_world_xyzw``.  The split dataset is therefore the only human-data
input required by this script.

The parent-frame change is

    T_S0_H(t) = inverse(T_W_S0) * T_W_S(t) * T_S(t)_H(t)

where S0 defaults to episode 0, frame 0 of the split human dataset.  The raw
PICO SPINE3 transform is converted to the same x-forward/y-left/z-up parent
axis convention already used by ``action_eef`` before composition.

After rebasing, the script re-estimates the same robust quantile/mean-rotation
mapping used by ``estimate_spine3_to_g1_mapping.py`` and reuses the IK path in
``retarget_spine3_to_g1.py``.  The old moving-SPINE3 mapping JSON is therefore
not reused.  Each episode's resulting ``action_joint_position`` is smoothed
before saving with endpoint-anchored Savitzky-Golay filtering by default.

Example:

    ./.venv/bin/python ego2robot/retarget_fixed_spine3_to_g1.py \
      --episode-idx 0 --max-frames 20 \
      --disable-auto-gripper \
      --output-dir /tmp/fixed_spine3_smoke

By default, the final ``hand_status`` and ``action_effector`` are generated
from the right EEF grasp/place/return motion for a 6 cm object.  The original
PICO fingertip-distance signal remains available as ``source_hand_status``.
This is an offline-only script and never commands the robot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ego2robot.estimate_spine3_to_g1_mapping import (  # noqa: E402
    estimate_side,
    load_action_eef,
)
from ego2robot.adjust_0723_gripper_trajectory import (  # noqa: E402
    build_gripper_trajectory,
    detect_height_minima,
    detect_return_start,
    resolve_boundaries,
    resolve_grasp_command,
)
from ego2robot.retarget_spine3_to_g1 import (  # noqa: E402
    ARM_JOINTS,
    G1ArmIK,
    load_robot_reference,
    read_parquet_columns,
    retarget_episode as run_mapped_ik,
)
from pipeline_safety import refuse_protected_dataset_write  # noqa: E402


DEFAULT_HUMAN_ROOT = (
    PROJECT_ROOT
    / "work/lerobot_session_20260723_084253_461_split_self_contained"
)
DEFAULT_ROBOT_ROOT = PROJECT_ROOT / "data/reference/genie1_pick_up_dice_804"
DEFAULT_URDF = PROJECT_ROOT / "assets/G1_120s/G1_120s.urdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/fixed_spine3_to_g1_0723_complete"

# Existing human action_eef parent coordinates are related to raw PICO body
# joint coordinates by p_action = A @ p_pico.
PICO_PARENT_TO_ACTION = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


class FixedSpine3RetargetError(RuntimeError):
    """Raised when a fixed-SPINE3 coordinate assumption is not satisfied."""


@dataclass(frozen=True)
class Spine3WorldPose:
    source_frame_index: int
    timestamp_ns: int
    position_world: np.ndarray
    rotation_world: Rotation


@dataclass(frozen=True)
class FixedSpine3Reference:
    episode_index: int
    frame_index: int
    source_frame_index: int
    timestamp_ns: int
    position_world: np.ndarray
    rotation_world: Rotation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN_ROOT)
    parser.add_argument("--robot-root", type=Path, default=DEFAULT_ROBOT_ROOT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episode-idx", type=int, default=0)
    selection.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)

    parser.add_argument("--reference-episode-idx", type=int, default=0)
    parser.add_argument("--reference-frame-idx", type=int, default=0)
    parser.add_argument("--low-quantile", type=float, default=0.05)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="Mapping JSON path (default: <output-dir>/fixed_spine3_to_g1_mapping.json)",
    )

    parser.add_argument("--left-mode", choices=("fixed", "mapped"), default="fixed")
    parser.add_argument("--fixed-left-reference-index", type=int, default=0)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.001)
    parser.add_argument("--posture-weight", type=float, default=0.0001)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--position-success-m", type=float, default=0.03)
    parser.add_argument("--orientation-success-deg", type=float, default=30.0)
    parser.add_argument(
        "--joint-smooth-window",
        type=int,
        default=11,
        help=(
            "Odd Savitzky-Golay window for action_joint_position; "
            "0 disables smoothing (default: 11)"
        ),
    )
    parser.add_argument("--joint-smooth-polyorder", type=int, default=2)
    parser.add_argument("--joint-smooth-passes", type=int, default=1)
    parser.add_argument("--disable-auto-gripper", action="store_true")
    parser.add_argument(
        "--gripper-pose-field",
        choices=("target_eef_wxyz", "achieved_eef_wxyz"),
        default="target_eef_wxyz",
    )
    parser.add_argument("--gripper-smooth-window", type=int, default=11)
    parser.add_argument("--gripper-minimum-prominence-m", type=float, default=0.02)
    parser.add_argument("--gripper-minimum-distance-frames", type=int, default=15)
    parser.add_argument("--gripper-return-confirmation-frames", type=int, default=3)
    parser.add_argument("--gripper-close-ramp-frames", type=int, default=10)
    parser.add_argument("--gripper-open-ramp-frames", type=int, default=10)
    parser.add_argument("--gripper-object-width-m", type=float, default=0.06)
    parser.add_argument("--gripper-grasp-compression-m", type=float, default=0.0)
    parser.add_argument("--gripper-min-width-m", type=float, default=0.035)
    parser.add_argument("--gripper-max-width-m", type=float, default=0.120)
    parser.add_argument("--gripper-grasp-command", type=float, default=None)
    args = parser.parse_args(argv)

    for name in (
        "episode_idx",
        "reference_episode_idx",
        "reference_frame_idx",
        "fixed_left_reference_index",
        "max_frames",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 <= args.low_quantile < 0.5:
        parser.error("--low-quantile must be in [0, 0.5)")
    if not 0.5 < args.high_quantile <= 1.0:
        parser.error("--high-quantile must be in (0.5, 1]")
    if args.max_nfev < 1:
        parser.error("--max-nfev must be positive")
    if args.joint_smooth_window != 0 and (
        args.joint_smooth_window < 3 or args.joint_smooth_window % 2 == 0
    ):
        parser.error("--joint-smooth-window must be 0 or an odd integer >= 3")
    if args.joint_smooth_polyorder < 1:
        parser.error("--joint-smooth-polyorder must be positive")
    if (
        args.joint_smooth_window
        and args.joint_smooth_polyorder >= args.joint_smooth_window
    ):
        parser.error(
            "--joint-smooth-polyorder must be smaller than --joint-smooth-window"
        )
    if args.joint_smooth_passes < 1:
        parser.error("--joint-smooth-passes must be positive")
    if args.gripper_smooth_window < 3 or args.gripper_smooth_window % 2 == 0:
        parser.error("--gripper-smooth-window must be an odd integer >= 3")
    for name in (
        "gripper_minimum_distance_frames",
        "gripper_return_confirmation_frames",
        "gripper_close_ramp_frames",
        "gripper_open_ramp_frames",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gripper_minimum_prominence_m <= 0:
        parser.error("--gripper-minimum-prominence-m must be positive")
    if not np.isfinite(args.gripper_object_width_m) or args.gripper_object_width_m <= 0:
        parser.error("--gripper-object-width-m must be finite and positive")
    if (
        not np.isfinite(args.gripper_grasp_compression_m)
        or args.gripper_grasp_compression_m < 0
        or args.gripper_grasp_compression_m >= args.gripper_object_width_m
    ):
        parser.error(
            "--gripper-grasp-compression-m must be finite, non-negative, and "
            "smaller than --gripper-object-width-m"
        )
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
    if args.gripper_grasp_command is not None and (
        not np.isfinite(args.gripper_grasp_command)
        or not 0.0 <= args.gripper_grasp_command <= 1.0
    ):
        parser.error("--gripper-grasp-command must be finite and within [0,1]")
    for name in (
        "position_weight",
        "orientation_weight",
        "smoothness_weight",
        "posture_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def endpoint_weights(frame_count: int, window_length: int) -> np.ndarray:
    """Return blend weights that preserve both trajectory endpoints."""
    half = window_length // 2
    weights = np.ones(frame_count, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, half + 1)
    weights[: half + 1] = ramp
    weights[-half - 1 :] = np.minimum(weights[-half - 1 :], ramp[::-1])
    return weights


def smooth_joint_positions(
    values: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
    passes: int,
) -> np.ndarray:
    """Match the former in-place script's endpoint-anchored joint smoothing."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise FixedSpine3RetargetError(
            f"joint trajectory must be a finite 2-D array, got {array.shape}"
        )
    if len(array) < window_length:
        raise FixedSpine3RetargetError(
            f"joint trajectory has {len(array)} frames, shorter than smoothing "
            f"window {window_length}; use --joint-smooth-window 0 for short smoke tests"
        )
    weights = endpoint_weights(len(array), window_length)[:, None]
    smoothed = array.copy()
    for _ in range(passes):
        filtered = savgol_filter(
            smoothed,
            window_length=window_length,
            polyorder=polyorder,
            axis=0,
            mode="interp",
        )
        smoothed += weights * (filtered - smoothed)
    smoothed[0] = array[0]
    smoothed[-1] = array[-1]
    return smoothed


def joint_step_stats(joints: np.ndarray) -> dict[str, Any]:
    array = np.asarray(joints, dtype=np.float64)
    if len(array) < 2:
        raise FixedSpine3RetargetError(
            "joint trajectory needs at least two frames for step statistics"
        )
    steps = np.abs(np.diff(array, axis=0))
    transition, joint = np.unravel_index(int(np.argmax(steps)), steps.shape)
    return {
        "max_step_rad": float(steps[transition, joint]),
        "transition": [int(transition), int(transition + 1)],
        "joint_index": int(joint),
        "rms_step_rad": float(np.sqrt(np.mean(steps**2))),
    }


def smooth_result_joints(
    result: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any] | None:
    if args.joint_smooth_window == 0:
        return None
    if "action_joint_position" not in result:
        raise FixedSpine3RetargetError(
            "IK result has no action_joint_position field"
        )
    original_value = np.asarray(result["action_joint_position"])
    original = np.asarray(original_value, dtype=np.float64)
    smoothed = smooth_joint_positions(
        original,
        window_length=args.joint_smooth_window,
        polyorder=args.joint_smooth_polyorder,
        passes=args.joint_smooth_passes,
    )
    result["action_joint_position"] = smoothed.astype(
        original_value.dtype, copy=False
    )
    return {
        "method": "endpoint_anchored_savgol_with_rotation_vector_orientation",
        "mode": "joints",
        "fields": ["action_joint_position"],
        "parameters": {
            "window_length": args.joint_smooth_window,
            "polyorder": args.joint_smooth_polyorder,
            "passes": args.joint_smooth_passes,
            "endpoints_preserved": True,
        },
        "metrics_before": {
            "action_joint_position": joint_step_stats(original),
        },
        "metrics_after": {
            "action_joint_position": joint_step_stats(smoothed),
        },
        "eef_recomputed_after_joint_smoothing": False,
    }


def generate_auto_gripper(
    result: dict[str, Any], args: argparse.Namespace
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Generate the same 0723 grasp/hold/return command during retargeting."""
    if args.disable_auto_gripper:
        return None, None
    if args.gripper_pose_field not in result:
        raise FixedSpine3RetargetError(
            f"IK result has no {args.gripper_pose_field!r} field"
        )
    poses = np.asarray(result[args.gripper_pose_field], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 14 or not np.isfinite(poses).all():
        raise FixedSpine3RetargetError(
            f"{args.gripper_pose_field} has shape {poses.shape}, expected finite (N,14)"
        )

    right_z = poses[:, 9]
    detected, prominences, smoothed_z = detect_height_minima(
        right_z,
        smooth_window=args.gripper_smooth_window,
        minimum_prominence_m=args.gripper_minimum_prominence_m,
        minimum_distance_frames=args.gripper_minimum_distance_frames,
    )
    return_start, initial_position_distances = detect_return_start(
        poses[:, 7:10],
        placement_frame=int(detected[1]),
        confirmation_frames=args.gripper_return_confirmation_frames,
    )
    gripper_config = argparse.Namespace(
        close_start_frame=None,
        close_end_frame=None,
        open_start_frame=None,
        open_end_frame=None,
        close_ramp_frames=args.gripper_close_ramp_frames,
        open_ramp_frames=args.gripper_open_ramp_frames,
        object_width_m=args.gripper_object_width_m,
        grasp_compression_m=args.gripper_grasp_compression_m,
        gripper_min_width_m=args.gripper_min_width_m,
        gripper_max_width_m=args.gripper_max_width_m,
        grasp_command=args.gripper_grasp_command,
    )
    close_start, close_end, open_start, open_end = resolve_boundaries(
        gripper_config,
        len(poses),
        detected,
        return_start,
    )
    grasp_command, target_opening_width_m = resolve_grasp_command(gripper_config)
    gripper = build_gripper_trajectory(
        len(poses),
        close_start,
        close_end,
        open_start,
        open_end,
        grasp_command,
    )
    report = {
        "schema": "retargeted_gripper_height_adjustment.v3",
        "method": "placement_then_return_triggered_width_calibrated_gripper",
        "pose_field": args.gripper_pose_field,
        "right_height_column": 9,
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
            "return_start_source": "automatic_consecutive_distance_reduction",
            "confirmation_frames": args.gripper_return_confirmation_frames,
            "distance_to_initial_position_m": float(
                initial_position_distances[return_start]
            ),
            "confirmed_distances_m": initial_position_distances[
                return_start : return_start
                + args.gripper_return_confirmation_frames
                + 1
            ].astype(float).tolist(),
        },
        "boundaries": {
            "close_start_frame": close_start,
            "close_end_frame": close_end,
            "open_start_frame": open_start,
            "open_end_frame": open_end,
        },
        "grasp_target": {
            "object_width_m": args.gripper_object_width_m,
            "grasp_compression_m": args.gripper_grasp_compression_m,
            "target_opening_width_m": target_opening_width_m,
            "gripper_min_width_m": args.gripper_min_width_m,
            "gripper_max_width_m": args.gripper_max_width_m,
            "command": grasp_command,
            "command_source": (
                "--gripper-grasp-command override"
                if args.gripper_grasp_command is not None
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
            "valid": "both generated command validity fields are true",
        },
    }
    return gripper, report


def scalar_int_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise FixedSpine3RetargetError(
            f"{name} has shape {array.shape}, expected a scalar column"
        )
    return array.astype(np.int64)


def gripper_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise FixedSpine3RetargetError(
            f"{name} has shape {array.shape}, expected (N, 2) in left/right order"
        )
    if not np.isfinite(array).all():
        raise FixedSpine3RetargetError(f"{name} contains NaN or infinity")
    tolerance = 1e-6
    if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        raise FixedSpine3RetargetError(
            f"{name} must be in [0, 1], got range [{minimum:.6f}, {maximum:.6f}]"
        )
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def gripper_valid_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 2:
        raise FixedSpine3RetargetError(
            f"{name} has shape {array.shape}, expected (N, 2) in left/right order"
        )
    return array.astype(np.bool_)


def load_human_dataset(root: Path) -> dict[str, np.ndarray]:
    columns = [
        "action_eef",
        "episode_index",
        "frame_index",
        "source_frame_index",
        "local_timestamps_ns",
        "hand_status",
        "hand_status_valid",
    ]
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata does not exist: {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixedSpine3RetargetError(
            f"invalid LeRobot metadata JSON: {info_path}: {exc}"
        ) from exc
    features = info.get("features", {})
    if isinstance(features, dict) and "spine3_world_xyzw" in features:
        columns.append("spine3_world_xyzw")
    data = read_parquet_columns(root, columns)
    action = np.asarray(data["action_eef"], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 14:
        raise FixedSpine3RetargetError(
            f"human action_eef has shape {action.shape}, expected (N, 14)"
        )
    if not np.isfinite(action).all():
        raise FixedSpine3RetargetError("human action_eef contains NaN or infinity")
    result = {
        "action_eef": action,
        "episode_index": scalar_int_array(data["episode_index"], "episode_index"),
        "frame_index": scalar_int_array(data["frame_index"], "frame_index"),
        "source_frame_index": scalar_int_array(
            data["source_frame_index"], "source_frame_index"
        ),
        "local_timestamps_ns": scalar_int_array(
            data["local_timestamps_ns"], "local_timestamps_ns"
        ),
        "hand_status": gripper_array(data["hand_status"], "hand_status"),
        "hand_status_valid": gripper_valid_array(
            data["hand_status_valid"], "hand_status_valid"
        ),
    }
    if "spine3_world_xyzw" in data:
        spine3_world = np.asarray(data["spine3_world_xyzw"], dtype=np.float64)
        if spine3_world.ndim != 2 or spine3_world.shape[1] != 7:
            raise FixedSpine3RetargetError(
                "spine3_world_xyzw has shape "
                f"{spine3_world.shape}, expected (N, 7)"
            )
        if not np.isfinite(spine3_world).all():
            raise FixedSpine3RetargetError(
                "spine3_world_xyzw contains NaN or infinity"
            )
        quaternion_norms = np.linalg.norm(spine3_world[:, 3:7], axis=1)
        if np.any(quaternion_norms < 1e-8):
            raise FixedSpine3RetargetError(
                "spine3_world_xyzw contains zero-length quaternions"
            )
        result["spine3_world_xyzw"] = spine3_world
    lengths = {key: len(value) for key, value in result.items()}
    if len(set(lengths.values())) != 1:
        raise FixedSpine3RetargetError(f"human field lengths differ: {lengths}")
    return result


def embedded_spine3_world_poses(
    human: dict[str, np.ndarray],
) -> dict[int, Spine3WorldPose]:
    poses = human.get("spine3_world_xyzw")
    if poses is None:
        raise FixedSpine3RetargetError(
            "human dataset has no spine3_world_xyzw field"
        )
    result: dict[int, Spine3WorldPose] = {}
    for source_index, timestamp, pose in zip(
        human["source_frame_index"],
        human["local_timestamps_ns"],
        poses,
        strict=True,
    ):
        index = int(source_index)
        candidate = Spine3WorldPose(
            source_frame_index=index,
            timestamp_ns=int(timestamp),
            position_world=np.asarray(pose[:3], dtype=np.float64),
            rotation_world=Rotation.from_quat(
                normalize_xyzw(pose[3:7], f"embedded SPINE3 source frame {index}")
            ),
        )
        previous = result.get(index)
        if previous is not None and (
            previous.timestamp_ns != candidate.timestamp_ns
            or not np.allclose(previous.position_world, candidate.position_world)
            or not np.allclose(
                previous.rotation_world.as_matrix(),
                candidate.rotation_world.as_matrix(),
            )
        ):
            raise FixedSpine3RetargetError(
                f"conflicting embedded SPINE3 poses for source frame {index}"
            )
        result[index] = candidate
    return result


def locate_dataset_row(
    human: dict[str, np.ndarray], episode_index: int, frame_index: int
) -> int:
    rows = np.flatnonzero(
        (human["episode_index"] == episode_index)
        & (human["frame_index"] == frame_index)
    )
    if len(rows) != 1:
        raise FixedSpine3RetargetError(
            f"expected one row for episode={episode_index}, frame={frame_index}; "
            f"found {len(rows)}"
        )
    return int(rows[0])


def normalize_xyzw(values: np.ndarray, context: str) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(quaternion).all() or norm < 1e-8:
        raise FixedSpine3RetargetError(
            f"invalid quaternion at {context}: norm={norm}"
        )
    return quaternion / norm


def rebase_hand_pose_to_fixed_spine3(
    action_pose_xyzw: np.ndarray,
    current_spine: Spine3WorldPose,
    reference_spine: FixedSpine3Reference,
) -> np.ndarray:
    """Return one corrected hand pose relative to the fixed reference SPINE3."""
    rotation_reference_from_current_pico = (
        reference_spine.rotation_world.inv() * current_spine.rotation_world
    ).as_matrix()
    translation_reference_from_current_pico = (
        reference_spine.rotation_world.inv().apply(
            current_spine.position_world - reference_spine.position_world
        )
    )

    # Change both SPINE parent bases from raw PICO axes to the action_eef axes.
    rotation_reference_from_current_action = (
        PICO_PARENT_TO_ACTION
        @ rotation_reference_from_current_pico
        @ PICO_PARENT_TO_ACTION.T
    )
    translation_reference_from_current_action = (
        PICO_PARENT_TO_ACTION @ translation_reference_from_current_pico
    )

    output = np.empty(7, dtype=np.float64)
    output[:3] = (
        rotation_reference_from_current_action @ action_pose_xyzw[:3]
        + translation_reference_from_current_action
    )
    hand_rotation_current = Rotation.from_quat(
        normalize_xyzw(action_pose_xyzw[3:7], "human action_eef")
    )
    hand_rotation_reference = Rotation.from_matrix(
        rotation_reference_from_current_action
    ) * hand_rotation_current
    output[3:7] = hand_rotation_reference.as_quat()
    return output


def rebase_all_human_poses(
    human: dict[str, np.ndarray],
    spine_poses: dict[int, Spine3WorldPose],
    reference: FixedSpine3Reference,
) -> np.ndarray:
    output = np.empty_like(human["action_eef"], dtype=np.float64)
    previous_quaternion: dict[tuple[int, int], np.ndarray] = {}
    for row, (action, source_index, episode_index) in enumerate(
        zip(
            human["action_eef"],
            human["source_frame_index"],
            human["episode_index"],
            strict=True,
        )
    ):
        current_spine = spine_poses[int(source_index)]
        for side_index, offset in enumerate((0, 7)):
            pose = rebase_hand_pose_to_fixed_spine3(
                action[offset : offset + 7], current_spine, reference
            )
            key = (int(episode_index), side_index)
            previous = previous_quaternion.get(key)
            if previous is not None and np.dot(previous, pose[3:7]) < 0.0:
                pose[3:7] *= -1.0
            previous_quaternion[key] = pose[3:7].copy()
            output[row, offset : offset + 7] = pose
    return output


def build_mapping(
    fixed_human_eef: np.ndarray,
    robot_eef: np.ndarray,
    reference: FixedSpine3Reference,
    human_root: Path,
    robot_root: Path,
    low_quantile: float,
    high_quantile: float,
) -> dict[str, Any]:
    return {
        "schema": "fixed_spine3_to_g1_statistical_mapping.v1",
        "method": {
            "position": "Match per-axis robust quantile spans and medians.",
            "orientation": "Match scipy rotation means; scale relative rotation by p95 spread ratio.",
            "warning": "Same-task datasets are not frame paired; use this only as an initial mapping.",
        },
        "human_dataset": str(human_root),
        "robot_dataset": str(robot_root),
        "human_frames": int(len(fixed_human_eef)),
        "robot_frames": int(len(robot_eef)),
        "quantiles": [low_quantile, 0.5, high_quantile],
        "reference_frame": {
            "name": "SPINE3_EPISODE0_FRAME0",
            "episode_index": reference.episode_index,
            "frame_index": reference.frame_index,
            "source_frame_index": reference.source_frame_index,
            "timestamp_ns": reference.timestamp_ns,
            "position_world_xyz": reference.position_world.tolist(),
            "rotation_world_xyzw": reference.rotation_world.as_quat().tolist(),
            "axes": "x-forward, y-left, z-up after PICO parent-axis correction",
        },
        # Keep this key for direct compatibility with map_side_pose and the
        # existing retarget_episode implementation.
        "spine3_to_arm_base_axis_rotation": np.eye(3).tolist(),
        "left": estimate_side(
            fixed_human_eef, robot_eef, 0, low_quantile, high_quantile
        ),
        "right": estimate_side(
            fixed_human_eef, robot_eef, 7, low_quantile, high_quantile
        ),
        "recommended_first_version": {
            "left_mode": "fixed_robot_pose",
            "right_position_mapping": "use right.position_mapping",
            "right_orientation_mapping": "use right.orientation_mapping with low IK weight",
        },
    }


def episode_rows(
    human: dict[str, np.ndarray], episode_index: int, max_frames: int
) -> np.ndarray:
    rows = np.flatnonzero(human["episode_index"] == episode_index)
    if not len(rows):
        raise FixedSpine3RetargetError(f"human dataset has no episode {episode_index}")
    rows = rows[np.argsort(human["frame_index"][rows])]
    return rows[:max_frames] if max_frames else rows


def maximum_pose_steps(poses_wxyz: np.ndarray) -> tuple[float, float]:
    if len(poses_wxyz) < 2:
        return 0.0, 0.0
    max_position = 0.0
    max_rotation = 0.0
    for offset in (0, 7):
        position_steps = np.linalg.norm(
            np.diff(poses_wxyz[:, offset : offset + 3], axis=0), axis=1
        )
        max_position = max(max_position, float(np.max(position_steps)))
        rotations = Rotation.from_quat(
            poses_wxyz[:, offset + 3 : offset + 7][:, [1, 2, 3, 0]]
        )
        rotation_steps = rotations[:-1].inv() * rotations[1:]
        max_rotation = max(
            max_rotation,
            float(np.max(np.linalg.norm(rotation_steps.as_rotvec(), axis=1))),
        )
    return max_position, max_rotation


def save_episode(
    result: dict[str, Any],
    human: dict[str, np.ndarray],
    fixed_human_eef: np.ndarray,
    rows: np.ndarray,
    output_dir: Path,
    joint_smoothing: dict[str, Any] | None,
    auto_gripper: np.ndarray | None,
    gripper_adjustment: dict[str, Any] | None,
) -> dict[str, Any]:
    episode_index = int(result["episode_idx"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / f"episode_{episode_index:06d}.npz"
    source_hand_status = human["hand_status"][rows]
    source_hand_status_valid = human["hand_status_valid"][rows]
    if auto_gripper is None:
        hand_status = source_hand_status.copy()
        hand_status_valid = source_hand_status_valid.copy()
    else:
        if auto_gripper.shape != source_hand_status.shape:
            raise FixedSpine3RetargetError(
                f"generated gripper has shape {auto_gripper.shape}, expected "
                f"{source_hand_status.shape}"
            )
        hand_status = auto_gripper
        hand_status_valid = np.ones_like(auto_gripper, dtype=np.bool_)
    payload = {
        "episode_frame_index": human["frame_index"][rows],
        "source_frame_index": human["source_frame_index"][rows],
        "local_timestamps_ns": human["local_timestamps_ns"][rows],
        "source_action_eef_spine3_xyzw": human["action_eef"][rows],
        "human_eef_fixed_spine3_xyzw": fixed_human_eef[rows],
        # Preserve the PICO fingertip-distance signal for provenance while the
        # standard command fields use the generated grasp/return trajectory.
        "source_hand_status": source_hand_status,
        "source_hand_status_valid": source_hand_status_valid,
        "hand_status": hand_status,
        "hand_status_valid": hand_status_valid,
        "action_effector": hand_status.copy(),
        "action_effector_valid": hand_status_valid.copy(),
        **{key: value for key, value in result.items() if key != "episode_idx"},
    }
    np.savez_compressed(trajectory_path, **payload)

    position_error = np.asarray(result["position_error_m"])
    orientation_deg = np.degrees(np.asarray(result["orientation_error_rad"]))
    ik_success = np.asarray(result["ik_success"])
    target = np.asarray(result["target_eef_wxyz"])
    max_position_step, max_rotation_step = maximum_pose_steps(target)
    gripper = hand_status
    gripper_valid = hand_status_valid
    report = {
        "episode_idx": episode_index,
        "frames": int(len(rows)),
        "trajectory": str(trajectory_path),
        "position_error_m": {
            "left_mean": float(np.mean(position_error[:, 0])),
            "left_max": float(np.max(position_error[:, 0])),
            "right_mean": float(np.mean(position_error[:, 1])),
            "right_max": float(np.max(position_error[:, 1])),
        },
        "orientation_error_deg": {
            "left_mean": float(np.mean(orientation_deg[:, 0])),
            "left_max": float(np.max(orientation_deg[:, 0])),
            "right_mean": float(np.mean(orientation_deg[:, 1])),
            "right_max": float(np.max(orientation_deg[:, 1])),
        },
        "ik_success_rate": {
            "left": float(np.mean(ik_success[:, 0])),
            "right": float(np.mean(ik_success[:, 1])),
        },
        "target_step": {
            "max_position_m": max_position_step,
            "max_rotation_deg": float(np.degrees(max_rotation_step)),
        },
        "gripper": {
            "field": "action_effector",
            "source_field": (
                "generated from EEF trajectory"
                if auto_gripper is not None
                else "source_hand_status direct copy"
            ),
            "order": ["left", "right"],
            "meaning": "0=closed, 1=open",
            "mapping": "direct_copy",
            "minimum": gripper.min(axis=0).astype(float).tolist(),
            "maximum": gripper.max(axis=0).astype(float).tolist(),
            "valid_rate": gripper_valid.mean(axis=0).astype(float).tolist(),
        },
    }
    if joint_smoothing is not None:
        report["joint_smoothing"] = joint_smoothing
    if gripper_adjustment is not None:
        report["gripper"] = gripper_adjustment["gripper"]
        report["gripper_adjustment"] = gripper_adjustment
    report_path = output_dir / f"episode_{episode_index:06d}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    human_root = args.human_root.expanduser().resolve()
    robot_root = args.robot_root.expanduser().resolve()
    output_dir = refuse_protected_dataset_write(
        args.output_dir,
        purpose="write G1 retarget output",
    )
    mapping_output = (
        refuse_protected_dataset_write(
            args.mapping_output,
            purpose="write fixed-SPINE3 mapping",
        )
        if args.mapping_output is not None
        else output_dir / "fixed_spine3_to_g1_mapping.json"
    )

    human = load_human_dataset(human_root)
    reference_row = locate_dataset_row(
        human, args.reference_episode_idx, args.reference_frame_idx
    )
    reference_source_index = int(human["source_frame_index"][reference_row])
    reference_timestamp = int(human["local_timestamps_ns"][reference_row])
    requested_indices = set(human["source_frame_index"].tolist())
    if "spine3_world_xyzw" not in human:
        raise FixedSpine3RetargetError(
            "split human dataset has no required spine3_world_xyzw field; "
            "regenerate and split it with the updated conversion pipeline"
        )
    spine_poses = embedded_spine3_world_poses(human)
    missing_spine_poses = sorted(requested_indices - set(spine_poses))
    if missing_spine_poses:
        raise FixedSpine3RetargetError(
            "SPINE3 data is missing requested source frames: "
            f"{missing_spine_poses[:10]}"
        )
    reference_pose = spine_poses[reference_source_index]
    if reference_pose.timestamp_ns != reference_timestamp:
        raise FixedSpine3RetargetError(
            "reference timestamp mismatch: "
            f"LeRobot={reference_timestamp}, SPINE3 source={reference_pose.timestamp_ns}"
        )
    for row, (source_index, timestamp) in enumerate(
        zip(human["source_frame_index"], human["local_timestamps_ns"], strict=True)
    ):
        if spine_poses[int(source_index)].timestamp_ns != int(timestamp):
            raise FixedSpine3RetargetError(
                f"timestamp mismatch at dataset row {row}, source frame {source_index}"
            )
    reference = FixedSpine3Reference(
        episode_index=args.reference_episode_idx,
        frame_index=args.reference_frame_idx,
        source_frame_index=reference_source_index,
        timestamp_ns=reference_timestamp,
        position_world=reference_pose.position_world.copy(),
        rotation_world=reference_pose.rotation_world,
    )
    fixed_human_eef = rebase_all_human_poses(human, spine_poses, reference)

    # S0 <- S0 must be exactly identity; this catches axis or transform-order
    # mistakes before statistical mapping or IK hides them.
    reference_error = float(
        np.max(np.abs(fixed_human_eef[reference_row] - human["action_eef"][reference_row]))
    )
    # Source action_eef is float32 while the rebased quaternion is normalized
    # in float64, so an otherwise exact identity composition can differ by a
    # few 1e-8 in quaternion components.
    if reference_error > 1e-6:
        raise FixedSpine3RetargetError(
            f"reference-frame closure error is too large: {reference_error:.3e}"
        )

    robot_eef = load_action_eef(robot_root)
    mapping = build_mapping(
        fixed_human_eef,
        robot_eef,
        reference,
        human_root,
        robot_root,
        args.low_quantile,
        args.high_quantile,
    )
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    mapping_output.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ik = G1ArmIK(args.urdf.expanduser().resolve())
    posture, fixed_left_joint, fixed_left_eef = load_robot_reference(
        robot_root,
        ik.lower,
        ik.upper,
        args.fixed_left_reference_index,
    )
    source_for_ik = {
        "action_eef": fixed_human_eef,
        "episode_index": human["episode_index"],
        "frame_index": human["frame_index"],
    }
    available = sorted(np.unique(human["episode_index"]).astype(int).tolist())
    episodes = available if args.all_episodes else [args.episode_idx]

    print("Fixed-SPINE3 human-to-G1 retarget")
    print(f"  human dataset:       {human_root}")
    print("  SPINE3 pose source:  split dataset field spine3_world_xyzw")
    print(
        "  fixed SPINE3:        "
        f"episode={reference.episode_index}, frame={reference.frame_index}, "
        f"source={reference.source_frame_index}"
    )
    print(f"  reference closure:   {reference_error:.3e}")
    print(f"  mapping:             {mapping_output}")
    print(f"  output:              {output_dir}")
    if args.disable_auto_gripper:
        print("  gripper:             source hand_status direct copy")
    else:
        print(
            "  gripper:             automatic grasp/hold/return, "
            f"object width={args.gripper_object_width_m:.3f} m"
        )
    if args.joint_smooth_window:
        print(
            "  joint smoothing:     "
            f"window={args.joint_smooth_window}, "
            f"polyorder={args.joint_smooth_polyorder}, "
            f"passes={args.joint_smooth_passes}, endpoints preserved"
        )
    else:
        print("  joint smoothing:     disabled")

    started = time.monotonic()
    reports = []
    for episode_index in episodes:
        print(f"Retargeting episode {episode_index}...", flush=True)
        rows = episode_rows(human, episode_index, args.max_frames)
        result = run_mapped_ik(
            episode_index,
            source_for_ik,
            mapping,
            ik,
            posture,
            fixed_left_joint,
            fixed_left_eef,
            args,
        )
        joint_smoothing = smooth_result_joints(result, args)
        if joint_smoothing is not None:
            before = joint_smoothing["metrics_before"]["action_joint_position"]
            after = joint_smoothing["metrics_after"]["action_joint_position"]
            print(
                "  joint smoothing: "
                f"max step {before['max_step_rad']:.6f} -> "
                f"{after['max_step_rad']:.6f} rad; "
                f"RMS {before['rms_step_rad']:.6f} -> "
                f"{after['rms_step_rad']:.6f} rad"
            )
        auto_gripper, gripper_adjustment = generate_auto_gripper(result, args)
        if gripper_adjustment is not None:
            boundaries = gripper_adjustment["boundaries"]
            grasp = gripper_adjustment["grasp_target"]
            print(
                "  gripper: "
                f"close {boundaries['close_start_frame']}.."
                f"{boundaries['close_end_frame']}, hold to "
                f"{boundaries['open_start_frame']}, open to "
                f"{boundaries['open_end_frame']}, command={grasp['command']:.6f}"
            )
        report = save_episode(
            result,
            human,
            fixed_human_eef,
            rows,
            output_dir,
            joint_smoothing,
            auto_gripper,
            gripper_adjustment,
        )
        reports.append(report)
        print(
            f"  frames={report['frames']} "
            f"left_pos_mean={report['position_error_m']['left_mean']:.4f} m "
            f"right_pos_mean={report['position_error_m']['right_mean']:.4f} m "
            f"success=({report['ik_success_rate']['left']:.1%}, "
            f"{report['ik_success_rate']['right']:.1%})",
            flush=True,
        )

    summary = {
        "schema": "fixed_spine3_to_g1_retarget.v1",
        "human_dataset": str(human_root),
        "robot_reference_dataset": str(robot_root),
        "mapping": str(mapping_output),
        "urdf": str(args.urdf.expanduser().resolve()),
        "reference_frame": mapping["reference_frame"],
        "reference_closure_max_abs": reference_error,
        "left_mode": args.left_mode,
        "fixed_left_reference_index": args.fixed_left_reference_index,
        "joint_smoothing": {
            "enabled": bool(args.joint_smooth_window),
            "field": "action_joint_position",
            "method": "endpoint_anchored_savgol",
            "window_length": args.joint_smooth_window,
            "polyorder": args.joint_smooth_polyorder,
            "passes": args.joint_smooth_passes,
            "endpoints_preserved": bool(args.joint_smooth_window),
            "eef_recomputed_after_joint_smoothing": False,
        },
        "automatic_gripper": {
            "enabled": not args.disable_auto_gripper,
            "method": "placement_then_return_triggered_width_calibrated_gripper",
            "pose_field": args.gripper_pose_field,
            "smooth_window": args.gripper_smooth_window,
            "minimum_prominence_m": args.gripper_minimum_prominence_m,
            "minimum_distance_frames": args.gripper_minimum_distance_frames,
            "return_confirmation_frames": args.gripper_return_confirmation_frames,
            "close_ramp_frames": args.gripper_close_ramp_frames,
            "open_ramp_frames": args.gripper_open_ramp_frames,
            "object_width_m": args.gripper_object_width_m,
            "grasp_compression_m": args.gripper_grasp_compression_m,
            "gripper_min_width_m": args.gripper_min_width_m,
            "gripper_max_width_m": args.gripper_max_width_m,
            "grasp_command_override": args.gripper_grasp_command,
        },
        "pose_fields": {
            "source_action_eef_spine3_xyzw": "moving per-frame SPINE3",
            "human_eef_fixed_spine3_xyzw": "episode-0/frame-0 fixed SPINE3",
            "target_eef_wxyz": "mapped target in arm_base_link",
            "achieved_eef_wxyz": "IK FK result in arm_base_link",
        },
        "gripper_fields": {
            "source_hand_status": "original PICO fingertip-distance openness",
            "source_hand_status_valid": "original PICO hand-tracking validity",
            "hand_status": "generated robot gripper trajectory when automatic gripper is enabled",
            "action_effector": "copy of hand_status used as robot gripper command",
            "hand_status_valid": "true for generated commands; source validity when disabled",
            "action_effector_valid": "copy of hand_status_valid",
        },
        "joint_order": list(ARM_JOINTS),
        "episodes": reports,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "retarget_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        FileNotFoundError,
        FixedSpine3RetargetError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
