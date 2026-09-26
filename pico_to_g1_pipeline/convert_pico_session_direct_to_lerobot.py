#!/usr/bin/env python3
"""Convert one raw PicoEgoRecorder session directly to a LeRobot dataset.

This is a standalone file: it does not import any EgoHumanoid project module.
Copy this one script to another computer and install the documented external
dependencies. Tracking JSONL is parsed in memory; the PICO video is decoded as
a stream. Navigation and ZED data are not used.

action_eef is relative to SPINE3 (body joint index 9) and uses the same fixed
axis and hand-frame corrections as process_vr_manipulation_pipeline.py. Its
order is left/right [x, y, z, qx, qy, qz, qw].

hand_status is continuous hand openness in left/right order. 0.0 means the
thumb and index fingertip are at the configured closed distance; 1.0 means
they are at the configured open distance. hand_status_valid reports whether
the corresponding PICO hand pose was tracked on that frame.

Typical usage:

    # System dependency (choose one)
    sudo apt-get install ffmpeg       # Ubuntu/Debian
    brew install ffmpeg               # macOS
    winget install Gyan.FFmpeg        # Windows PowerShell

    # Python 3.10+ environment. LeRobot 0.4.x writes dataset format v3.0.
    python3 -m venv .venv
    .venv/bin/pip install \
      numpy scipy av "lerobot>=0.4,<0.5"

    # This check does not need a session directory.
    .venv/bin/python convert_pico_session_direct_to_lerobot.py --check-dependencies

The input directory must contain body_tracking.jsonl, hands.jsonl,
controllers.jsonl, camera.mp4 and video_metadata.json. Convert it with::

    ./.venv/bin/python \
      data_alignment/human_data_process/convert_pico_session_direct_to_lerobot.py \
      /path/to/session_20260723_084253_461 \
      --output-path /path/to/lerobot_session_20260723_084253_461_spine3 \
      --repo-id pico_vr_manipulation_spine3 \
      --task "dual-arm manipulation" \
      --image-mode video

Keep the original tracking rate and disable EEF smoothing:

    ./.venv/bin/python \
      data_alignment/human_data_process/convert_pico_session_direct_to_lerobot.py \
      /path/to/session_ID \
      --output-path /path/to/lerobot_session_ID \
      --downsample-factor 1 \
      --sg-window 0

Short smoke test:

    ./.venv/bin/python \
      data_alignment/human_data_process/convert_pico_session_direct_to_lerobot.py \
      /path/to/session_ID \
      --output-path /tmp/lerobot_smoke_test \
      --max-frames 20 \
      --overwrite

Use --help for all options. image-mode video stores each eye as an MP4 in the
LeRobot dataset; image uses LeRobot image storage.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any
import uuid

from pipeline_safety import refuse_protected_dataset_write

SPINE3_INDEX = 9
SPINE3_NAME = "SPINE3"
LEFT_HAND_INDEX = 22
RIGHT_HAND_INDEX = 23
REQUIRED_FILES = (
    "body_tracking.jsonl",
    "hands.jsonl",
    "controllers.jsonl",
    "camera.mp4",
    "video_metadata.json",
)

BODY_JOINT_NAMES = (
    "Pelvis", "LEFT_HIP", "RIGHT_HIP", "SPINE1", "LEFT_KNEE", "RIGHT_KNEE",
    "SPINE2", "LEFT_ANKLE", "RIGHT_ANKLE", "SPINE3", "LEFT_FOOT", "RIGHT_FOOT",
    "NECK", "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HAND", "RIGHT_HAND",
)
EEF_NAMES = [
    "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw",
    "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw",
]
SPINE3_WORLD_NAMES = [
    "x", "y", "z", "qx", "qy", "qz", "qw",
]
DELTA_EEF_NAMES = [
    "left_dx", "left_dy", "left_dz", "left_roll", "left_pitch", "left_yaw",
    "right_dx", "right_dy", "right_dz", "right_roll", "right_pitch", "right_yaw",
]
THUMB_TIP_INDEX = 5
INDEX_TIP_INDEX = 10
PIP_DEPENDENCIES = (
    "numpy",
    "scipy",
    "av",
    "lerobot>=0.4,<0.5",
)


def pip_install_command() -> str:
    """Return an install command that uses the interpreter running this file."""
    executable = f'"{sys.executable}"' if " " in sys.executable else sys.executable
    dependencies = " ".join(f'"{item}"' for item in PIP_DEPENDENCIES)
    return f"{executable} -m pip install {dependencies}"


def load_python_dependencies() -> None:
    """Import optional runtime packages after CLI parsing so --help always works."""
    global av, np, savgol_filter, Rotation  # noqa: PLW0603
    global LeRobotDataset, LeRobotDatasetMetadata  # noqa: PLW0603

    missing: list[str] = []
    modules: dict[str, Any] = {}
    for package, module_name in (("numpy", "numpy"), ("scipy", "scipy"), ("av", "av"), ("lerobot", "lerobot")):
        try:
            modules[package] = importlib.import_module(module_name)
        except ImportError:
            missing.append(package)
    if missing:
        raise DirectConversionError(
            f"Missing Python dependencies: {', '.join(missing)}\n"
            f"Install them with:\n  {pip_install_command()}"
        )

    av = modules["av"]
    np = modules["numpy"]
    try:
        savgol_filter = importlib.import_module("scipy.signal").savgol_filter
        Rotation = importlib.import_module("scipy.spatial.transform").Rotation
        dataset_module = importlib.import_module("lerobot.datasets.lerobot_dataset")
        LeRobotDataset = dataset_module.LeRobotDataset
        LeRobotDatasetMetadata = dataset_module.LeRobotDatasetMetadata
        if dataset_module.CODEBASE_VERSION != "v3.0":
            raise DirectConversionError(
                "This converter requires the LeRobot v3.0 dataset API; "
                f"the installed package uses {dataset_module.CODEBASE_VERSION}."
            )
    except (AttributeError, ImportError) as exc:
        raise DirectConversionError(
            "Installed packages do not expose the LeRobot v3 dataset APIs. Install compatible dependencies with:\n"
            f"  {pip_install_command()}\nOriginal error: {exc}"
        ) from exc


def command_path(value: str) -> str:
    """Resolve an executable name or a user-provided path without assuming an OS."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    resolved = shutil.which(expanded)
    if resolved:
        return resolved
    candidate = Path(expanded)
    if candidate.is_file():
        return str(candidate.resolve())
    raise DirectConversionError(
        f"Cannot find executable: {value!r}. Install FFmpeg or pass --ffprobe /path/to/ffprobe."
    )


def ffmpeg_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return "winget install Gyan.FFmpeg"
    if system == "Darwin":
        return "brew install ffmpeg"
    return "sudo apt-get install ffmpeg"


def portable_path(value: str) -> Path:
    """Accept ~ and environment variables on all supported platforms."""
    return Path(os.path.expandvars(os.path.expanduser(value)))


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    frame_times_s: np.ndarray
    average_frame_rate: float


@dataclass(frozen=True)
class PipelineConfig:
    downsample_factor: int
    sg_window: int
    sg_poly: int
    sg_passes: int
    hand_closed_distance_m: float
    hand_open_distance_m: float
    hand_status_smoothing_window: int
    stereo_order: str
    video_start_offset_ms: float
    max_output_frames: int


class DirectConversionError(RuntimeError):
    """Raised when a raw session cannot be converted safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw PICO stereo manipulation session directly to LeRobot"
    )
    parser.add_argument("session_dir", type=portable_path, nargs="?", help="Raw PicoEgoRecorder session directory")
    parser.add_argument(
        "--output-path",
        type=portable_path,
        help="LeRobot dataset root (default: <session_dir>_lerobot next to the source)",
    )
    parser.add_argument("--check-dependencies", action="store_true", help="Check packages and ffprobe, then exit")
    parser.add_argument(
        "--print-install-command",
        action="store_true",
        help="Print the Python dependency installation command, then exit",
    )
    parser.add_argument(
        "--ffprobe",
        default=os.environ.get("FFPROBE", "ffprobe"),
        help="ffprobe executable name/path (default: FFPROBE environment variable or ffprobe)",
    )
    parser.add_argument("--repo-id", default="pico_vr_manipulation_spine3")
    parser.add_argument("--task", default="dual-arm manipulation")
    parser.add_argument("--downsample-factor", type=int, default=5)
    parser.add_argument("--sg-window", type=int, default=51)
    parser.add_argument("--sg-poly", type=int, default=2)
    parser.add_argument("--sg-passes", type=int, default=2)
    parser.add_argument(
        "--hand-closed-distance-m",
        type=float,
        default=0.02,
        help="Thumb/index fingertip distance mapped to hand_status=0 (default: 0.02 m)",
    )
    parser.add_argument(
        "--hand-open-distance-m",
        type=float,
        default=0.14,
        help="Thumb/index fingertip distance mapped to hand_status=1 (default: 0.14 m)",
    )
    parser.add_argument(
        "--hand-status-smoothing-window",
        type=int,
        default=5,
        help="Odd median-filter window for continuous hand_status; 0 or 1 disables it (default: 5)",
    )
    parser.add_argument(
        "--inactive-hand-policy",
        choices=("hold", "identity"),
        default="hold",
        help="Pose used when PICO hand tracking is inactive",
    )
    parser.add_argument(
        "--stereo-order", choices=("left-right", "right-left"), default="left-right"
    )
    parser.add_argument(
        "--video-start-offset-ms",
        type=float,
        default=0.0,
        help="Correction added to video_metadata.json mp4_started_timestamp_ns",
    )
    parser.add_argument("--fps", type=int, default=0, help="LeRobot FPS; 0 infers it")
    parser.add_argument("--image-mode", choices=("video", "image"), default="video")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Debug limit after overlap/downsampling"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.downsample_factor <= 0:
        parser.error("--downsample-factor must be positive")
    if args.sg_window < 0:
        parser.error("--sg-window must be non-negative")
    if args.sg_poly < 1 or args.sg_passes < 1:
        parser.error("--sg-poly and --sg-passes must be positive")
    if args.max_frames < 0 or args.fps < 0:
        parser.error("frame/FPS values must be non-negative")
    if not 0 <= args.hand_closed_distance_m < args.hand_open_distance_m:
        parser.error("hand distances must satisfy 0 <= closed < open")
    if args.hand_status_smoothing_window < 0:
        parser.error("--hand-status-smoothing-window must be non-negative")
    if args.hand_status_smoothing_window > 1 and args.hand_status_smoothing_window % 2 == 0:
        parser.error("--hand-status-smoothing-window must be odd, 0, or 1")
    if args.image_writer_threads < 0 or args.image_writer_processes < 0:
        parser.error("image writer counts must be non-negative")
    if not args.repo_id.strip() or not args.task.strip():
        parser.error("--repo-id and --task cannot be empty")
    informational = args.check_dependencies or args.print_install_command
    if not informational and args.session_dir is None:
        parser.error("session_dir is required")
    if args.session_dir is not None and args.output_path is None:
        args.output_path = args.session_dir.with_name(f"{args.session_dir.name}_lerobot")
    return args


def load_json(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise DirectConversionError(f"Missing required file: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectConversionError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DirectConversionError(f"Expected a JSON object in {path}")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise DirectConversionError(f"Blank record in {path} at line {line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DirectConversionError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise DirectConversionError(f"Expected object in {path} at line {line_number}")
                yield value
    except OSError as exc:
        raise DirectConversionError(f"Cannot read {path}: {exc}") from exc


def finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectConversionError(f"{context} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise DirectConversionError(f"{context} is not finite: {result!r}")
    return result


def pose7(value: dict[str, Any], context: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise DirectConversionError(f"{context} must be an object")
    position = value.get("position")
    rotation = value.get("rotation")
    if not isinstance(position, dict) or not isinstance(rotation, dict):
        raise DirectConversionError(f"{context} must contain position and rotation objects")
    result = np.asarray(
        [finite_float(position.get(axis), f"{context}.position.{axis}") for axis in "xyz"]
        + [finite_float(rotation.get(axis), f"{context}.rotation.{axis}") for axis in "xyzw"],
        dtype=np.float32,
    )
    if float(np.linalg.norm(result[3:])) < 1e-6:
        raise DirectConversionError(f"{context} contains a zero-length quaternion")
    return result


def identity_poses(count: int) -> np.ndarray:
    poses = np.zeros((count, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    return poses


def indexed_items(items: Any, count: int, index_key: str, context: str) -> list[dict[str, Any]]:
    if not isinstance(items, list) or len(items) != count:
        actual = len(items) if isinstance(items, list) else type(items).__name__
        raise DirectConversionError(f"{context} must contain {count} items; got {actual}")
    ordered: list[dict[str, Any] | None] = [None] * count
    for item in items:
        if not isinstance(item, dict):
            raise DirectConversionError(f"{context} contains a non-object item")
        index = item.get(index_key)
        if not isinstance(index, int) or not 0 <= index < count:
            raise DirectConversionError(f"Invalid {context} index: {index!r}")
        if ordered[index] is not None:
            raise DirectConversionError(f"Duplicate {context} index: {index}")
        ordered[index] = item
    if any(item is None for item in ordered):
        raise DirectConversionError(f"{context} has missing indices")
    return [item for item in ordered if item is not None]


def parse_body(record: dict[str, Any], line_number: int) -> np.ndarray:
    context = f"body_tracking.jsonl line {line_number}"
    if not record.get("is_tracking", False):
        raise DirectConversionError(f"{context}: body tracking is inactive")
    if not record.get("pose_valid", False) or not record.get("numeric_valid", False):
        raise DirectConversionError(f"{context}: body pose is invalid")
    joints = indexed_items(record.get("joints"), 24, "index", f"{context}.joints")
    result = np.empty((24, 7), dtype=np.float32)
    for index, (joint, expected_role) in enumerate(zip(joints, BODY_JOINT_NAMES, strict=True)):
        if joint.get("role") != expected_role:
            raise DirectConversionError(f"{context}: unexpected role for joint {index}")
        result[index] = pose7(joint.get("local_pose"), f"{context}.joints[{index}].local_pose")
    return result


def parse_controller(record: dict[str, Any], side: str, line_number: int) -> np.ndarray:
    value = record.get(side)
    if not isinstance(value, dict):
        raise DirectConversionError(f"controllers.jsonl line {line_number}.{side} must be an object")
    return pose7(value, f"controllers.jsonl line {line_number}.{side}")


def parse_hand(record: dict[str, Any], side: str, line_number: int,
               previous_valid: np.ndarray | None, inactive_policy: str
               ) -> tuple[np.ndarray, bool, np.ndarray | None]:
    context = f"hands.jsonl line {line_number}.{side}"
    hand = record.get(side)
    if not isinstance(hand, dict):
        raise DirectConversionError(f"{context} must be an object")
    active = bool(hand.get("tracking_available", False) and hand.get("is_active", False)
                  and hand.get("sample_valid", False))
    if not active:
        if inactive_policy == "hold" and previous_valid is not None:
            return previous_valid.copy(), False, previous_valid
        return identity_poses(26), False, previous_valid
    joints = indexed_items(hand.get("joints"), 26, "joint", f"{context}.joints")
    result = np.empty((26, 7), dtype=np.float32)
    for index, joint in enumerate(joints):
        result[index] = pose7(joint, f"{context}.joints[{index}]")
    return result, True, result


def record_timestamp(record: dict[str, Any], context: str, key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DirectConversionError(f"{context}.{key} must be an integer")
    return value


def validate_source(session_dir: Path) -> None:
    if not session_dir.is_dir():
        raise DirectConversionError(f"Session directory does not exist: {session_dir}")
    missing = [name for name in REQUIRED_FILES if not (session_dir / name).is_file()]
    if missing:
        raise DirectConversionError(f"Session is missing required files: {missing}")


def load_tracking_arrays(
    session_dir: Path,
    inactive_hand_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    values: dict[str, list[Any]] = {
        "body_pose": [],
        "left_hand_pose": [],
        "right_hand_pose": [],
        "left_hand_active": [],
        "right_hand_active": [],
        "local_timestamps_ns": [],
    }
    previous_hands: dict[str, np.ndarray | None] = {"left": None, "right": None}
    inactive_counts = {"left": 0, "right": 0}
    previous_timestamp: int | None = None

    streams: Iterable[tuple[Any, Any, Any]] = itertools.zip_longest(
        iter_jsonl(session_dir / "body_tracking.jsonl"),
        iter_jsonl(session_dir / "hands.jsonl"),
        iter_jsonl(session_dir / "controllers.jsonl"),
        fillvalue=None,
    )
    for line_number, (body, hands, controllers) in enumerate(streams, start=1):
        if body is None or hands is None or controllers is None:
            raise DirectConversionError(f"JSONL stream lengths differ at line {line_number}")

        realtime = []
        monotonic = []
        for record, filename in (
            (body, "body_tracking.jsonl"),
            (hands, "hands.jsonl"),
            (controllers, "controllers.jsonl"),
        ):
            context = f"{filename} line {line_number}"
            realtime.append(record_timestamp(record, context, "timestamp_ns"))
            monotonic.append(
                record_timestamp(record, context, "monotonic_timestamp_ns")
            )
        if len(set(realtime)) != 1 or len(set(monotonic)) != 1:
            raise DirectConversionError(
                f"Source streams are not synchronized at line {line_number}: {realtime}"
            )
        timestamp = realtime[0]
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise DirectConversionError(f"Non-increasing timestamp at line {line_number}")
        previous_timestamp = timestamp

        values["local_timestamps_ns"].append(timestamp)
        values["body_pose"].append(parse_body(body, line_number))
        for side in ("left", "right"):
            parse_controller(controllers, side, line_number)
            pose, active, previous_hands[side] = parse_hand(
                hands,
                side,
                line_number,
                previous_hands[side],
                inactive_hand_policy,
            )
            values[f"{side}_hand_pose"].append(pose)
            values[f"{side}_hand_active"].append(active)
            if not active:
                inactive_counts[side] += 1
        if line_number % 2000 == 0:
            print(f"  parsed {line_number} tracking frames...", flush=True)

    if len(values["body_pose"]) < 2:
        raise DirectConversionError("Session needs at least two tracking records")
    arrays = {
        "body_pose": np.asarray(values["body_pose"], dtype=np.float32),
        "left_hand_pose": np.asarray(values["left_hand_pose"], dtype=np.float32),
        "right_hand_pose": np.asarray(values["right_hand_pose"], dtype=np.float32),
        "left_hand_active": np.asarray(values["left_hand_active"], dtype=np.bool_),
        "right_hand_active": np.asarray(values["right_hand_active"], dtype=np.bool_),
        "local_timestamps_ns": np.asarray(values["local_timestamps_ns"], dtype=np.int64),
    }
    return arrays, inactive_counts


def parse_fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, separator, denominator = value.partition("/")
    if separator:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def probe_video(video_path: Path, ffprobe: str) -> VideoProbe:
    command = [
        command_path(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "stream=width,height,avg_frame_rate:packet=pts_time",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError as exc:
        raise DirectConversionError(
            f"ffprobe is required. Install it with: {ffmpeg_install_hint()}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DirectConversionError(f"ffprobe failed for {video_path}: {exc.stderr.strip()}") from exc
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        frame_times = np.sort(np.asarray(
            [float(packet["pts_time"]) for packet in payload["packets"] if "pts_time" in packet],
            dtype=np.float64,
        ))
        width = int(stream["width"])
        height = int(stream["height"])
        average_frame_rate = parse_fraction(stream.get("avg_frame_rate"))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DirectConversionError(f"Cannot parse ffprobe output for {video_path}") from exc
    if len(frame_times) == 0 or np.any(np.diff(frame_times) <= 0):
        raise DirectConversionError(f"Video PTS are empty or not increasing: {video_path}")
    if width <= 0 or height <= 0 or width % 2:
        raise DirectConversionError(f"Expected an even-width side-by-side video, got {width}x{height}")
    return VideoProbe(width, height, frame_times, average_frame_rate)


def nearest_indices(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    indices = np.clip(np.searchsorted(sorted_values, queries), 0, len(sorted_values) - 1)
    previous = np.maximum(indices - 1, 0)
    return np.where(
        np.abs(sorted_values[indices] - queries) < np.abs(sorted_values[previous] - queries),
        indices,
        previous,
    ).astype(np.int64)


def build_timeline(source_timestamps_ns: np.ndarray, camera_timestamps_ns: np.ndarray,
                   config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap = np.flatnonzero(
        (source_timestamps_ns >= camera_timestamps_ns[0])
        & (source_timestamps_ns <= camera_timestamps_ns[-1])
    )
    if len(overlap) < 2:
        raise DirectConversionError("Tracking and video timelines have no usable overlap")
    source_indices = np.arange(
        int(overlap[0]), int(overlap[-1]) + 1, config.downsample_factor, dtype=np.int64
    )
    if config.max_output_frames:
        source_indices = source_indices[:config.max_output_frames]
    if len(source_indices) < 2:
        raise DirectConversionError("Target timeline needs at least two frames")
    target_timestamps = source_timestamps_ns[source_indices]
    return source_indices, target_timestamps, nearest_indices(camera_timestamps_ns, target_timestamps)


def validate_pose_array(poses: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(poses)):
        raise DirectConversionError(f"{name} contains non-finite values")
    if np.any(np.linalg.norm(poses[..., 3:7], axis=-1) < 1e-6):
        raise DirectConversionError(f"{name} contains zero-length quaternions")


def make_savgol_params(frame_count: int, window: int, poly: int) -> tuple[int, int]:
    maximum = frame_count if frame_count % 2 else frame_count - 1
    valid_window = max(3, min(window, maximum))
    if valid_window % 2 == 0:
        valid_window -= 1
    return valid_window, min(max(1, poly), valid_window - 1)


def smooth_quaternions(quaternions: np.ndarray, window: int, poly: int) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0:
            quaternions[index] *= -1
    reference = Rotation.from_quat(quaternions[len(quaternions) // 2])
    rotation_vectors = (reference.inv() * Rotation.from_quat(quaternions)).as_rotvec()
    for axis in range(3):
        rotation_vectors[:, axis] = savgol_filter(rotation_vectors[:, axis], window, poly, mode="nearest")
    return (reference * Rotation.from_rotvec(rotation_vectors)).as_quat()


def smooth_poses(poses: np.ndarray, window: int, poly: int, passes: int) -> np.ndarray:
    if window <= 0 or len(poses) < 3:
        return poses
    valid_window, valid_poly = make_savgol_params(len(poses), window, poly)
    result = np.asarray(poses, dtype=np.float64).copy()
    for _ in range(passes):
        for axis in range(3):
            result[:, axis] = savgol_filter(result[:, axis], valid_window, valid_poly, mode="nearest")
        result[:, 3:7] = smooth_quaternions(result[:, 3:7], valid_window, valid_poly)
    return result


def compute_eef(body_pose: np.ndarray, sg_window: int, sg_poly: int,
                sg_passes: int) -> tuple[np.ndarray, np.ndarray]:
    validate_pose_array(body_pose, "body_pose")
    base = np.asarray(body_pose[:, SPINE3_INDEX], dtype=np.float64)
    base_rotation = Rotation.from_quat(base[:, 3:7])
    fixed_rotation = Rotation.from_euler("z", -90, degrees=True) * Rotation.from_euler("x", 90, degrees=True)
    local_x_180 = Rotation.from_euler("x", 180, degrees=True)
    left_local_z_180 = Rotation.from_euler("z", 180, degrees=True)
    outputs = []
    for hand_index, is_left in ((LEFT_HAND_INDEX, True), (RIGHT_HAND_INDEX, False)):
        hand = np.asarray(body_pose[:, hand_index], dtype=np.float64)
        relative_position = base_rotation.inv().apply(hand[:, :3] - base[:, :3])
        relative_rotation = Rotation.from_matrix(np.einsum(
            "nji,njk->nik", base_rotation.as_matrix(), Rotation.from_quat(hand[:, 3:7]).as_matrix()
        ))
        rotation = fixed_rotation * relative_rotation * local_x_180
        if is_left:
            rotation = rotation * left_local_z_180
        poses = np.column_stack((fixed_rotation.apply(relative_position), rotation.as_quat()))
        outputs.append(smooth_poses(poses, sg_window, sg_poly, sg_passes))
    return outputs[0], outputs[1]


def compute_delta_eef(left_eef: np.ndarray, right_eef: np.ndarray) -> np.ndarray:
    output = np.zeros((len(left_eef), 12), dtype=np.float64)
    for offset, poses in ((0, left_eef), (6, right_eef)):
        if len(poses) < 2:
            continue
        previous = Rotation.from_quat(poses[:-1, 3:7])
        current = Rotation.from_quat(poses[1:, 3:7])
        output[1:, offset:offset + 3] = previous.inv().apply(poses[1:, :3] - poses[:-1, :3])
        output[1:, offset + 3:offset + 6] = (previous.inv() * current).as_euler("xyz")
    return output


def thumb_index_tip_distance(hand_pose: np.ndarray) -> np.ndarray:
    """Return the thumb-tip to index-tip Euclidean distance in meters."""
    thumb_tip = np.asarray(hand_pose[:, THUMB_TIP_INDEX, :3], dtype=np.float64)
    index_tip = np.asarray(hand_pose[:, INDEX_TIP_INDEX, :3], dtype=np.float64)
    distance = np.linalg.norm(thumb_tip - index_tip, axis=1)
    if not np.all(np.isfinite(distance)):
        raise DirectConversionError("Thumb/index fingertip distance contains non-finite values")
    return distance


def hold_invalid_samples(values: np.ndarray, valid: np.ndarray, side: str) -> np.ndarray:
    """Forward-fill invalid tracking and back-fill an invalid leading prefix."""
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise DirectConversionError(f"No valid {side} hand-tracking samples are available")
    result = np.asarray(values, dtype=np.float64).copy()
    first_valid = int(valid_indices[0])
    result[:first_valid] = result[first_valid]
    for index in range(first_valid + 1, len(result)):
        if not valid[index]:
            result[index] = result[index - 1]
    return result


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 2:
        return np.asarray(values, dtype=np.float64)
    effective = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if effective <= 1:
        return np.asarray(values, dtype=np.float64)
    radius = effective // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, effective)
    return np.median(windows, axis=-1)


def distance_to_openness(distance_m: np.ndarray, closed_m: float, open_m: float) -> np.ndarray:
    """Map physical fingertip distance to 0=closed and 1=open."""
    return np.clip((distance_m - closed_m) / (open_m - closed_m), 0.0, 1.0)


def compute_hand_status(left_pose: np.ndarray, right_pose: np.ndarray,
                        left_active: np.ndarray, right_active: np.ndarray,
                        config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    statuses = []
    for side, pose, valid in (
        ("left", left_pose, left_active),
        ("right", right_pose, right_active),
    ):
        distance = thumb_index_tip_distance(pose)
        distance = hold_invalid_samples(distance, valid, side)
        openness = distance_to_openness(
            distance,
            config.hand_closed_distance_m,
            config.hand_open_distance_m,
        )
        openness = median_smooth(openness, config.hand_status_smoothing_window)
        statuses.append(np.asarray(openness, dtype=np.float32))
    return np.column_stack(statuses), np.column_stack((left_active, right_active))


def infer_fps(timestamps_ns: np.ndarray, requested_fps: int) -> int:
    if requested_fps:
        return requested_fps
    fps = round(1e9 / float(np.median(np.diff(timestamps_ns))))
    if fps <= 0:
        raise DirectConversionError(f"Invalid inferred FPS: {fps}")
    return fps


def vector_feature(dtype: str, names: list[str]) -> dict[str, Any]:
    return {"dtype": dtype, "shape": (len(names),), "names": names}


def build_features(height: int, width: int, image_mode: str) -> dict[str, Any]:
    image_feature = {
        "dtype": "video" if image_mode == "video" else "image",
        "shape": (3, height, width),
        "names": ["channel", "height", "width"],
    }
    return {
        "observation.images.left": dict(image_feature),
        "observation.images.right": dict(image_feature),
        "action_eef": vector_feature("float32", EEF_NAMES),
        "action_delta_eef": vector_feature("float32", DELTA_EEF_NAMES),
        # Keep the moving SPINE3 parent pose with each sample so downstream
        # fixed-SPINE3 retargeting is self-contained after episode splitting.
        "spine3_world_xyzw": vector_feature("float32", SPINE3_WORLD_NAMES),
        "hand_status": vector_feature("float32", ["left", "right"]),
        "hand_status_valid": vector_feature("bool", ["left", "right"]),
        "local_timestamps_ns": vector_feature("int64", ["timestamp_ns"]),
        "camera_timestamp": vector_feature("int64", ["timestamp_ns"]),
        "timestamp_diff_ms": vector_feature("float32", ["difference_ms"]),
        "source_frame_index": vector_feature("int64", ["source_frame_index"]),
        "camera_frame_index": vector_feature("int64", ["camera_frame_index"]),
    }


def validate_lerobot_output(root: Path, repo_id: str, expected_frames: int,
                            expected_features: dict[str, Any]) -> dict[str, Any]:
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    format_version = str(metadata.info.get("codebase_version", ""))
    if format_version != "v3.0":
        raise DirectConversionError(
            f"LeRobot output format is {format_version!r}; expected 'v3.0'"
        )
    if metadata.total_episodes != 1 or metadata.total_frames != expected_frames:
        raise DirectConversionError(
            f"LeRobot output is {metadata.total_episodes} episodes/{metadata.total_frames} frames; "
            f"expected 1/{expected_frames}"
        )
    missing = sorted(set(expected_features) - set(metadata.features))
    if missing:
        raise DirectConversionError(f"LeRobot output is missing features: {missing}")
    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    task_files = sorted((root / "meta").glob("tasks.parquet"))
    video_files = sorted(root.rglob("*.mp4"))
    image_files = sorted((root / "images").rglob("*.png"))
    expected_videos = 2 if metadata.video_keys else 0
    expected_images = 0 if metadata.video_keys else expected_frames * 2
    if (
        len(data_files) != 1
        or len(episode_files) != 1
        or len(task_files) != 1
        or len(video_files) != expected_videos
        or len(image_files) != expected_images
    ):
        raise DirectConversionError(
            "Unexpected LeRobot v3 output files: "
            f"data={len(data_files)}, episodes={len(episode_files)}, tasks={len(task_files)}, "
            f"video={len(video_files)}, images={len(image_files)}"
        )
    return {
        "format_version": format_version,
        "episodes": metadata.total_episodes, "frames": metadata.total_frames,
        "fps": metadata.fps, "features": sorted(expected_features),
        "data_files": len(data_files), "episode_files": len(episode_files),
        "task_files": len(task_files), "video_files": len(video_files),
        "image_files": len(image_files),
    }


def write_conversion_report(root: Path, report: dict[str, Any]) -> None:
    path = root / "meta" / "conversion_report.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def check_dependencies(ffprobe: str) -> None:
    executable = command_path(ffprobe)
    try:
        result = subprocess.run(
            [executable, "-version"], check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DirectConversionError(
            f"ffprobe is unavailable. Install it with: {ffmpeg_install_hint()}"
        ) from exc
    print("Python dependencies: OK")
    print(result.stdout.splitlines()[0] if result.stdout else "ffprobe available")
    print(f"ffprobe path: {executable}")
    print("Standalone project imports: none")


def add_stereo_frames(
    dataset: LeRobotDataset,
    video_path: Path,
    probe: VideoProbe,
    camera_indices: np.ndarray,
    frame_data: dict[str, np.ndarray],
    task: str,
    stereo_order: str,
) -> int:
    output_positions: dict[int, list[int]] = {}
    for output_index, camera_index in enumerate(camera_indices.tolist()):
        output_positions.setdefault(int(camera_index), []).append(output_index)
    maximum_wanted = max(output_positions)
    single_width = probe.width // 2
    written = 0

    try:
        with av.open(str(video_path)) as container:
            video_stream = container.streams.video[0]
            for decoded_index, frame in enumerate(container.decode(video_stream)):
                if decoded_index > maximum_wanted:
                    break
                positions = output_positions.get(decoded_index)
                if positions is None:
                    continue
                frame_time = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else math.nan
                )
                expected_time = float(probe.frame_times_s[decoded_index])
                if not math.isfinite(frame_time) or abs(frame_time - expected_time) > 1e-3:
                    raise DirectConversionError(
                        f"Decoded/ffprobe PTS mismatch at frame {decoded_index}: "
                        f"{frame_time} vs {expected_time}"
                    )
                image_bgr = frame.to_ndarray(format="bgr24")
                if image_bgr.shape[:2] != (probe.height, probe.width):
                    raise DirectConversionError(
                        f"Video frame {decoded_index} has unexpected shape {image_bgr.shape}"
                    )
                first = image_bgr[:, :single_width]
                second = image_bgr[:, single_width:]
                if stereo_order == "left-right":
                    left_bgr, right_bgr = first, second
                else:
                    right_bgr, left_bgr = first, second
                # PyAV yields BGR; reversing the final axis avoids an OpenCV dependency.
                left_rgb = left_bgr[..., ::-1].copy()
                right_rgb = right_bgr[..., ::-1].copy()

                for output_index in positions:
                    scalar = lambda name, dtype: np.asarray(  # noqa: E731
                        frame_data[name][output_index], dtype=dtype
                    ).reshape(1)
                    dataset.add_frame(
                        {
                            "observation.images.left": left_rgb,
                            "observation.images.right": right_rgb,
                            "action_eef": frame_data["action_eef"][output_index],
                            "action_delta_eef": frame_data["action_delta_eef"][output_index],
                            "spine3_world_xyzw": frame_data["spine3_world_xyzw"][output_index],
                            "hand_status": frame_data["hand_status"][output_index],
                            "hand_status_valid": frame_data["hand_status_valid"][output_index],
                            "local_timestamps_ns": scalar("local_timestamps_ns", np.int64),
                            "camera_timestamp": scalar("camera_timestamp", np.int64),
                            "timestamp_diff_ms": scalar("timestamp_diff_ms", np.float32),
                            "source_frame_index": scalar("source_frame_index", np.int64),
                            "camera_frame_index": scalar("camera_frame_index", np.int64),
                            "task": task,
                        }
                    )
                    written += 1
                    if written % 500 == 0 or written == len(camera_indices):
                        print(f"  wrote {written}/{len(camera_indices)} frames", flush=True)
    except av.error.FFmpegError as exc:
        raise DirectConversionError(f"Cannot decode {video_path}: {exc}") from exc

    if written != len(camera_indices):
        raise DirectConversionError(f"Wrote {written} stereo pairs, expected {len(camera_indices)}")
    return written


def convert(args: argparse.Namespace) -> Path:
    session_dir = args.session_dir.expanduser().resolve()
    output = refuse_protected_dataset_write(
        args.output_path,
        purpose="write converted LeRobot data",
    )
    validate_source(session_dir)
    if output.exists() and not args.overwrite:
        raise DirectConversionError(f"Output exists: {output}; use --overwrite to replace it")

    manifest = load_json(session_dir / "manifest.json")
    video_metadata = load_json(
        session_dir / "video_metadata.json", required=True
    )
    video_start_ns = video_metadata.get("mp4_started_timestamp_ns")
    if isinstance(video_start_ns, bool) or not isinstance(video_start_ns, int):
        raise DirectConversionError(
            "video_metadata.json has no integer mp4_started_timestamp_ns"
        )
    video_start_ns += round(args.video_start_offset_ms * 1e6)

    print("Raw PICO session -> LeRobot (no intermediate HDF5)")
    print(f"  source:   {session_dir}")
    print(f"  output:   {output}")
    print(f"  EEF base: {SPINE3_NAME} (joint {SPINE3_INDEX})")
    print(
        "  hand:     continuous openness 0=closed/1=open "
        f"({args.hand_closed_distance_m:g}..{args.hand_open_distance_m:g} m)"
    )
    arrays, inactive_counts = load_tracking_arrays(
        session_dir, args.inactive_hand_policy
    )
    expected_counts = manifest.get("record_counts", {})
    if isinstance(expected_counts, dict):
        expected = expected_counts.get("body_tracking")
        if isinstance(expected, int) and expected != len(arrays["body_pose"]):
            raise DirectConversionError(
                f"manifest expects {expected} body frames, got {len(arrays['body_pose'])}"
            )

    probe = probe_video(session_dir / "camera.mp4", args.ffprobe)
    camera_timestamps = video_start_ns + np.rint(
        (probe.frame_times_s - probe.frame_times_s[0]) * 1e9
    ).astype(np.int64)
    config = PipelineConfig(
        downsample_factor=args.downsample_factor,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        sg_passes=args.sg_passes,
        hand_closed_distance_m=args.hand_closed_distance_m,
        hand_open_distance_m=args.hand_open_distance_m,
        hand_status_smoothing_window=args.hand_status_smoothing_window,
        stereo_order=args.stereo_order,
        video_start_offset_ms=args.video_start_offset_ms,
        max_output_frames=args.max_frames,
    )
    source_indices, target_timestamps, camera_indices = build_timeline(
        arrays["local_timestamps_ns"], camera_timestamps, config
    )
    left_eef, right_eef = compute_eef(
        arrays["body_pose"], args.sg_window, args.sg_poly, args.sg_passes
    )
    hand_status, hand_status_valid = compute_hand_status(
        arrays["left_hand_pose"],
        arrays["right_hand_pose"],
        arrays["left_hand_active"],
        arrays["right_hand_active"],
        config,
    )
    selected_left = left_eef[source_indices]
    selected_right = right_eef[source_indices]
    matched_camera_timestamps = camera_timestamps[camera_indices]
    frame_data = {
        "action_eef": np.hstack((selected_left, selected_right)).astype(np.float32),
        "action_delta_eef": compute_delta_eef(
            selected_left, selected_right
        ).astype(np.float32),
        "spine3_world_xyzw": arrays["body_pose"][
            source_indices, SPINE3_INDEX
        ].astype(np.float32),
        "hand_status": hand_status[source_indices].astype(np.float32),
        "hand_status_valid": hand_status_valid[source_indices].astype(np.bool_),
        "local_timestamps_ns": target_timestamps.astype(np.int64),
        "camera_timestamp": matched_camera_timestamps.astype(np.int64),
        "timestamp_diff_ms": (
            np.abs(matched_camera_timestamps - target_timestamps) / 1e6
        ).astype(np.float32),
        "source_frame_index": source_indices.astype(np.int64),
        "camera_frame_index": camera_indices.astype(np.int64),
    }
    fps = infer_fps(target_timestamps, args.fps)
    features = build_features(
        probe.height, probe.width // 2, args.image_mode
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=temporary,
            fps=fps,
            robot_type="pico_vr_dual_arm",
            features=features,
            use_videos=args.image_mode == "video",
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
        )
        written = add_stereo_frames(
            dataset,
            session_dir / "camera.mp4",
            probe,
            camera_indices,
            frame_data,
            args.task,
            args.stereo_order,
        )
        dataset.save_episode()
        dataset.stop_image_writer()
        dataset.finalize()
        dataset = None

        validation = validate_lerobot_output(
            temporary,
            args.repo_id,
            expected_frames=written,
            expected_features=features,
        )
        report = {
            "schema": "pico_raw_session_direct_to_lerobot.v1",
            "source_session": str(session_dir),
            "source_session_id": manifest.get("session_id", session_dir.name),
            "source_coordinate_frame": "pico_unity_world_hmd_aligned",
            "repo_id": args.repo_id,
            "task": args.task,
            "intermediate_hdf5_created": False,
            "eef_reference_joint": SPINE3_NAME,
            "eef_reference_joint_index": SPINE3_INDEX,
            "eef_pose_format": "left/right xyz_qxqyqzqw",
            "spine3_world_pose_field": "spine3_world_xyzw",
            "inactive_hand_policy": args.inactive_hand_policy,
            "inactive_hand_frames": inactive_counts,
            "hand_status": {
                "meaning": "continuous hand openness: 0=closed, 1=open",
                "method": "thumb_index_fingertip_distance",
                "thumb_tip_joint_index": THUMB_TIP_INDEX,
                "index_tip_joint_index": INDEX_TIP_INDEX,
                "closed_distance_m": args.hand_closed_distance_m,
                "open_distance_m": args.hand_open_distance_m,
                "smoothing": "median",
                "smoothing_window": args.hand_status_smoothing_window,
                "invalid_sample_policy": "hold previous; leading prefix uses first valid sample",
            },
            "downsample_factor": args.downsample_factor,
            "smoothing": {
                "sg_window": args.sg_window,
                "sg_poly": args.sg_poly,
                "sg_passes": args.sg_passes,
            },
            "stereo_order": args.stereo_order,
            "video_start_offset_ms": args.video_start_offset_ms,
            "source_tracking_frames": len(arrays["local_timestamps_ns"]),
            "source_video_frames": len(probe.frame_times_s),
            "validation": validation,
        }
        write_conversion_report(temporary, report)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        print(f"Completed: {output}")
        return output
    except Exception:
        if dataset is not None:
            dataset.stop_image_writer()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_install_command:
        print(pip_install_command())
        return 0
    if sys.version_info < (3, 10):
        raise DirectConversionError(
            f"Python 3.10 or newer is required; found {platform.python_version()}"
        )
    load_python_dependencies()
    if args.check_dependencies:
        check_dependencies(args.ffprobe)
        return 0
    convert(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        DirectConversionError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
