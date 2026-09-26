#!/usr/bin/env python3
"""Split one LeRobot v3.0 episode with the LeRobot 0.4.x dataset API.

Ranges use source-episode-local indices and the closed interval convention
``[start_frame_idx, end_frame_idx]``.  The output is rebuilt with the LeRobot
API so Parquet files, videos, indices, tasks, statistics, and metadata remain
consistent.
预先检查
python split_human_lerobot_episode.py \
  --src-root /path/to/lerobot_session_v3 \
  --output-root /path/to/lerobot_session_v3_split \
  --ranges-file /path/to/frame_range.json \
  --repo-id lerobot_session_v3_split \
  --min-frames 30 \
  --dry-run
正式转换
python split_human_lerobot_episode.py \
  --src-root /path/to/lerobot_session_v3 \
  --output-root /path/to/lerobot_session_v3_split \
  --ranges-file /path/to/frame_range.json \
  --repo-id lerobot_session_v3_split \
  --min-frames 30
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.video_utils import decode_video_frames
import numpy as np
import torch

from pipeline_safety import refuse_protected_dataset_write

DEFAULT_RESET_DELTA_FIELDS = ("action_delta_eef", "obs_delta_eef")


class SplitError(RuntimeError):
    """Raised when ranges, source data, or generated data are invalid."""


@dataclass(frozen=True)
class SegmentRange:
    output_episode_index: int
    name: str
    start_frame_idx: int
    end_frame_idx: int

    @property
    def length(self) -> int:
        return self.end_frame_idx - self.start_frame_idx + 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a LeRobot episode using inclusive source-local frame ranges [start_frame_idx, end_frame_idx]"
        )
    )
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ranges-file", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Output dataset repo id (default: <source-directory-name>_split)",
    )
    parser.add_argument(
        "--source-repo-id",
        default=None,
        help="Source repo id used by LeRobot (default: source directory name)",
    )
    parser.add_argument(
        "--source-episode",
        type=int,
        default=None,
        help="Source episode index; overrides a value in an object-form ranges JSON",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Minimum output episode length (inclusive end means end-start+1)",
    )
    parser.add_argument(
        "--short-segment-policy",
        choices=("error", "skip"),
        default="error",
    )
    parser.add_argument(
        "--image-mode",
        choices=("preserve", "video", "image"),
        default="preserve",
    )
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument(
        "--reset-delta-field",
        action="append",
        dest="reset_delta_fields",
        help=(
            "Field whose first row is reset to zero; repeat for multiple fields. "
            "Defaults to action_delta_eef and obs_delta_eef when present."
        ),
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument(
        "--video-backend",
        choices=("torchcodec", "pyav", "video_reader"),
        default=None,
        help="Source video decoder (default: LeRobot safe default)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Debug limit after validation; 0 processes all ranges",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.min_frames < 1:
        parser.error("--min-frames must be at least 1")
    if args.source_episode is not None and args.source_episode < 0:
        parser.error("--source-episode must be non-negative")
    if args.max_segments < 0:
        parser.error("--max-segments must be non-negative")
    if args.image_writer_threads < 0 or args.image_writer_processes < 0:
        parser.error("image writer counts must be non-negative")
    if args.repo_id is not None and not args.repo_id.strip():
        parser.error("--repo-id cannot be empty")
    return args


def resolve_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.src_root.expanduser().resolve()
    output = refuse_protected_dataset_write(
        args.output_root,
        purpose="write split LeRobot data",
    )
    if not source.is_dir():
        raise SplitError(f"Source LeRobot root does not exist: {source}")
    if output == source or source in output.parents or output in source.parents:
        raise SplitError("Source and output roots must be separate sibling trees; neither may contain the other")
    return source, output


def load_ranges_payload(path: Path) -> tuple[list[dict], int | None]:
    ranges_path = path.expanduser().resolve()
    if not ranges_path.is_file():
        raise SplitError(f"Ranges file does not exist: {ranges_path}")
    try:
        with ranges_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"Cannot read ranges JSON {ranges_path}: {exc}") from exc

    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise SplitError("Object-form ranges JSON must contain a 'segments' list")
        source_episode = payload.get("source_episode_index")
        if source_episode is not None and (isinstance(source_episode, bool) or not isinstance(source_episode, int)):
            raise SplitError("source_episode_index must be an integer")
        return segments, source_episode
    raise SplitError("Ranges JSON must be either a list or an object containing 'segments'")


def parse_and_validate_ranges(
    raw_ranges: list[dict],
    *,
    episode_length: int,
    min_frames: int,
    short_policy: str,
    allow_overlap: bool,
) -> tuple[list[SegmentRange], list[dict]]:
    accepted: list[SegmentRange] = []
    skipped: list[dict] = []
    previous_start: int | None = None
    previous_end: int | None = None

    for input_index, raw in enumerate(raw_ranges):
        if not isinstance(raw, dict):
            raise SplitError(f"Range {input_index} must be a JSON object")
        for field in ("start_frame_idx", "end_frame_idx"):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SplitError(f"Range {input_index}: {field} must be an integer")
        start = raw["start_frame_idx"]
        end = raw["end_frame_idx"]
        name = raw.get("name", f"segment_{input_index:03d}")
        if not isinstance(name, str) or not name.strip():
            raise SplitError(f"Range {input_index}: name must be a non-empty string")
        if not 0 <= start <= end < episode_length:
            raise SplitError(
                f"Range {input_index} [{start}, {end}] is outside source episode [0, {episode_length - 1}]"
            )
        if previous_start is not None and start < previous_start:
            raise SplitError(f"Range {input_index} starts at {start}, before the previous start {previous_start}")
        if not allow_overlap and previous_end is not None and start <= previous_end:
            raise SplitError(
                f"Range {input_index} [{start}, {end}] overlaps the previous range ending at {previous_end}"
            )

        length = end - start + 1
        if length < min_frames:
            record = {
                "input_range_index": input_index,
                "name": name,
                "start_frame_idx": start,
                "end_frame_idx": end,
                "frames": length,
                "reason": f"shorter_than_min_frames_{min_frames}",
            }
            if short_policy == "error":
                raise SplitError(
                    f"Range {input_index} [{start}, {end}] has {length} frames, less than --min-frames {min_frames}"
                )
            skipped.append(record)
        else:
            accepted.append(
                SegmentRange(
                    output_episode_index=len(accepted),
                    name=name,
                    start_frame_idx=start,
                    end_frame_idx=end,
                )
            )
        previous_start = start
        previous_end = end

    if not accepted:
        raise SplitError("No valid ranges remain")
    return accepted, skipped


def get_episode_record(dataset: LeRobotDataset, episode_index: int) -> dict:
    if episode_index < 0 or episode_index >= dataset.meta.total_episodes:
        raise SplitError(
            f"Source episode {episode_index} does not exist; dataset has {dataset.meta.total_episodes} episodes"
        )
    if dataset.meta.episodes is None:
        raise SplitError("LeRobot v3 episode metadata is unavailable")
    return dataset.meta.episodes[episode_index]


def get_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    episode = get_episode_record(dataset, episode_index)
    start = int(episode["dataset_from_index"])
    end_exclusive = int(episode["dataset_to_index"])
    return start, end_exclusive


def build_output_features(
    source_features: dict,
    image_mode: str,
) -> tuple[dict, list[str], list[str]]:
    custom_features = {
        key: deepcopy(feature) for key, feature in source_features.items() if key not in DEFAULT_FEATURES
    }
    image_keys = [key for key, feature in custom_features.items() if feature["dtype"] in {"image", "video"}]
    if not image_keys:
        raise SplitError("Source dataset has no image/video features")

    if image_mode != "preserve":
        for key in image_keys:
            custom_features[key]["dtype"] = image_mode
            custom_features[key].pop("info", None)

    output_video_keys = [key for key in image_keys if custom_features[key]["dtype"] == "video"]
    return custom_features, image_keys, output_video_keys


def validate_reset_fields(features: dict, requested: list[str] | None) -> list[str]:
    names = requested if requested is not None else list(DEFAULT_RESET_DELTA_FIELDS)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SplitError(f"Duplicate --reset-delta-field values: {duplicates}")
    present = []
    for name in names:
        if name not in features:
            if requested is not None:
                raise SplitError(f"Requested delta reset field does not exist: {name}")
            continue
        feature = features[name]
        if feature["dtype"] in {"image", "video", "string"}:
            raise SplitError(f"Delta reset field must be numeric: {name}")
        present.append(name)
    return present


def numpy_frame_value(value: object, feature: dict, context: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.dtype(feature["dtype"]))
    expected_shape = tuple(feature["shape"])
    if array.shape != expected_shape:
        try:
            array = array.reshape(expected_shape)
        except ValueError as exc:
            raise SplitError(f"{context}: cannot reshape {array.shape} to {expected_shape}") from exc
    return array


def load_segment_images(
    source: LeRobotDataset,
    *,
    source_episode: int,
    global_start: int,
    local_start: int,
    length: int,
    image_keys: list[str],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    episode = get_episode_record(source, source_episode)
    for key in image_keys:
        source_dtype = source.meta.features[key]["dtype"]
        if source_dtype == "video":
            video_path = source.root / source.meta.get_video_file_path(source_episode, key)
            video_start = float(episode[f"videos/{key}/from_timestamp"])
            timestamps = [
                video_start + (local_start + offset) / source.fps for offset in range(length)
            ]
            frames = decode_video_frames(
                video_path,
                timestamps,
                source.tolerance_s,
                source.video_backend,
            )
            if tuple(frames.shape[:1]) != (length,):
                raise SplitError(f"Decoded {tuple(frames.shape)} for {key}, expected {length} frames")
            output[key] = frames.cpu().numpy()
        else:
            images = []
            for offset in range(length):
                item = source[global_start + offset]
                image = item[key]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                images.append(np.asarray(image))
            output[key] = np.stack(images)
    return output


def create_segment_frames(
    source: LeRobotDataset,
    output: LeRobotDataset,
    *,
    source_episode: int,
    source_episode_global_start: int,
    segment: SegmentRange,
    custom_features: dict,
    image_keys: list[str],
    reset_delta_fields: list[str],
) -> None:
    global_start = source_episode_global_start + segment.start_frame_idx
    images = load_segment_images(
        source,
        source_episode=source_episode,
        global_start=global_start,
        local_start=segment.start_frame_idx,
        length=segment.length,
        image_keys=image_keys,
    )

    for offset in range(segment.length):
        source_item = source.hf_dataset[global_start + offset]
        task_index = int(source_item["task_index"].item())
        if source.meta.tasks is None or task_index >= len(source.meta.tasks):
            raise SplitError(f"Invalid task_index {task_index} at source row {global_start + offset}")
        task = str(source.meta.tasks.iloc[task_index].name)
        frame: dict[str, object] = {"task": task}
        for key, feature in custom_features.items():
            if key in image_keys:
                frame[key] = images[key][offset]
                continue
            value = numpy_frame_value(
                source_item[key],
                feature,
                context=f"{key} at source frame {segment.start_frame_idx + offset}",
            )
            if offset == 0 and key in reset_delta_fields:
                value = np.zeros(tuple(feature["shape"]), dtype=np.dtype(feature["dtype"]))
            frame[key] = value
        output.add_frame(frame)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def validate_output(
    root: Path,
    *,
    repo_id: str,
    segments: list[SegmentRange],
    custom_features: dict,
    image_keys: list[str],
    output_video_keys: list[str],
    reset_delta_fields: list[str],
) -> dict:
    expected_frames = sum(segment.length for segment in segments)
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    format_version = str(metadata.info.get("codebase_version", ""))
    if format_version != "v3.0":
        raise SplitError(f"Output dataset format is {format_version!r}, expected 'v3.0'")
    if metadata.total_episodes != len(segments):
        raise SplitError(f"Output has {metadata.total_episodes} episodes, expected {len(segments)}")
    if metadata.total_frames != expected_frames:
        raise SplitError(f"Output has {metadata.total_frames} frames, expected {expected_frames}")
    if metadata.fps <= 0:
        raise SplitError(f"Output has invalid FPS: {metadata.fps}")
    missing = sorted(set(custom_features) - set(metadata.features))
    if missing:
        raise SplitError(f"Output is missing features: {missing}")

    data_files = sorted((root / "data").rglob("*.parquet"))
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    task_files = sorted((root / "meta").glob("tasks.parquet"))
    video_files = sorted(root.rglob("*.mp4"))
    image_files = sorted((root / "images").rglob("*.png"))
    if not data_files or not episode_files or len(task_files) != 1:
        raise SplitError(
            "Output is missing LeRobot v3 metadata/data files: "
            f"data={len(data_files)}, episodes={len(episode_files)}, tasks={len(task_files)}"
        )
    # LeRobot v3 embeds dtype=image values in Parquet and deletes the temporary
    # PNG staging files after each episode is saved.
    if image_files:
        raise SplitError(f"Output retained {len(image_files)} unexpected temporary image files")

    referenced_videos: set[Path] = set()
    for episode_index in range(len(segments)):
        for key in output_video_keys:
            video_path = root / metadata.get_video_file_path(episode_index, key)
            if not video_path.is_file():
                raise SplitError(f"Output video is missing: {video_path}")
            referenced_videos.add(video_path.resolve())
    if {path.resolve() for path in video_files} != referenced_videos:
        raise SplitError(
            f"Output has {len(video_files)} video files, but metadata references {len(referenced_videos)}"
        )

    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    for episode_index, segment in enumerate(segments):
        start, end = get_episode_bounds(dataset, episode_index)
        if end - start != segment.length:
            raise SplitError(f"Output episode {episode_index} has {end - start} frames, expected {segment.length}")
        first = dataset.hf_dataset[start]
        last = dataset.hf_dataset[end - 1]
        if int(first["frame_index"].item()) != 0:
            raise SplitError(f"Output episode {episode_index} frame_index does not start at 0")
        if int(last["frame_index"].item()) != segment.length - 1:
            raise SplitError(f"Output episode {episode_index} has invalid final frame_index")
        if abs(float(first["timestamp"].item())) > 1e-6:
            raise SplitError(f"Output episode {episode_index} timestamp does not start at 0")
        for key in reset_delta_fields:
            if not np.allclose(first[key].numpy(), 0, atol=1e-7):
                raise SplitError(f"Output episode {episode_index} first {key} is not zero")

    # Decode one frame from the first and last episode to catch broken video/image paths.
    for index in sorted({0, expected_frames - 1}):
        item = dataset[index]
        for key in image_keys:
            expected_shape = tuple(metadata.features[key]["shape"])
            if tuple(item[key].shape) != expected_shape:
                raise SplitError(
                    f"Decoded {key} at output row {index} has {tuple(item[key].shape)}, expected {expected_shape}"
                )

    return {
        "format_version": format_version,
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "fps": metadata.fps,
        "data_files": len(data_files),
        "episode_files": len(episode_files),
        "task_files": len(task_files),
        "video_files": len(video_files),
        "image_files": len(image_files),
        "features": sorted(custom_features),
        "sample_image_decode": "passed",
        "first_delta_zero_check": "passed",
    }


def print_preflight(
    *,
    source: Path,
    output: Path,
    source_episode: int,
    source_episode_length: int,
    segments: list[SegmentRange],
    skipped: list[dict],
    fps: int,
    image_keys: list[str],
    image_mode: str,
    reset_fields: list[str],
) -> None:
    lengths = np.asarray([segment.length for segment in segments])
    print("LeRobot episode split preflight")
    print(f"  source root:          {source}")
    print(f"  output root:          {output}")
    print(f"  source episode:       {source_episode}")
    print(f"  source episode frames:{source_episode_length}")
    print("  interval convention:  [start_frame_idx, end_frame_idx] (inclusive)")
    print(f"  output episodes:      {len(segments)}")
    print(f"  output frames:        {int(lengths.sum())}")
    print(f"  episode length range: {int(lengths.min())}..{int(lengths.max())}")
    print(f"  skipped ranges:       {len(skipped)}")
    print(f"  fps:                  {fps}")
    print(f"  image keys:           {image_keys}")
    print(f"  image mode:           {image_mode}")
    print(f"  reset first-row delta:{reset_fields}")


def convert(args: argparse.Namespace) -> Path | None:
    source_root, output_root = resolve_roots(args)
    source_repo_id = args.source_repo_id or source_root.name
    output_repo_id = args.repo_id or f"{source_root.name}_split"
    source = LeRobotDataset(
        repo_id=source_repo_id,
        root=source_root,
        video_backend=args.video_backend,
    )
    if str(source.meta._version) != "3.0" or CODEBASE_VERSION != "v3.0":
        raise SplitError(
            f"This script requires LeRobot dataset v3.0; source={source.meta._version}, "
            f"installed API={CODEBASE_VERSION}"
        )

    raw_ranges, json_source_episode = load_ranges_payload(args.ranges_file)
    source_episode = (
        args.source_episode
        if args.source_episode is not None
        else (json_source_episode if json_source_episode is not None else 0)
    )
    global_start, global_end = get_episode_bounds(source, source_episode)
    source_episode_length = global_end - global_start
    segments, skipped = parse_and_validate_ranges(
        raw_ranges,
        episode_length=source_episode_length,
        min_frames=args.min_frames,
        short_policy=args.short_segment_policy,
        allow_overlap=args.allow_overlap,
    )
    if args.max_segments:
        segments = segments[: args.max_segments]
        segments = [
            SegmentRange(
                output_episode_index=index,
                name=segment.name,
                start_frame_idx=segment.start_frame_idx,
                end_frame_idx=segment.end_frame_idx,
            )
            for index, segment in enumerate(segments)
        ]

    custom_features, image_keys, output_video_keys = build_output_features(
        source.meta.features,
        args.image_mode,
    )
    reset_fields = validate_reset_fields(custom_features, args.reset_delta_fields)
    effective_image_mode = (
        args.image_mode
        if args.image_mode != "preserve"
        else ",".join(sorted({source.meta.features[key]["dtype"] for key in image_keys}))
    )
    print_preflight(
        source=source_root,
        output=output_root,
        source_episode=source_episode,
        source_episode_length=source_episode_length,
        segments=segments,
        skipped=skipped,
        fps=source.fps,
        image_keys=image_keys,
        image_mode=effective_image_mode,
        reset_fields=reset_fields,
    )
    if args.dry_run:
        print("Dry run completed; no files were written.")
        return None

    if output_root.exists() and not args.overwrite:
        raise SplitError(f"Output already exists: {output_root}; use --overwrite to replace it")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.tmp")
    robot_type = source.meta.info.get("robot_type", "unknown")

    output_dataset = None
    try:
        output_dataset = LeRobotDataset.create(
            repo_id=output_repo_id,
            root=temporary,
            fps=source.fps,
            robot_type=robot_type,
            features=custom_features,
            use_videos=bool(output_video_keys),
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
        )
        for segment in segments:
            print(
                f"  writing episode {segment.output_episode_index}: "
                f"source [{segment.start_frame_idx}, {segment.end_frame_idx}] "
                f"({segment.length} frames)",
                flush=True,
            )
            create_segment_frames(
                source,
                output_dataset,
                source_episode=source_episode,
                source_episode_global_start=global_start,
                segment=segment,
                custom_features=custom_features,
                image_keys=image_keys,
                reset_delta_fields=reset_fields,
            )
            output_dataset.save_episode()
        output_dataset.stop_image_writer()
        output_dataset.finalize()
        output_dataset = None

        validation = validate_output(
            temporary,
            repo_id=output_repo_id,
            segments=segments,
            custom_features=custom_features,
            image_keys=image_keys,
            output_video_keys=output_video_keys,
            reset_delta_fields=reset_fields,
        )
        report = {
            "schema": "lerobot_episode_split.v1",
            "source_dataset": str(source_root),
            "source_repo_id": source_repo_id,
            "output_repo_id": output_repo_id,
            "ranges_file": str(args.ranges_file.expanduser().resolve()),
            "source_episode_index": source_episode,
            "source_episode_frames": source_episode_length,
            "interval_convention": "[start_frame_idx, end_frame_idx] inclusive",
            "length_formula": "end_frame_idx - start_frame_idx + 1",
            "image_mode": effective_image_mode,
            "image_keys": image_keys,
            "delta_fields_reset_on_first_frame": reset_fields,
            "skipped_ranges": skipped,
            "episodes": [asdict(segment) | {"frames": segment.length} for segment in segments],
            "validation": validation,
        }
        write_json_atomic(temporary / "meta" / "split_report.json", report)

        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        print(f"Completed: {output_root}")
        return output_root
    except Exception:
        if output_dataset is not None:
            output_dataset.stop_image_writer()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    convert(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SplitError, OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
