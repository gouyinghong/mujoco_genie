#!/usr/bin/env python3
"""Create a separate episode that pauses the arms for gradual gripper closure.

Original arrays and corrected-joint cache are read only. Copied source/IK fields
retain their provenance; inserted frames are synthetic, not new measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_a2d_physics import (  # noqa: E402
    DEFAULT_MANIFEST, PHYSICS_LAYOUT_SCHEMA, choose_record, load_manifest, successful_records,
)


def insert_grasp_hold(arrays: dict[str, np.ndarray], *, close_start: int,
                      close_end: int, hold_frame: int, side: int,
                      duration_s: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    times = arrays["local_timestamps_ns"]
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("close-duration-s must be positive and finite")
    if not 0 <= close_start <= hold_frame <= close_end < len(times):
        raise ValueError("hold-frame must be within the recorded closing interval")
    if side not in (0, 1) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid gripper side or non-increasing timestamps")
    duration_ns = int(round(duration_s * 1e9))
    n = max(1, int(np.ceil(duration_ns / np.median(np.diff(times)))))
    offsets = np.rint(np.linspace(0, duration_ns, n + 1)).astype(np.int64)[1:]
    if np.any(np.diff(np.r_[0, offsets]) <= 0):
        raise ValueError("Duration is too short for nanosecond timestamps")
    indices = np.r_[np.arange(hold_frame + 1), np.full(n, hold_frame),
                    np.arange(hold_frame + 1, len(times))]
    result = {key: value[indices].copy() if value.ndim and value.shape[0] == len(times)
              else value.copy() for key, value in arrays.items()}
    result["local_timestamps_ns"] = np.r_[
        times[:hold_frame + 1], times[hold_frame] + offsets,
        times[hold_frame + 1:] + duration_ns,
    ]
    result["episode_frame_index"] = np.arange(len(indices), dtype=np.int64)
    result["parent_episode_frame_index"] = indices
    result["synthetic_hold_frame"] = np.zeros(len(indices), dtype=bool)
    result["synthetic_hold_frame"][hold_frame + 1:hold_frame + n + 1] = True
    effector = result["action_effector"]
    opened = arrays["action_effector"][close_start, side]
    closed = arrays["action_effector"][close_end, side]
    if opened - closed < 0.1:
        raise ValueError("Selected interval must contain substantial gripper closure")
    effector[close_start:hold_frame + 1, side] = opened
    effector[hold_frame + 1:hold_frame + n + 1, side] = np.linspace(opened, closed, n + 1)[1:]
    effector[hold_frame + n + 1:close_end + n + 1, side] = closed
    return result, indices


def prepare(manifest_path: Path, episode: str, output_dir: Path,
            duration_s: float = 2.0, hold_frame: int | None = None) -> Path:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    _, record = choose_record(successful_records(manifest), 0, episode)
    source = Path(manifest["dataset_dir"]) / episode
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    side = int(np.argmax(np.ptp(arrays["action_effector"], axis=0)))
    boundaries = record["boundaries"]
    hold_frame = int(boundaries["close_end_frame"] if hold_frame is None else hold_frame)
    result, indices = insert_grasp_hold(
        arrays, close_start=int(boundaries["close_start_frame"]),
        close_end=int(boundaries["close_end_frame"]), hold_frame=hold_frame,
        side=side, duration_s=duration_s,
    )
    inserted = len(indices) - len(arrays["local_timestamps_ns"])
    cache = None
    if record.get("cache"):
        cache_source = (manifest_path.parent / record["cache"]).resolve()
        with np.load(cache_source, allow_pickle=False) as archive:
            joints = archive["joint_positions"].copy()
        if joints.shape != arrays["action_joint_position"].shape:
            raise ValueError("Corrected joint cache shape does not match source episode")
        cache = joints[indices]
    summary_text = Path(manifest["summary"]).read_text()
    output_dir = output_dir.resolve()
    # Refuse even an existing empty directory: no source or prior output is overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / episode, **result)
    (output_dir / "retarget_summary.json").write_text(summary_text)
    derived = copy.deepcopy(record)
    if cache is not None:
        np.savez_compressed(output_dir / "corrected_joints.npz", joint_positions=cache)
        derived["cache"] = "corrected_joints.npz"
    derived["frames"] = len(indices)
    derived["dice_center_frame"] = hold_frame
    derived["boundaries"].update(
        close_start_frame=hold_frame, close_end_frame=hold_frame + inserted,
        open_start_frame=boundaries["open_start_frame"] + inserted,
        open_end_frame=boundaries["open_end_frame"] + inserted,
    )
    derived["dice"].update(grasp_frame=hold_frame + inserted,
                           release_frame=record["dice"]["release_frame"] + inserted)
    derived.pop("metrics", None)  # Source validation does not certify the edited episode.
    layout_path = output_dir / "physics_layout.json"
    layout = {"schema": PHYSICS_LAYOUT_SCHEMA, "datasets": {output_dir.name: {episode: {
        "status": "unvalidated", "score": 0,
        "dice_position": record["dice"]["initial_position"],
        "dice_yaw_deg": record["dice"]["initial_yaw_deg"],
        "gripper_control": "closed-loop",
    }}}}
    layout_path.write_text(json.dumps(layout, indent=2) + "\n")
    document = {key: copy.deepcopy(manifest[key]) for key in ("schema", "model", "fixed_torso")}
    document.update(dataset_dir=str(output_dir), summary=str(output_dir / "retarget_summary.json"),
                    physics_layout=str(layout_path), episodes=[derived],
                    counts={"total": 1, "ok": 1, "failed": 0},
                    processing={"type": "stationary_gripper_closure", "source_episode": str(source),
                                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "source_manifest": str(manifest_path), "hold_frame": hold_frame,
                                "close_duration_s": duration_s, "inserted_frames": inserted,
                                "synthetic": True, "physics_validated": False})
    output = output_dir / "manifest.json"
    output.write_text(json.dumps(document, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode", default="episode_000000.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--close-duration-s", type=float, default=2.0)
    parser.add_argument("--hold-frame", type=int,
                        help="Frame at which to pause the arms; defaults to the original close-end frame")
    args = parser.parse_args()
    print(prepare(args.manifest, args.episode, args.output_dir,
                  args.close_duration_s, args.hold_frame))


if __name__ == "__main__":
    main()
