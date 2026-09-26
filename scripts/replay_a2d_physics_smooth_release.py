#!/usr/bin/env python3
"""Experimentally replay one episode with stationary gradual close and release.

This creates a new derived episode and never edits the source trajectory.  The
arms pause while the selected gripper closes, resume the recorded transport,
pause again at release, and then linearly open over ``--open-duration-s``.
The existing ``recorded`` and ``fast`` release implementations are unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_a2d_grasp_hold import prepare as prepare_grasp_hold  # noqa: E402
from scripts.replay_a2d_physics import DEFAULT_MANIFEST, load_manifest  # noqa: E402


def insert_smooth_release(
    arrays: dict[str, np.ndarray],
    *,
    open_start: int,
    open_end: int,
    side: int,
    duration_s: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Replace the recorded opening interval with a stationary linear ramp."""

    times = np.asarray(arrays["local_timestamps_ns"], dtype=np.int64)
    effectors = np.asarray(arrays["action_effector"])
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("open-duration-s must be positive and finite")
    if not 0 <= open_start < open_end < len(times):
        raise ValueError("Invalid recorded opening interval")
    if side not in (0, 1) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid gripper side or non-increasing timestamps")

    closed = float(effectors[open_start, side])
    opened = float(effectors[open_end, side])
    if opened - closed < 0.1:
        raise ValueError("Selected interval must contain substantial gripper opening")

    duration_ns = int(round(duration_s * 1e9))
    sample_ns = int(round(float(np.median(np.diff(times)))))
    inserted = max(1, int(np.ceil(duration_ns / sample_ns)))
    offsets = np.rint(np.linspace(0, duration_ns, inserted + 1)).astype(np.int64)[1:]
    if np.any(np.diff(np.r_[0, offsets]) <= 0):
        raise ValueError("Duration is too short for nanosecond timestamps")

    # Keep the first closed sample, replace the original ramp, then continue
    # after its fully-open endpoint. Repeating open_start freezes arms and all
    # other recorded state during the synthetic release.
    indices = np.r_[
        np.arange(open_start + 1),
        np.full(inserted, open_start),
        np.arange(open_end + 1, len(times)),
    ]
    result = {
        key: value[indices].copy()
        if value.ndim and value.shape[0] == len(times)
        else value.copy()
        for key, value in arrays.items()
    }
    shift_ns = duration_ns - int(times[open_end] - times[open_start])
    result["local_timestamps_ns"] = np.r_[
        times[: open_start + 1],
        times[open_start] + offsets,
        times[open_end + 1 :] + shift_ns,
    ]
    result["episode_frame_index"] = np.arange(len(indices), dtype=np.int64)
    result["synthetic_release_frame"] = np.zeros(len(indices), dtype=bool)
    result["synthetic_release_frame"][open_start + 1 : open_start + inserted + 1] = True

    ramp = np.linspace(closed, opened, inserted + 1, dtype=np.float32)[1:]
    result["action_effector"][open_start + 1 : open_start + inserted + 1, side] = ramp
    # Keep the standard gripper fields consistent for downstream collection.
    if "hand_status" in result:
        result["hand_status"] = result["action_effector"].copy()
    return result, indices


def prepare_smooth_release(
    manifest_path: Path,
    episode: str,
    output_dir: Path,
    *,
    close_duration_s: float,
    open_duration_s: float,
) -> Path:
    """Create a closure-prepared copy whose recorded opening is replaced."""

    prepared_manifest = prepare_grasp_hold(
        manifest_path,
        episode,
        output_dir,
        duration_s=close_duration_s,
    )
    document = load_manifest(prepared_manifest)
    record = document["episodes"][0]
    episode_path = Path(document["dataset_dir"]) / episode
    with np.load(episode_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}

    side = int(np.argmax(np.ptp(arrays["action_effector"], axis=0)))
    boundaries = record["boundaries"]
    result, indices = insert_smooth_release(
        arrays,
        open_start=int(boundaries["open_start_frame"]),
        open_end=int(boundaries["open_end_frame"]),
        side=side,
        duration_s=open_duration_s,
    )
    np.savez_compressed(episode_path, **result)

    cache_value = record.get("cache")
    if cache_value:
        cache_path = (prepared_manifest.parent / cache_value).resolve()
        with np.load(cache_path, allow_pickle=False) as archive:
            joints = np.asarray(archive["joint_positions"])
        if joints.shape[0] != len(arrays["local_timestamps_ns"]):
            raise ValueError("Corrected joint cache does not match prepared episode")
        np.savez_compressed(cache_path, joint_positions=joints[indices])

    inserted = len(indices) - len(arrays["local_timestamps_ns"])
    open_start = int(boundaries["open_start_frame"])
    record["frames"] = len(indices)
    boundaries["open_end_frame"] = open_start + int(np.count_nonzero(
        result["synthetic_release_frame"]
    ))
    record["dice"]["release_frame"] = open_start + 1
    processing = document.setdefault("processing", {})
    processing.update(
        type="stationary_gripper_closure_and_release",
        open_duration_s=open_duration_s,
        release_inserted_frames=int(np.count_nonzero(result["synthetic_release_frame"])),
        release_net_frame_change=inserted,
        release_mode="recorded_interpolated",
    )
    prepared_manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return prepared_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode", default="episode_000000.npz")
    parser.add_argument("--close-duration-s", type=float, default=2.0)
    parser.add_argument("--open-duration-s", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--exit-when-finished", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.speed <= 0 or not np.isfinite(args.speed):
        parser.error("--speed must be positive and finite")
    if args.output_dir is None:
        args.output_dir = Path("logs/a2d_smooth_release") / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    return args


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    prepared = prepare_smooth_release(
        args.manifest.expanduser().resolve(),
        args.episode,
        output / "data",
        close_duration_s=args.close_duration_s,
        open_duration_s=args.open_duration_s,
    )
    metrics = output / "metrics.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/replay_a2d_physics.py"),
        "--manifest", str(prepared),
        "--episode", args.episode,
        "--gripper-release-mode", "recorded",
        "--arm-contact-mode", "constrained",
        "--physics-timestep", "0.0005",
        "--contact-impratio", "100",
        "--gripper-close-bias", "0",
        "--gripper-sliding-friction", "3",
        "--dice-linear-damping", "0.02",
        "--dice-angular-damping", "0.0005",
        "--speed", str(args.speed),
        "--metrics-output", str(metrics),
    ]
    if args.headless:
        command.append("--headless")
    else:
        command.append("--start-immediately")
    if args.exit_when_finished:
        command.append("--exit-when-finished")
    print(f"Prepared smooth-release manifest: {prepared}", flush=True)
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
