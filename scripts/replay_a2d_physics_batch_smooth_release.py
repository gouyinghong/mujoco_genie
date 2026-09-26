#!/usr/bin/env python3
"""Batch the independent smooth-release replay with configurable acceptance."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_a2d_physics import DEFAULT_MANIFEST, load_manifest  # noqa: E402
from scripts.replay_a2d_physics_batch import select_records  # noqa: E402


SCHEMA = "a2d_physics_smooth_release_batch.v1"
FIELDS = (
    "episode", "status", "reason", "metrics_file", "log_file",
    "pick_and_place_success", "relaxed_low_slip_pick_and_place_success",
    "max_dice_translation_in_gripper_m", "max_dice_rotation_in_gripper_deg",
    "landed_in_box", "physics_warnings",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify(
    metrics: dict,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> bool:
    """Apply relaxed low-slip limits without changing core replay metrics."""

    return bool(
        metrics.get("pick_and_place_success") is True
        and metrics["max_dice_translation_in_gripper_m"] <= translation_tolerance_m
        and metrics["max_dice_rotation_in_gripper_deg"] <= rotation_tolerance_deg
    )


def save_report(output: Path, rows: list[dict], args: argparse.Namespace) -> dict:
    statuses = ("pending", "running", "passed", "failed", "error", "skipped", "interrupted")
    report = {
        "schema": SCHEMA,
        "updated_at": now(),
        "criteria": {
            "translation_tolerance_m": args.low_slip_translation_tolerance_m,
            "rotation_tolerance_deg": args.low_slip_rotation_tolerance_deg,
            "requires_pick_and_place_success": True,
        },
        "counts": {status: sum(row["status"] == status for row in rows) for status in statuses},
        "episodes": rows,
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    with (output / "summary.csv.tmp").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.csv.tmp").replace(output / "summary.csv")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", nargs="+", help="Select filenames in manifest order")
    parser.add_argument("--episode", action="append", dest="single_episodes",
                        help="Select one episode; may be repeated")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--close-duration-s", type=float, default=2.0)
    parser.add_argument("--open-duration-s", type=float, default=1.0)
    parser.add_argument("--low-slip-translation-tolerance-m", type=float, default=0.003)
    parser.add_argument("--low-slip-rotation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.episodes and args.single_episodes:
        parser.error("Use either --episodes or --episode, not both")
    args.episodes = args.episodes or args.single_episodes
    for name in ("close_duration_s", "open_duration_s", "speed",
                 "low_slip_translation_tolerance_m", "low_slip_rotation_tolerance_deg"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("start-index must be nonnegative and limit positive")
    if args.output_dir is None:
        args.output_dir = Path("logs/a2d_smooth_release_batch") / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    return args


def run(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    records = select_records(manifest, args.episodes, args.start_index, args.limit)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    (output / "logs").mkdir()
    rows = [{"episode": record["episode"], "status": "pending"} for record in records]
    save_report(output, rows, args)
    print(f"Smooth-release batch: {len(rows)} episodes; output={output}", flush=True)

    interrupted = False
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        if record.get("status") != "ok":
            row.update(status="skipped", reason=f"Source manifest status={record.get('status')}")
            save_report(output, rows, args)
            continue
        stem = Path(record["episode"]).stem
        episode_output = output / "episodes" / stem
        metrics_path = episode_output / "metrics.json"
        log_path = output / "logs" / f"{stem}.log"
        row.update(status="running", reason="", metrics_file=str(metrics_path), log_file=str(log_path))
        save_report(output, rows, args)
        print(f"[{index + 1}/{len(rows)}] {record['episode']}", flush=True)
        command = [
            sys.executable,
            str(ROOT / "scripts/replay_a2d_physics_smooth_release.py"),
            "--manifest", str(manifest_path),
            "--episode", record["episode"],
            "--close-duration-s", str(args.close_duration_s),
            "--open-duration-s", str(args.open_duration_s),
            "--speed", str(args.speed),
            "--output-dir", str(episode_output),
        ]
        command.append("--headless" if args.headless else "--exit-when-finished")
        try:
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if process.returncode in (130, -2):
                raise KeyboardInterrupt
            if process.returncode:
                raise RuntimeError(f"Replay exited with code {process.returncode}; see {log_path}")
            metrics = json.loads(metrics_path.read_text())
            passed = classify(
                metrics,
                args.low_slip_translation_tolerance_m,
                args.low_slip_rotation_tolerance_deg,
            )
            row.update({key: metrics.get(key) for key in FIELDS if key in metrics})
            row["relaxed_low_slip_pick_and_place_success"] = passed
            row["status"] = "passed" if passed else "failed"
            row["reason"] = "" if passed else "Did not pass relaxed low-slip pick-and-place criteria"
            print(
                f"  {row['status']}: displacement="
                f"{metrics['max_dice_translation_in_gripper_m'] * 1000:.2f} mm, "
                f"rotation={metrics['max_dice_rotation_in_gripper_deg']:.2f} deg",
                flush=True,
            )
        except KeyboardInterrupt:
            row.update(status="interrupted", reason="User interrupted replay")
            interrupted = True
        except Exception as error:
            row.update(status="error", reason=f"{type(error).__name__}: {error}")
            print(f"  error: {row['reason']}", flush=True)
        finally:
            save_report(output, rows, args)
        if interrupted:
            break

    report = save_report(output, rows, args)
    print(f"Results: {report['counts']}\nReport: {output / 'summary.csv'}", flush=True)
    return 130 if interrupted else int(report["counts"]["failed"] + report["counts"]["error"] > 0)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
