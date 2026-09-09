#!/usr/bin/env python3
"""Prepare independent closure copies and replay A2D episodes with dynamics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.a2d_closed_loop import add_gripper_arguments
from scripts.prepare_a2d_grasp_hold import prepare
from scripts.replay_a2d_physics import DEFAULT_MANIFEST, load_manifest

SCHEMA = "a2d_physics_batch.v1"
COMPLETE = {"passed", "failed", "skipped"}
METRICS = ("pick_and_place_success", "low_slip_pick_and_place_success",
           "max_dice_translation_in_gripper_m", "max_dice_rotation_in_gripper_deg",
           "max_dice_drop_relative_to_gripper_m", "retention_bilateral_contact_fraction",
           "landed_in_box", "physics_warnings", "postopening_finger_contact_duration_s",
           "last_finger_contact_delay_from_opening_s")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def select_records(manifest: dict, episodes: list[str] | None, start: int, limit: int | None) -> list[dict]:
    records = manifest["episodes"]
    names = [r["episode"] for r in records]
    if len(set(names)) != len(names):
        raise ValueError("Manifest contains duplicate episode names")
    if any(Path(n).name != n or not n.endswith(".npz") for n in names):
        raise ValueError("Episode names must be plain .npz filenames")
    if episodes:
        missing = set(episodes) - set(names)
        if missing:
            raise ValueError(f"Episodes absent from manifest: {sorted(missing)}")
        records = [r for r in records if r["episode"] in episodes]
    if start < 0 or (limit is not None and limit <= 0):
        raise ValueError("start-index must be nonnegative and limit positive")
    selected = records[start:] if limit is None else records[start:start + limit]
    if not selected:
        raise ValueError("No episodes selected")
    return selected


def atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_report(output: Path, rows: list[dict]) -> dict:
    counts = {status: sum(r["status"] == status for r in rows)
              for status in ("pending", "running", "passed", "failed", "error", "skipped", "interrupted")}
    report = {"schema": SCHEMA, "updated_at": now(), "total_selected": len(rows),
              "counts": counts, "episodes": rows}
    atomic_json(output / "summary.json", report)
    fields = ("episode", "status", "reason", "prepared_manifest", "metrics_file", "log_file", *METRICS)
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "summary.csv")
    return report


def replay_command(args: argparse.Namespace, manifest: Path, episode: str, metrics: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts/replay_a2d_physics.py"),
               "--manifest", str(manifest), "--episode", episode,
               "--metrics-output", str(metrics)]
    for name in ("gripper_control", "gripper_kp", "gripper_kv", "gripper_max_torque",
                 "gripper_close_bias", "gripper_sliding_friction", "gripper_release_mode", "grasp_lower_m",
                 "arm_contact_mode", "physics_timestep", "contact_impratio", "speed",
                 "settle_time_s", "post_rollout_s", "dice_linear_damping", "dice_angular_damping"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    command.extend(["--headless"] if args.headless else ["--start-immediately", "--exit-when-finished"])
    return command


def prepare_episode(args: argparse.Namespace, manifest: dict, record: dict, output: Path) -> Path:
    # Already processed single-episode manifests must not receive a second hold.
    if args.keep_timing or manifest.get("processing", {}).get("type") in {"stationary_gripper_closure", "object_centric_augmentation"}:
        return args.manifest
    base = output / "data" / Path(record["episode"]).stem
    candidate = base
    attempt = 0
    while candidate.exists():
        existing = candidate / "manifest.json"
        if existing.is_file():
            document = load_manifest(existing)
            processing = document.get("processing", {})
            if (processing.get("source_sha256") == digest(Path(manifest["dataset_dir"]) / record["episode"])
                    and processing.get("close_duration_s") == args.close_duration_s
                    and processing.get("hold_frame") == record["boundaries"]["close_end_frame"]):
                return existing
            raise ValueError(f"Prepared episode does not match this run: {existing}")
        # Interrupted preparation may leave a partial directory. Preserve it.
        attempt += 1
        candidate = base.with_name(base.name + f"_retry_{attempt}")
    return prepare(args.manifest, record["episode"], candidate, duration_s=args.close_duration_s)


def run_batch(args: argparse.Namespace) -> int:
    args.manifest = args.manifest.expanduser().resolve()
    manifest = load_manifest(args.manifest)
    records = select_records(manifest, args.episodes, args.start_index, args.limit)
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                  if key not in {"output_dir", "resume", "timeout_s"}}
    fingerprints = {"manifest": digest(args.manifest), "model": digest(Path(manifest["model"])),
                    "summary": digest(Path(manifest["summary"]))}
    for record in records:
        fingerprints[record["episode"]] = digest(Path(manifest["dataset_dir"]) / record["episode"])
        if record.get("cache"):
            fingerprints[record["episode"] + ":cache"] = digest(args.manifest.parent / record["cache"])
    config = {"schema": SCHEMA, "parameters": parameters, "input_sha256": fingerprints}
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        saved_config = json.loads((output / "configuration.json").read_text())
        # Reports created before the release-mode option used recorded opening.
        saved_config.get("parameters", {}).setdefault("gripper_release_mode", "recorded")
        if saved_config != config:
            raise ValueError("Resume parameters or input files changed; use the original command or a new output directory")
        saved = json.loads((output / "summary.json").read_text())
        if saved.get("schema") != SCHEMA:
            raise ValueError("Unsupported batch report schema")
        rows = saved["episodes"]
        if [r["episode"] for r in rows] != [r["episode"] for r in records]:
            raise ValueError("Resume episode selection does not match report")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "configuration.json", config)
        rows = [{"episode": r["episode"], "status": "pending"} for r in records]
    (output / "results").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    save_report(output, rows)
    print(f"Batch: {len(rows)} episodes; output={output}", flush=True)
    interrupted = False
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        if row["status"] in COMPLETE:
            continue
        if record.get("status") != "ok":
            row.update(status="skipped", reason=f"Source manifest status={record.get('status')}: {record.get('error', '')}")
            save_report(output, rows)
            continue
        stem = Path(record["episode"]).stem
        metrics_path, log_path = output / "results" / f"{stem}.json", output / "logs" / f"{stem}.log"
        row.update(status="running", reason="", metrics_file=str(metrics_path), log_file=str(log_path))
        save_report(output, rows)
        print(f"[{index + 1}/{len(rows)}] {record['episode']}", flush=True)
        try:
            prepared = prepare_episode(args, manifest, record, output)
            row["prepared_manifest"] = str(prepared)
            row["validation_mode"] = "headless" if args.headless else "fresh_headless_rollout_after_viewer"
            command = replay_command(args, prepared, record["episode"], metrics_path)
            row["command"] = command
            save_report(output, rows)
            # A child may finish writing metrics just before the parent is interrupted.
            if not metrics_path.exists():
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"\nBatch attempt at {now()}\n")
                    stream.flush()
                    process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=args.timeout_s or None, check=False)
                if process.returncode in (130, -2):
                    raise KeyboardInterrupt
                if process.returncode:
                    raise RuntimeError(f"Replay exited with code {process.returncode}; see {log_path}")
            metrics = json.loads(metrics_path.read_text())
            if not isinstance(metrics.get("low_slip_pick_and_place_success"), bool):
                raise ValueError(f"Incomplete validation metrics: {metrics_path}")
            row.update({key: metrics.get(key) for key in METRICS})
            row["status"] = "passed" if metrics["low_slip_pick_and_place_success"] else "failed"
            row["reason"] = "" if row["status"] == "passed" else "Did not pass low-slip pick-and-place criteria"
            print(f"  {row['status']}: displacement={metrics['max_dice_translation_in_gripper_m'] * 1000:.2f} mm, "
                  f"rotation={metrics['max_dice_rotation_in_gripper_deg']:.2f} deg", flush=True)
        except KeyboardInterrupt:
            row.update(status="interrupted", reason="User interrupted replay")
            interrupted = True
        except Exception as error:
            row.update(status="error", reason=f"{type(error).__name__}: {error}")
            print(f"  error: {row['reason']}", flush=True)
        finally:
            save_report(output, rows)
        if interrupted:
            break
    report = save_report(output, rows)
    print(f"Results: {report['counts']}\nReport: {output / 'summary.csv'}", flush=True)
    return 130 if interrupted else int(report["counts"]["failed"] + report["counts"]["error"] > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gripper_arguments(parser)
    parser.set_defaults(arm_contact_mode="constrained", physics_timestep=.0005,
                        contact_impratio=100, gripper_close_bias=0, gripper_sliding_friction=3)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episodes", nargs="+", help="Select filenames; replay follows manifest order")
    parser.add_argument("--start-index", type=int, default=0, help="Offset in selected manifest records, including skipped records")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--headless", action="store_true", help="Validate without opening viewers")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse identical run; retry errors/interrupted episodes, skip completed results")
    parser.add_argument("--keep-timing", action="store_true", help="Use existing timing and layouts without preparing closure copies")
    parser.add_argument("--close-duration-s", type=float, default=2)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--settle-time-s", type=float, default=.4)
    parser.add_argument("--post-rollout-s", type=float, default=1)
    parser.add_argument("--dice-linear-damping", type=float, default=.02)
    parser.add_argument("--dice-angular-damping", type=float, default=.0005)
    parser.add_argument("--timeout-s", type=float, default=0, help="Per-child wall timeout; 0 allows unlimited viewing/pause")
    args = parser.parse_args(argv)
    import math
    for name in ("close_duration_s", "speed", "settle_time_s", "post_rollout_s", "timeout_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in {"close_duration_s", "speed"} and value == 0):
            parser.error(f"Invalid --{name.replace('_', '-')}")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    if args.output_dir is None:
        args.output_dir = Path("logs/a2d_physics_batch") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return args


if __name__ == "__main__":
    raise SystemExit(run_batch(parse_args()))
