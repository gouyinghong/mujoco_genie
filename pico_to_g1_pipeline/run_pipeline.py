#!/usr/bin/env python3
"""Run the isolated raw-PICO -> split LeRobot -> G1 retarget pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from pipeline_safety import PIPELINE_ROOT, refuse_protected_dataset_write


RAW_SESSION = PIPELINE_ROOT / "data/raw/session_20260723_084253_461"
ROBOT_REFERENCE = PIPELINE_ROOT / "data/reference/genie1_pick_up_dice_804"
URDF = PIPELINE_ROOT / "assets/G1_120s/G1_120s.urdf"
LONG_DATASET = PIPELINE_ROOT / "work/lerobot_session_20260723_084253_461_self_contained"
SPLIT_DATASET = PIPELINE_ROOT / "work/lerobot_session_20260723_084253_461_split_self_contained"
RETARGET_OUTPUT = PIPELINE_ROOT / "outputs/fixed_spine3_to_g1_0723_complete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "convert", "split", "retarget"),
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and validate paths without running them",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {path}")


def run(command: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PIPELINE_ROOT, check=True)


def main() -> int:
    args = parse_args()
    # 保留 .venv/bin/python 路径；resolve() 会把它展开为系统解释器并绕开虚拟环境。
    python = sys.executable

    require_directory(RAW_SESSION)
    require_directory(ROBOT_REFERENCE)
    require_file(URDF)
    require_file(PIPELINE_ROOT / "frame_range.json")
    for path, purpose in (
        (LONG_DATASET, "write converted LeRobot data"),
        (SPLIT_DATASET, "write split LeRobot data"),
        (RETARGET_OUTPUT, "write retarget output"),
    ):
        refuse_protected_dataset_write(path, purpose=purpose)

    commands = {
        "convert": [
            python,
            str(PIPELINE_ROOT / "convert_pico_session_direct_to_lerobot.py"),
            str(RAW_SESSION),
            "--output-path",
            str(LONG_DATASET),
            "--downsample-factor",
            "1",
            "--sg-window",
            "0",
        ],
        "split": [
            python,
            str(PIPELINE_ROOT / "split_human_lerobot_episode.py"),
            "--src-root",
            str(LONG_DATASET),
            "--output-root",
            str(SPLIT_DATASET),
            "--ranges-file",
            str(PIPELINE_ROOT / "frame_range.json"),
        ],
        "retarget": [
            python,
            str(PIPELINE_ROOT / "ego2robot/retarget_fixed_spine3_to_g1.py"),
            "--human-root",
            str(SPLIT_DATASET),
            "--robot-root",
            str(ROBOT_REFERENCE),
            "--urdf",
            str(URDF),
            "--output-dir",
            str(RETARGET_OUTPUT),
            "--all-episodes",
        ],
    }
    stages = ("convert", "split", "retarget") if args.stage == "all" else (args.stage,)
    for stage in stages:
        output = {
            "convert": LONG_DATASET,
            "split": SPLIT_DATASET,
            "retarget": RETARGET_OUTPUT,
        }[stage]
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing stage output: {output}"
            )
        run(commands[stage], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
