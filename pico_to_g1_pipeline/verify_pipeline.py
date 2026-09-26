#!/usr/bin/env python3
"""Verify the isolated pipeline inputs without modifying any data."""

from __future__ import annotations

import importlib
from pathlib import Path

from pipeline_safety import PIPELINE_ROOT, PROTECTED_DATASETS_ROOT


def main() -> int:
    required = (
        PIPELINE_ROOT / "data/raw/session_20260723_084253_461/body_tracking.jsonl",
        PIPELINE_ROOT / "data/raw/session_20260723_084253_461/hands.jsonl",
        PIPELINE_ROOT / "data/raw/session_20260723_084253_461/controllers.jsonl",
        PIPELINE_ROOT / "data/raw/session_20260723_084253_461/camera.mp4",
        PIPELINE_ROOT / "data/raw/session_20260723_084253_461/video_metadata.json",
        PIPELINE_ROOT / "data/reference/genie1_pick_up_dice_804/meta/info.json",
        PIPELINE_ROOT / "assets/G1_120s/G1_120s.urdf",
        PIPELINE_ROOT / "frame_range.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pipeline inputs: {missing}")

    for module in ("numpy", "scipy", "pyarrow", "lerobot", "av", "pinocchio"):
        imported = importlib.import_module(module)
        print(f"{module}: {getattr(imported, '__version__', 'installed')}")

    print(f"pipeline root:       {PIPELINE_ROOT}")
    print(f"protected datasets:  {PROTECTED_DATASETS_ROOT}")
    print("pipeline input verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
