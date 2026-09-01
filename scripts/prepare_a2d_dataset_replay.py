#!/usr/bin/env python3
"""Precompute fixed-torso replay corrections and scene layouts for a dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (
    DEFAULT_BODY_LIFT_M,
    DEFAULT_BODY_PITCH_RAD,
    prepare_dataset_layouts,
)
from scripts.convert_a2d_to_mjcf import DEFAULT_A2D_WITH_BOX_MJCF


DEFAULT_DATASET = Path(
    "datasets/fixed_spine3_to_g1_0723_add_effector_gripper_6cm_return"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_A2D_WITH_BOX_MJCF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--body-lift-m", type=float, default=DEFAULT_BODY_LIFT_M)
    parser.add_argument(
        "--body-pitch-rad", type=float, default=DEFAULT_BODY_PITCH_RAD
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--episode",
        action="append",
        dest="episodes",
        metavar="EPISODE",
        help=(
            "Only recompute this episode and merge it into the existing manifest. "
            "May be specified more than once."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes and args.max_episodes is not None:
        raise ValueError("--episode and --max-episodes cannot be used together")
    output = args.output or args.dataset_dir.parent / "replay_layouts.json"
    result = prepare_dataset_layouts(
        args.model,
        args.dataset_dir,
        output,
        body_lift_m=args.body_lift_m,
        body_pitch_rad=args.body_pitch_rad,
        max_episodes=args.max_episodes,
        episode_names=tuple(args.episodes) if args.episodes else None,
    )
    print(f"Layout manifest: {output.resolve()}")
    print(
        f"Episodes: total={result['counts']['total']}, "
        f"ok={result['counts']['ok']}, failed={result['counts']['failed']}"
    )


if __name__ == "__main__":
    main()
