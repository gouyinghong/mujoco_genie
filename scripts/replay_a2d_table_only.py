#!/usr/bin/env python3
"""Replay one prepared A2D episode with only the robot and table visible."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (  # noqa: E402
    fixed_upper_body_pose,
    load_corrected_trajectory,
)
from scripts.replay_a2d import (  # noqa: E402
    apply_kinematic_pose,
    bind_joints,
    disable_dice,
    replay_in_viewer,
    validate_joint_limits,
)
from scripts.replay_a2d_dataset import load_manifest  # noqa: E402


DEFAULT_MANIFEST = Path("datasets/replay_layouts.json")
DEFAULT_MODEL = Path("assets/A2D_Omnipicker/A2D.xml")
DEFAULT_EPISODE = "episode_000012.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--show-target", action="store_true")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument(
        "--no-loop",
        action="store_false",
        dest="loop",
        help="Play once instead of looping",
    )
    parser.set_defaults(loop=True)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("speed must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    episode_name = Path(args.episode).name
    record = next(
        (
            item
            for item in manifest["episodes"]
            if item.get("episode") == episode_name and item.get("status") == "ok"
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Manifest contains no replayable record for {episode_name}")

    model_path = args.model.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    box_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box"
    )
    if box_body_id >= 0:
        raise ValueError(
            f"Table-only replay requires a model without cardboard_box: {model_path}"
        )
    if not disable_dice(model):
        raise ValueError(f"Model does not contain the expected dice body: {model_path}")

    dataset_dir = Path(manifest["dataset_dir"])
    summary_path = Path(manifest["summary"])
    cache_path = (manifest_path.parent / record["cache"]).resolve()
    trajectory = load_corrected_trajectory(
        dataset_dir / episode_name,
        summary_path,
        cache_path,
    )
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)
    torso = manifest["fixed_torso"]
    upper_body_pose = fixed_upper_body_pose(
        float(torso["body_lift_m"]),
        float(torso["body_pitch_rad"]),
    )

    print(
        f"Table-only replay: {episode_name}; frames={trajectory.frames}, "
        f"body_lift={float(torso['body_lift_m']):.6f} m, "
        f"body_pitch={float(torso['body_pitch_rad']):.6f} rad, "
        f"arm_z_offset={float(record['right_arm_z_offset_m']):.4f} m",
        flush=True,
    )

    if args.headless:
        data = mujoco.MjData(model)
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            0.0,
            show_target=args.show_target,
            upper_body_pose=upper_body_pose,
        )
        print("Validated robot-and-table replay with dice disabled")
        return

    replay_in_viewer(
        model,
        trajectory,
        bindings,
        None,
        speed=args.speed,
        loop=args.loop,
        wait_for_start=not args.start_immediately,
        upper_body_pose=upper_body_pose,
        show_target=args.show_target,
        show_collision=args.show_collision,
    )


if __name__ == "__main__":
    main()
