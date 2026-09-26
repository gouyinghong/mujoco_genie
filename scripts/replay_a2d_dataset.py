#!/usr/bin/env python3
"""Interactively replay prepared A2D dataset episodes with one fixed torso pose."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (  # noqa: E402
    DEFAULT_REPLAY_MANIFEST,
    LAYOUT_SCHEMA,
    fixed_upper_body_pose,
    load_corrected_trajectory,
)
from scripts.replay_a2d import (  # noqa: E402
    apply_kinematic_pose,
    apply_texture_gamma,
    bind_joints,
    build_dice_replay_plan,
    set_cardboard_box_pose,
    set_target_visibility,
    trajectory_frame_at_time,
    validate_joint_limits,
)


DEFAULT_MANIFEST = DEFAULT_REPLAY_MANIFEST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    parser.add_argument("--show-target", action="store_true")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--box-texture-gamma", type=float, default=0.65)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also load failed records that still have generated cache/layout data",
    )
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema") != LAYOUT_SCHEMA:
        raise ValueError(
            f"Unsupported manifest schema {document.get('schema')!r}; "
            f"expected {LAYOUT_SCHEMA!r}"
        )
    return document


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    records = [
        record
        for record in manifest["episodes"]
        if record.get("status") == "ok"
        or (
            args.include_failed
            and all(key in record for key in ("cache", "box", "dice_center_frame"))
        )
    ]
    if args.max_episodes is not None:
        records = records[: args.max_episodes]
    if not records:
        raise ValueError("Manifest contains no replayable episodes")
    if not 0 <= args.start_index < len(records):
        raise ValueError(f"start-index must be in [0, {len(records) - 1}]")

    dataset_dir = Path(manifest["dataset_dir"])
    summary_path = Path(manifest["summary"])
    model = mujoco.MjModel.from_xml_path(manifest["model"])
    apply_texture_gamma(model, "cardboard_box_texture", args.box_texture_gamma)
    torso = manifest["fixed_torso"]
    upper_body_pose = fixed_upper_body_pose(
        float(torso["body_lift_m"]), float(torso["body_pitch_rad"])
    )
    data = mujoco.MjData(model)

    def prepare(record_index: int):
        record = records[record_index]
        cache_value = record.get("cache")
        cache_path = (
            None
            if cache_value is None
            else (manifest_path.parent / cache_value).resolve()
        )
        trajectory = load_corrected_trajectory(
            dataset_dir / record["episode"], summary_path, cache_path
        )
        bindings = bind_joints(model, trajectory.joint_names)
        validate_joint_limits(model, trajectory, bindings)
        box = record["box"]
        set_cardboard_box_pose(
            model,
            x=float(box["x"]),
            y=float(box["y"]),
            yaw_deg=float(box["yaw_deg"]),
        )
        dice_plan = build_dice_replay_plan(
            model,
            trajectory,
            bindings,
            upper_body_pose=upper_body_pose,
            dice_on_table=True,
            align_dice_to_gripper=True,
            dice_center_frame=int(record["dice_center_frame"]),
            dice_xy_offset_m=tuple(record.get("dice_xy_offset_m", (0.0, 0.0))),
        )
        if dice_plan is None:
            raise ValueError(f"Could not create dice plan for {record['episode']}")
        data.qpos[:] = model.qpos0
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            0.0,
            show_target=args.show_target,
            dice_plan=dice_plan,
            upper_body_pose=upper_body_pose,
        )
        print(
            f"Episode {record_index + 1}/{len(records)}: {record['episode']} "
            f"[{record['status']}]; "
            f"frames={trajectory.frames}, center={record['dice_center_frame']}, "
            f"grasp={dice_plan.grasp_frame}, release={dice_plan.release_frame}, "
            f"arm_z_offset={record['right_arm_z_offset_m']:.4f} m, "
            f"place_lift={record.get('placement_lift_m', 0.0):.3f} m, "
            f"box=({box['x']:.4f}, {box['y']:.4f}, {box['yaw_deg']:.1f} deg)",
            flush=True,
        )
        return trajectory, bindings, dice_plan

    if args.headless:
        for index in range(len(records)):
            prepare(index)
        print(f"Validated {len(records)} prepared episodes")
        return

    from mujoco import viewer as mujoco_viewer

    set_target_visibility(model, args.show_target)
    current_index = args.start_index
    trajectory, bindings, dice_plan = prepare(current_index)
    command_lock = threading.Lock()
    commands: list[str] = []

    def key_callback(keycode: int) -> None:
        command = {
            ord(" "): "toggle",
            ord("N"): "next",
            ord("P"): "previous",
            257: "restart",  # GLFW_KEY_ENTER
        }.get(keycode)
        if command is not None:
            with command_lock:
                commands.append(command)

    with mujoco_viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        viewer.cam.lookat[:] = (0.65, 0.0, 0.8)
        viewer.cam.distance = 1.7
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0
        viewer.opt.geomgroup[0] = int(args.show_collision)
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0
        paused = not args.start_immediately
        elapsed = 0.0
        previous_wall_time = time.monotonic()
        print(
            "Controls: SPACE pause/resume, N next episode, P previous episode, "
            "ENTER restart current episode."
        )

        while viewer.is_running():
            iteration_start = time.monotonic()
            wall_delta = iteration_start - previous_wall_time
            previous_wall_time = iteration_start
            with command_lock:
                pending = commands.copy()
                commands.clear()
            for command in pending:
                if command == "toggle":
                    paused = not paused
                    frame = trajectory_frame_at_time(trajectory, elapsed)
                    print(
                        f"{'Paused' if paused else 'Resumed'}: "
                        f"{records[current_index]['episode']} frame "
                        f"{frame}/{trajectory.frames - 1}",
                        flush=True,
                    )
                elif command == "restart":
                    elapsed = 0.0
                    paused = False
                else:
                    delta = 1 if command == "next" else -1
                    current_index = (current_index + delta) % len(records)
                    trajectory, bindings, dice_plan = prepare(current_index)
                    elapsed = 0.0
                    paused = not args.start_immediately
                wall_delta = 0.0

            if not paused:
                elapsed += wall_delta * args.speed
            if elapsed >= trajectory.duration_s:
                elapsed = trajectory.duration_s
                if args.auto_advance and current_index + 1 < len(records):
                    current_index += 1
                    trajectory, bindings, dice_plan = prepare(current_index)
                    elapsed = 0.0
                    paused = False
                else:
                    paused = True

            apply_kinematic_pose(
                model,
                data,
                trajectory,
                bindings,
                elapsed,
                show_target=args.show_target,
                dice_plan=dice_plan,
                upper_body_pose=upper_body_pose,
            )
            # N/P are also built-in MuJoCo visualization shortcuts. Keep their
            # render effects disabled because this player reserves the keys
            # for episode navigation.
            viewer.opt.geomgroup[0] = int(args.show_collision)
            viewer.opt.geomgroup[1] = 1
            viewer.opt.geomgroup[2] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0
            viewer.sync()
            remaining = 1.0 / 120.0 - (time.monotonic() - iteration_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
