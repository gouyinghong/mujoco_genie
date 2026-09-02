#!/usr/bin/env python3
"""Replay prepared A2D episodes through the calibrated 1280x800 head camera."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (  # noqa: E402
    LAYOUT_SCHEMA,
    fixed_upper_body_pose,
    load_corrected_trajectory,
)
from scripts.a2d_head_camera import (  # noqa: E402
    HEAD_CAMERA_DISTORTION,
    HEAD_CAMERA_FPS,
    HEAD_CAMERA_HEIGHT,
    HEAD_CAMERA_NAME,
    HEAD_CAMERA_WIDTH,
    camera_matrix,
)
from scripts.replay_a2d import (  # noqa: E402
    A2D_UPPER_BODY_POSE,
    apply_kinematic_pose,
    apply_texture_gamma,
    bind_joints,
    build_dice_replay_plan,
    set_cardboard_box_pose,
    set_target_visibility,
    trajectory_frame_at_time,
    validate_joint_limits,
)


DEFAULT_MANIFEST = Path("datasets/replay_layouts.json")
WINDOW_NAME = "A2D head_color (calibrated 1280x800)"
CAMERA_OCCLUDER_GROUP = 5


def parse_args() -> argparse.Namespace:
    defaults = dict(A2D_UPPER_BODY_POSE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    parser.add_argument("--show-target", action="store_true")
    parser.add_argument("--box-texture-gamma", type=float, default=0.65)
    parser.add_argument(
        "--head-yaw-deg",
        type=float,
        default=float(np.degrees(defaults["joint_head_yaw"])),
    )
    parser.add_argument(
        "--head-pitch-deg",
        type=float,
        default=float(np.degrees(defaults["joint_head_pitch"])),
    )
    parser.add_argument(
        "--no-distortion",
        action="store_true",
        help="Show the ideal pinhole image without the calibrated plumb-bob distortion",
    )
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema") != LAYOUT_SCHEMA:
        raise ValueError(
            f"Unsupported manifest schema {document.get('schema')!r}; "
            f"expected {LAYOUT_SCHEMA!r}"
        )
    return document


def with_head_pose(
    upper_body_pose: tuple[tuple[str, float], ...],
    head_yaw_deg: float,
    head_pitch_deg: float,
) -> tuple[tuple[str, float], ...]:
    overrides = {
        "joint_head_yaw": float(np.deg2rad(head_yaw_deg)),
        "joint_head_pitch": float(np.deg2rad(head_pitch_deg)),
    }
    return tuple((name, overrides.get(name, value)) for name, value in upper_body_pose)


def hide_closed_head_shell(model: mujoco.MjModel) -> None:
    """Hide the closed STL only in this dedicated camera-rendering model."""

    head_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "link-pitch_head"
    )
    if head_id < 0:
        raise ValueError("Model has no body named 'link-pitch_head'")
    visual_ids = np.flatnonzero(
        (model.geom_bodyid == head_id) & (model.geom_contype == 0)
    )
    if len(visual_ids) != 1:
        raise ValueError(
            f"Expected one head visual geom, found {len(visual_ids)}"
        )
    model.geom_group[visual_ids] = CAMERA_OCCLUDER_GROUP


def distortion_maps() -> tuple[np.ndarray, np.ndarray]:
    matrix = camera_matrix()
    return cv2.initInverseRectificationMap(
        matrix,
        HEAD_CAMERA_DISTORTION,
        np.eye(3),
        matrix,
        (HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT),
        cv2.CV_32FC1,
    )


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    records = [
        record for record in manifest["episodes"] if record.get("status") == "ok"
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
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME
    )
    if camera_id < 0:
        raise ValueError(
            f"Model {manifest['model']} has no camera named {HEAD_CAMERA_NAME!r}. "
            "Regenerate or update A2D_with_box.xml."
        )
    model.vis.global_.offwidth = HEAD_CAMERA_WIDTH
    model.vis.global_.offheight = HEAD_CAMERA_HEIGHT
    hide_closed_head_shell(model)
    apply_texture_gamma(model, "cardboard_box_texture", args.box_texture_gamma)
    set_target_visibility(model, args.show_target)
    torso = manifest["fixed_torso"]
    upper_body_pose = with_head_pose(
        fixed_upper_body_pose(
            float(torso["body_lift_m"]), float(torso["body_pitch_rad"])
        ),
        args.head_yaw_deg,
        args.head_pitch_deg,
    )
    data = mujoco.MjData(model)

    def prepare(record_index: int):
        record = records[record_index]
        cache_path = (manifest_path.parent / record["cache"]).resolve()
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
            f"[ok]; frames={trajectory.frames}, "
            f"head=({args.head_yaw_deg:.3f}, {args.head_pitch_deg:.3f}) deg; "
            "camera=calibrated extrinsics",
            flush=True,
        )
        return trajectory, bindings, dice_plan

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1
    scene_option.geomgroup[2] = 1
    scene_option.geomgroup[CAMERA_OCCLUDER_GROUP] = 0
    renderer = mujoco.Renderer(
        model, height=HEAD_CAMERA_HEIGHT, width=HEAD_CAMERA_WIDTH
    )
    map_x, map_y = (None, None) if args.no_distortion else distortion_maps()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, HEAD_CAMERA_WIDTH, HEAD_CAMERA_HEIGHT)

    current_index = args.start_index
    trajectory, bindings, dice_plan = prepare(current_index)
    paused = not args.start_immediately
    elapsed = 0.0
    previous_wall_time = time.monotonic()
    next_render_time = 0.0
    print(
        "Controls in head_color window: SPACE pause/resume, N/P episode, "
        "ENTER restart, Q/ESC quit.",
        flush=True,
    )
    try:
        while True:
            iteration_start = time.monotonic()
            wall_delta = iteration_start - previous_wall_time
            previous_wall_time = iteration_start
            key = cv2.waitKeyEx(1)
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
                frame = trajectory_frame_at_time(trajectory, elapsed)
                print(
                    f"{'Paused' if paused else 'Resumed'}: "
                    f"{records[current_index]['episode']} frame "
                    f"{frame}/{trajectory.frames - 1}",
                    flush=True,
                )
            elif key in (10, 13):
                elapsed = 0.0
                paused = False
            elif key in (ord("n"), ord("N"), ord("p"), ord("P")):
                delta = 1 if key in (ord("n"), ord("N")) else -1
                current_index = (current_index + delta) % len(records)
                trajectory, bindings, dice_plan = prepare(current_index)
                elapsed = 0.0
                paused = not args.start_immediately

            if not paused:
                elapsed += wall_delta * args.speed
            if elapsed >= trajectory.duration_s:
                elapsed = trajectory.duration_s
                if args.auto_advance and current_index + 1 < len(records):
                    current_index += 1
                    trajectory, bindings, dice_plan = prepare(current_index)
                    elapsed = 0.0
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
            if iteration_start >= next_render_time:
                renderer.update_scene(
                    data, camera=HEAD_CAMERA_NAME, scene_option=scene_option
                )
                image_rgb = renderer.render()
                if map_x is not None and map_y is not None:
                    image_rgb = cv2.remap(
                        image_rgb,
                        map_x,
                        map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                cv2.imshow(WINDOW_NAME, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                next_render_time = iteration_start + 1.0 / HEAD_CAMERA_FPS
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            remaining = 1.0 / 120.0 - (time.monotonic() - iteration_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        renderer.close()
        cv2.destroyWindow(WINDOW_NAME)


if __name__ == "__main__":
    main()
