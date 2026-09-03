#!/usr/bin/env python3
"""Render prepared A2D replay episodes into a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import sys
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
    HEAD_CAMERA_FPS,
    HEAD_CAMERA_HEIGHT,
    HEAD_CAMERA_NAME,
    HEAD_CAMERA_WIDTH,
)
from scripts.replay_a2d import (  # noqa: E402
    A2D_UPPER_BODY_POSE,
    apply_kinematic_pose,
    apply_texture_gamma,
    bind_joints,
    build_dice_replay_plan,
    interpolate_effector_state,
    interpolate_joint_state,
    set_cardboard_box_pose,
    set_target_visibility,
    validate_joint_limits,
)
from scripts.replay_a2d_head_camera import (  # noqa: E402
    CAMERA_OCCLUDER_GROUP,
    distortion_maps,
    hide_closed_head_shell,
    with_head_pose,
)


DEFAULT_MANIFEST = Path("datasets/replay_layouts.json")
DEFAULT_OUTPUT_DIR = Path("collected_datasets/a2d_head_camera_roi_lerobot")
DEFAULT_REPO_ID = "local/a2d_mujoco_pick_place_roi"
DEFAULT_TASK = "Place the green dice into the brown cardboard box."
IMAGE_FEATURE = "observation.images.head_color"
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
GRIPPER_NAMES = ("left_gripper_openness", "right_gripper_openness")
PREVIEW_WINDOW = "LeRobot collection: head_color"
ROI_X = 126
ROI_Y = 320
ROI_WIDTH = 848
ROI_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    defaults = dict(A2D_UPPER_BODY_POSE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--fps", type=int, default=int(HEAD_CAMERA_FPS))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
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
        help="Store ideal pinhole frames instead of applying real-camera distortion",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display frames while collecting; Q or Esc aborts collection",
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
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


def fixed_rate_sample_times(duration_s: float, fps: int) -> np.ndarray:
    """Return fixed-rate timestamps no later than the source trajectory end."""

    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_count = int(np.floor(duration_s * fps + 1e-9)) + 1
    return np.arange(frame_count, dtype=float) / float(fps)


def compose_robot_state(
    arm_joint_positions: np.ndarray, gripper_openness: np.ndarray
) -> np.ndarray:
    """Compose [left arm 7, right arm 7, left/right gripper] float32 state."""

    arm = np.asarray(arm_joint_positions, dtype=np.float32)
    gripper = np.asarray(gripper_openness, dtype=np.float32)
    if arm.shape != (14,):
        raise ValueError(f"Expected 14 arm joint positions, got {arm.shape}")
    if gripper.shape != (2,):
        raise ValueError(f"Expected two gripper values, got {gripper.shape}")
    if not np.isfinite(arm).all() or not np.isfinite(gripper).all():
        raise ValueError("Robot state contains NaN or infinite values")
    if np.any((gripper < 0.0) | (gripper > 1.0)):
        raise ValueError("Gripper openness must be in [0, 1]")
    return np.concatenate((arm, gripper)).astype(np.float32, copy=False)


def crop_head_camera_roi(image_rgb: np.ndarray) -> np.ndarray:
    """Crop the calibrated 1280x800 frame to image[320:800, 126:974]."""

    if image_rgb.shape != (HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH, 3):
        raise ValueError(
            "Expected full head-camera RGB image with shape "
            f"({HEAD_CAMERA_HEIGHT}, {HEAD_CAMERA_WIDTH}, 3), got {image_rgb.shape}"
        )
    x_end = ROI_X + ROI_WIDTH
    y_end = ROI_Y + ROI_HEIGHT
    return np.ascontiguousarray(image_rgb[ROI_Y:y_end, ROI_X:x_end])


def lerobot_features(joint_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if len(joint_names) != 14:
        raise ValueError(f"Expected 14 arm joint names, got {len(joint_names)}")
    state_names = [*joint_names, *GRIPPER_NAMES]
    return {
        IMAGE_FEATURE: {
            "dtype": "video",
            "shape": (ROI_HEIGHT, ROI_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_FEATURE: {
            "dtype": "float32",
            "shape": (16,),
            "names": state_names,
        },
        ACTION_FEATURE: {
            "dtype": "float32",
            "shape": (16,),
            "names": state_names,
        },
    }


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LeRobot is not installed in .venv. Install the optional dependency with:\n"
            "  uv pip install --python .venv/bin/python -r requirements-lerobot.txt"
        ) from exc
    return LeRobotDataset


def has_pending_frames(dataset: Any) -> bool:
    """Support both LeRobot 0.4.x and the newer writer-delegated API."""

    checker = getattr(dataset, "has_pending_frames", None)
    if checker is not None:
        return bool(checker())
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if episode_buffer is None:
        writer = getattr(dataset, "writer", None)
        episode_buffer = getattr(writer, "episode_buffer", None)
    return episode_buffer is not None and int(episode_buffer.get("size", 0)) > 0


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.image_writer_threads < 0:
        raise ValueError("image-writer-threads must be non-negative")

    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Choose a new --output-dir to avoid overwriting collected data."
        )
    manifest = load_manifest(manifest_path)
    records = [
        record for record in manifest["episodes"] if record.get("status") == "ok"
    ]
    if args.start_index < 0 or args.start_index >= len(records):
        raise ValueError(f"start-index must be in [0, {len(records) - 1}]")
    records = records[args.start_index :]
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max-episodes must be positive")
        records = records[: args.max_episodes]
    if not records:
        raise ValueError("Manifest contains no selected replayable episodes")

    dataset_dir = Path(manifest["dataset_dir"])
    summary_path = Path(manifest["summary"])
    model = mujoco.MjModel.from_xml_path(manifest["model"])
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME
    )
    if camera_id < 0:
        raise ValueError(
            f"Model {manifest['model']} has no camera named {HEAD_CAMERA_NAME!r}"
        )
    model.vis.global_.offwidth = HEAD_CAMERA_WIDTH
    model.vis.global_.offheight = HEAD_CAMERA_HEIGHT
    hide_closed_head_shell(model)
    apply_texture_gamma(model, "cardboard_box_texture", args.box_texture_gamma)
    set_target_visibility(model, False)
    torso = manifest["fixed_torso"]
    upper_body_pose = with_head_pose(
        fixed_upper_body_pose(
            float(torso["body_lift_m"]), float(torso["body_pitch_rad"])
        ),
        args.head_yaw_deg,
        args.head_pitch_deg,
    )
    data = mujoco.MjData(model)

    first_record = records[0]
    first_cache = (manifest_path.parent / first_record["cache"]).resolve()
    first_trajectory = load_corrected_trajectory(
        dataset_dir / first_record["episode"], summary_path, first_cache
    )
    if first_trajectory.effector_positions is None:
        raise ValueError(f"{first_record['episode']} has no action_effector channel")

    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=lerobot_features(first_trajectory.joint_names),
        root=output_dir,
        robot_type="a2d_omnipicker_mujoco",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads,
        vcodec="h264",
    )

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1
    scene_option.geomgroup[2] = 1
    scene_option.geomgroup[CAMERA_OCCLUDER_GROUP] = 0
    renderer = mujoco.Renderer(
        model, height=HEAD_CAMERA_HEIGHT, width=HEAD_CAMERA_WIDTH
    )
    map_x, map_y = (None, None) if args.no_distortion else distortion_maps()
    if args.preview:
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(PREVIEW_WINDOW, ROI_WIDTH, ROI_HEIGHT)

    total_frames = 0
    aborted = False
    try:
        for output_episode_index, record in enumerate(records):
            cache_path = (manifest_path.parent / record["cache"]).resolve()
            trajectory = (
                first_trajectory
                if output_episode_index == 0
                else load_corrected_trajectory(
                    dataset_dir / record["episode"], summary_path, cache_path
                )
            )
            if trajectory.joint_names != first_trajectory.joint_names:
                raise ValueError(
                    f"Joint order changed in {record['episode']}: "
                    f"{trajectory.joint_names}"
                )
            if trajectory.effector_positions is None:
                raise ValueError(f"{record['episode']} has no action_effector channel")
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
            sample_times = fixed_rate_sample_times(trajectory.duration_s, args.fps)
            stored_frame_count = len(sample_times) - 1
            if stored_frame_count <= 0:
                raise ValueError(
                    f"{record['episode']} needs at least two sampled states to form "
                    "(state_t, action_t=state_t+1) transitions"
                )
            print(
                f"[{output_episode_index + 1}/{len(records)}] "
                f"{record['episode']}: collecting {stored_frame_count} transitions",
                flush=True,
            )

            for frame_index, (time_s, next_time_s) in enumerate(
                zip(sample_times[:-1], sample_times[1:], strict=True)
            ):
                apply_kinematic_pose(
                    model,
                    data,
                    trajectory,
                    bindings,
                    float(time_s),
                    show_target=False,
                    dice_plan=dice_plan,
                    upper_body_pose=upper_body_pose,
                )
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
                image_rgb = crop_head_camera_roi(image_rgb)

                effector_state = interpolate_effector_state(trajectory, float(time_s))
                if effector_state is None:
                    raise ValueError(
                        f"{record['episode']} has no gripper value at frame {frame_index}"
                    )
                state = compose_robot_state(
                    data.qpos[bindings.qpos_addresses], effector_state[0]
                )
                next_joint_positions, _ = interpolate_joint_state(
                    trajectory, float(next_time_s)
                )
                next_effector_state = interpolate_effector_state(
                    trajectory, float(next_time_s)
                )
                if next_effector_state is None:
                    raise ValueError(
                        f"{record['episode']} has no next gripper value at frame "
                        f"{frame_index}"
                    )
                action = compose_robot_state(
                    next_joint_positions, next_effector_state[0]
                )
                dataset.add_frame(
                    {
                        IMAGE_FEATURE: image_rgb,
                        STATE_FEATURE: state,
                        ACTION_FEATURE: action,
                        "task": args.task,
                    }
                )
                if args.preview:
                    cv2.imshow(
                        PREVIEW_WINDOW, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    )
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        aborted = True
                        break
                if (
                    (frame_index + 1) % 30 == 0
                    or frame_index + 1 == stored_frame_count
                ):
                    print(
                        f"  transition {frame_index + 1}/{stored_frame_count}",
                        flush=True,
                    )

            if aborted:
                if has_pending_frames(dataset):
                    dataset.clear_episode_buffer()
                break
            dataset.save_episode(parallel_encoding=False)
            total_frames += stored_frame_count
            print(
                f"  saved as LeRobot episode {output_episode_index}", flush=True
            )
    except BaseException:
        if has_pending_frames(dataset):
            dataset.clear_episode_buffer()
        dataset.finalize()
        raise
    else:
        dataset.finalize()
    finally:
        renderer.close()
        if args.preview:
            cv2.destroyWindow(PREVIEW_WINDOW)

    if aborted:
        print(
            f"Collection stopped by user. Saved complete episodes in {output_dir}",
            flush=True,
        )
    else:
        print(
            f"Done: {len(records)} episodes, {total_frames} frames -> {output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
