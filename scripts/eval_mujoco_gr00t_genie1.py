#!/usr/bin/env python3
"""Run a GR00T policy server closed-loop in the A2D MuJoCo scene.

The policy I/O matches ``collect_a2d_lerobot.py``:

    state/action = [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
    image = distorted head_color[320:800, 126:974] (848x480 RGB)

The robot is position-controlled kinematically while the die remains a dynamic
MuJoCo free body, so grasping and placement still depend on physical contacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

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
from scripts.collect_a2d_lerobot import (  # noqa: E402
    DEFAULT_TASK,
    ROI_HEIGHT,
    ROI_WIDTH,
    crop_head_camera_roi,
)
from scripts.replay_a2d import (  # noqa: E402
    A2D_UPPER_BODY_POSE,
    JointBindings,
    apply_kinematic_pose,
    apply_texture_gamma,
    bind_joints,
    build_dice_replay_plan,
    set_cardboard_box_pose,
    set_gripper_command,
    set_target_visibility,
    set_upper_body_pose,
    validate_joint_limits,
)
from scripts.replay_a2d_head_camera import (  # noqa: E402
    CAMERA_OCCLUDER_GROUP,
    distortion_maps,
    hide_closed_head_shell,
    with_head_pose,
)


DEFAULT_MANIFEST = Path("datasets/replay_layouts.json")
DEFAULT_POLICY_HOST = "172.20.103.219"
DEFAULT_POLICY_PORT = 5555
ACTION_DIM = 16
ARM_DIM = 14
GRIPPER_DIM = 2
LEFT_ARM_SLICE = slice(0, 7)
LEFT_GRIPPER_INDEX = 7
RIGHT_ARM_SLICE = slice(8, 15)
RIGHT_GRIPPER_INDEX = 15
HEAD_WINDOW = "GR00T policy input: head_color ROI 848x480"


def parse_args() -> argparse.Namespace:
    defaults = dict(A2D_UPPER_BODY_POSE)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Index among manifest episodes whose status is ok; used for initial layout",
    )
    parser.add_argument("--policy-host", default=DEFAULT_POLICY_HOST)
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT)
    parser.add_argument("--policy-timeout-ms", type=int, default=30000)
    parser.add_argument("--language", default=DEFAULT_TASK)
    parser.add_argument("--frequency", type=float, default=float(HEAD_CAMERA_FPS))
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=0,
        help="Use at most this many actions from each returned chunk; 0 uses the full chunk",
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--success-hold-steps", type=int, default=10)
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
    parser.add_argument("--no-distortion", action="store_true")
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-head-window", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--save-actions", type=Path)
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


def normalize_action_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, 1, -1)
    if array.ndim == 2:
        return array.reshape(1, *array.shape)
    if array.ndim != 3:
        raise ValueError(f"Expected action rank 1, 2, or 3, got {array.shape}")
    return array


def _action_value(action: dict[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in action:
            return np.asarray(action[name], dtype=np.float32)
    raise KeyError(
        f"Policy action is missing all aliases {names}; available={sorted(action)}"
    )


def iter_action_chunk(action: dict[str, Any]) -> Iterator[np.ndarray]:
    """Yield actions in the training dataset's left/right/gripper order."""

    if "action" in action:
        flat = normalize_action_array(np.asarray(action["action"], dtype=np.float32))
        if flat.shape[0] != 1 or flat.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Expected flat policy action shaped (1, H, {ACTION_DIM}), got {flat.shape}"
            )
        for horizon_index in range(flat.shape[1]):
            yield flat[0, horizon_index].copy()
        return

    left_arm = normalize_action_array(
        _action_value(action, "left_arm", "action.left_arm")
    )
    right_arm = normalize_action_array(
        _action_value(action, "right_arm", "action.right_arm")
    )
    left_gripper = normalize_action_array(
        _action_value(
            action,
            "left_gripper",
            "left_hand",
            "action.left_gripper",
            "action.left_hand",
        )
    )
    right_gripper = normalize_action_array(
        _action_value(
            action,
            "right_gripper",
            "right_hand",
            "action.right_gripper",
            "action.right_hand",
        )
    )
    arrays = (left_arm, left_gripper, right_arm, right_gripper)
    horizon = left_arm.shape[1]
    expected_dims = (7, 1, 7, 1)
    for array, width in zip(arrays, expected_dims, strict=True):
        if array.shape != (1, horizon, width):
            raise ValueError(
                "Structured policy actions must share shape (1, H, D); "
                f"got {[item.shape for item in arrays]}"
            )
    for horizon_index in range(horizon):
        yield np.concatenate(
            [array[0, horizon_index].reshape(-1) for array in arrays]
        ).astype(np.float32)


def compose_policy_state(
    arm_joint_positions: np.ndarray, gripper_openness: np.ndarray
) -> np.ndarray:
    """Compose the Genie1 interleaved arm/gripper state used by the server."""

    arms = np.asarray(arm_joint_positions, dtype=np.float32).reshape(-1)
    grippers = np.asarray(gripper_openness, dtype=np.float32).reshape(-1)
    if arms.shape != (ARM_DIM,) or grippers.shape != (GRIPPER_DIM,):
        raise ValueError(
            f"Expected 14 arm and 2 gripper values, got {arms.shape} and "
            f"{grippers.shape}"
        )
    if not np.isfinite(arms).all() or not np.isfinite(grippers).all():
        raise ValueError("Robot state must contain only finite values")
    if np.any((grippers < 0.0) | (grippers > 1.0)):
        raise ValueError("Gripper openness must be in [0, 1]")
    return np.concatenate(
        (arms[:7], grippers[:1], arms[7:], grippers[1:])
    ).astype(np.float32, copy=False)


def policy_arm_positions(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state).reshape(ACTION_DIM)
    return np.concatenate((state[LEFT_ARM_SLICE], state[RIGHT_ARM_SLICE]))


def policy_gripper_positions(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state).reshape(ACTION_DIM)
    return state[[LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]]


def make_policy_observation(
    image_rgb: np.ndarray, state: np.ndarray, language: str
) -> dict[str, Any]:
    """Create the nested GR00T server observation used by the reference script."""

    image = np.asarray(image_rgb)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if image.shape != (ROI_HEIGHT, ROI_WIDTH, 3) or image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 policy image ({ROI_HEIGHT}, {ROI_WIDTH}, 3), "
            f"got shape={image.shape} dtype={image.dtype}"
        )
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM}-D state, got {state.shape}")
    return {
        "video": {"ego_view": image[np.newaxis, np.newaxis, :]},
        "state": {
            "left_arm": state[LEFT_ARM_SLICE].reshape(1, 1, 7),
            "left_gripper": state[LEFT_GRIPPER_INDEX : LEFT_GRIPPER_INDEX + 1].reshape(
                1, 1, 1
            ),
            "right_arm": state[RIGHT_ARM_SLICE].reshape(1, 1, 7),
            "right_gripper": state[
                RIGHT_GRIPPER_INDEX : RIGHT_GRIPPER_INDEX + 1
            ].reshape(1, 1, 1),
        },
        "language": {"task_description": [[language]]},
    }


def clamp_policy_action(
    model: mujoco.MjModel, bindings: JointBindings, action: np.ndarray
) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError(f"Policy action must be finite and {ACTION_DIM}-D")
    result = action.copy()
    clipped = False
    arm_action_indices = (*range(7), *range(8, 15))
    for action_index, joint_id in zip(
        arm_action_indices, bindings.joint_ids, strict=True
    ):
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            bounded = float(np.clip(result[action_index], lower, upper))
            clipped = clipped or bounded != result[action_index]
            result[action_index] = bounded
    gripper_indices = (LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX)
    bounded_grippers = np.clip(result[list(gripper_indices)], 0.0, 1.0)
    clipped = clipped or not np.array_equal(
        bounded_grippers, result[list(gripper_indices)]
    )
    result[list(gripper_indices)] = bounded_grippers
    return result.astype(np.float32), clipped


def robot_state(
    data: mujoco.MjData, bindings: JointBindings, gripper_openness: np.ndarray
) -> np.ndarray:
    return compose_policy_state(
        data.qpos[bindings.qpos_addresses], np.asarray(gripper_openness)
    )


def write_kinematic_robot_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: JointBindings,
    target: np.ndarray,
    upper_body_pose: tuple[tuple[str, float], ...],
) -> None:
    """Hold robot joints at a target without constraining dynamic scene objects."""

    target = np.asarray(target, dtype=float).reshape(ACTION_DIM)
    data.qpos[bindings.qpos_addresses] = policy_arm_positions(target)
    data.qvel[bindings.dof_addresses] = 0.0
    set_gripper_command(
        model, data, policy_gripper_positions(target), np.zeros(GRIPPER_DIM)
    )
    set_upper_body_pose(model, data, upper_body_pose)


def step_kinematic_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bindings: JointBindings,
    target: np.ndarray,
    upper_body_pose: tuple[tuple[str, float], ...],
    duration_s: float,
) -> None:
    substeps = max(1, int(round(duration_s / float(model.opt.timestep))))
    for _ in range(substeps):
        write_kinematic_robot_pose(model, data, bindings, target, upper_body_pose)
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
    write_kinematic_robot_pose(model, data, bindings, target, upper_body_pose)
    mujoco.mj_forward(model, data)


def die_is_placed(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Return true when the whole die footprint is inside the cardboard box."""

    die_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box")
    if min(die_id, box_id) < 0:
        return False
    rotation = data.xmat[box_id].reshape(3, 3)
    local = rotation.T @ (data.xpos[die_id] - data.xpos[box_id])
    inside_xy = abs(local[0]) <= 0.092 and abs(local[1]) <= 0.052
    inside_z = 0.020 <= local[2] <= 0.065
    die_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dof_address = int(model.jnt_dofadr[die_joint])
    settled = float(np.linalg.norm(data.qvel[dof_address : dof_address + 3])) < 0.15
    return bool(inside_xy and inside_z and settled)


class VideoWriter:
    def __init__(self, path: Path, fps: float):
        import av

        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=max(1, round(fps)))
        self.stream.width = ROI_WIDTH
        self.stream.height = ROI_HEIGHT
        self.stream.pix_fmt = "yuv420p"
        self.closed = False

    def append(self, image_rgb: np.ndarray) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(image_rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.closed = True


def import_policy_client():
    try:
        from gr00t_utils.server_client import PolicyClient
    except ImportError as error:
        raise RuntimeError(
            "GR00T client dependencies are missing. Install them with:\n"
            "  uv pip install --python .venv/bin/python "
            "-r requirements-gr00t-client.txt"
        ) from error
    return PolicyClient


def render_policy_image(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    scene_option: mujoco.MjvOption,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    renderer.update_scene(data, camera=HEAD_CAMERA_NAME, scene_option=scene_option)
    image = renderer.render()
    if maps is not None:
        image = cv2.remap(
            image,
            maps[0],
            maps[1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    return crop_head_camera_roi(image)


def save_actions(path: Path, actions: list[np.ndarray]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        np.stack(actions).astype(np.float32)
        if actions
        else np.empty((0, ACTION_DIM), dtype=np.float32)
    )
    if path.suffix.lower() == ".csv":
        names = [
            *[f"left_arm_{index + 1}" for index in range(7)],
            "left_gripper_openness",
            *[f"right_arm_{index + 1}" for index in range(7)],
            "right_gripper_openness",
        ]
        np.savetxt(path, array, delimiter=",", header=",".join(names), comments="")
    elif path.suffix.lower() == ".npz":
        np.savez_compressed(path, action=array)
    else:
        np.save(path, array)


def main() -> int:
    args = parse_args()
    if args.frequency <= 0 or args.max_steps <= 0:
        raise ValueError("frequency and max-steps must be positive")
    if args.replan_steps < 0 or args.warmup_steps < 0:
        raise ValueError("replan-steps and warmup-steps must be non-negative")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    records = [item for item in manifest["episodes"] if item.get("status") == "ok"]
    if not 0 <= args.episode_index < len(records):
        raise ValueError(f"episode-index must be in [0, {len(records) - 1}]")
    record = records[args.episode_index]
    dataset_dir = Path(manifest["dataset_dir"])
    summary_path = Path(manifest["summary"])
    cache_path = (manifest_path.parent / record["cache"]).resolve()
    trajectory = load_corrected_trajectory(
        dataset_dir / record["episode"], summary_path, cache_path
    )

    model = mujoco.MjModel.from_xml_path(manifest["model"])
    model.vis.global_.offwidth = HEAD_CAMERA_WIDTH
    model.vis.global_.offheight = HEAD_CAMERA_HEIGHT
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME) < 0:
        raise ValueError(f"Model has no camera named {HEAD_CAMERA_NAME!r}")
    hide_closed_head_shell(model)
    apply_texture_gamma(model, "cardboard_box_texture", args.box_texture_gamma)
    set_target_visibility(model, False)
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)
    torso = manifest["fixed_torso"]
    upper_body_pose = with_head_pose(
        fixed_upper_body_pose(
            float(torso["body_lift_m"]), float(torso["body_pitch_rad"])
        ),
        args.head_yaw_deg,
        args.head_pitch_deg,
    )
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
    if dice_plan is None or trajectory.effector_positions is None:
        raise ValueError(f"Could not initialize die/grippers for {record['episode']}")

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    apply_kinematic_pose(
        model,
        data,
        trajectory,
        bindings,
        0.0,
        show_target=False,
        dice_plan=dice_plan,
        upper_body_pose=upper_body_pose,
    )
    current_target = compose_policy_state(
        trajectory.joint_positions[0], trajectory.effector_positions[0]
    )
    dice_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dice_dof = int(model.jnt_dofadr[dice_joint])
    data.qvel[dice_dof : dice_dof + 6] = 0.0
    write_kinematic_robot_pose(model, data, bindings, current_target, upper_body_pose)
    mujoco.mj_forward(model, data)

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = 0
    scene_option.geomgroup[1] = 1
    scene_option.geomgroup[2] = 1
    scene_option.geomgroup[CAMERA_OCCLUDER_GROUP] = 0
    renderer = mujoco.Renderer(
        model, height=HEAD_CAMERA_HEIGHT, width=HEAD_CAMERA_WIDTH
    )
    maps = None if args.no_distortion else distortion_maps()
    first_image = render_policy_image(renderer, data, scene_option, maps)
    first_state = robot_state(
        data, bindings, policy_gripper_positions(current_target)
    )
    first_observation = make_policy_observation(first_image, first_state, args.language)
    print(
        f"[MUJOCO_VLA] layout={record['episode']} box="
        f"({box['x']:.4f}, {box['y']:.4f}, {box['yaw_deg']:.1f} deg)",
        flush=True,
    )
    print(
        f"[MUJOCO_VLA] image={first_image.shape} state={first_state.shape} "
        f"frequency={args.frequency:g} Hz language={args.language!r}",
        flush=True,
    )
    if args.dry_run:
        del first_observation
        renderer.close()
        print("[MUJOCO_VLA] dry-run passed; no policy request was sent", flush=True)
        return 0

    try:
        PolicyClient = import_policy_client()
    except BaseException:
        renderer.close()
        raise
    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        strict=False,
    )
    print(
        f"[MUJOCO_VLA] connecting policy tcp://{args.policy_host}:{args.policy_port}",
        flush=True,
    )
    try:
        policy.reset({"language": args.language})
    except BaseException:
        policy.close()
        renderer.close()
        raise

    viewer = None
    controls = {"paused": not args.start_immediately}

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            controls["paused"] = not controls["paused"]
            print(
                f"[MUJOCO_VLA] {'paused' if controls['paused'] else 'resumed'}",
                flush=True,
            )

    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(
            model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
        )
    show_head = not args.headless and not args.no_head_window
    if show_head:
        cv2.namedWindow(HEAD_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(HEAD_WINDOW, ROI_WIDTH, ROI_HEIGHT)
    writer = VideoWriter(args.save_video, args.frequency) if args.save_video else None
    action_queue: deque[np.ndarray] = deque()
    executed_actions: list[np.ndarray] = []
    success_count = 0
    step_index = 0
    aborted = False
    print(
        "[MUJOCO_VLA] controls: SPACE pause/resume, Q/ESC quit; "
        f"initially {'running' if args.start_immediately else 'paused'}",
        flush=True,
    )

    try:
        for _ in range(args.warmup_steps):
            step_kinematic_robot(
                model,
                data,
                bindings,
                current_target,
                upper_body_pose,
                1.0 / args.frequency,
            )

        while step_index < args.max_steps:
            loop_start = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                break
            image = render_policy_image(renderer, data, scene_option, maps)
            if show_head:
                cv2.imshow(HEAD_WINDOW, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    aborted = True
                    break
                if key == ord(" "):
                    key_callback(ord(" "))
            if viewer is not None:
                viewer.sync()
            if controls["paused"]:
                time.sleep(0.01)
                continue

            if writer is not None:
                writer.append(image)
            state = robot_state(
                data, bindings, policy_gripper_positions(current_target)
            )
            if not action_queue:
                observation = make_policy_observation(image, state, args.language)
                action_response, _ = policy.get_action(observation)
                chunk = list(iter_action_chunk(action_response))
                if args.replan_steps:
                    chunk = chunk[: args.replan_steps]
                if not chunk:
                    raise RuntimeError("Policy returned an empty action chunk")
                action_queue.extend(chunk)
                print(
                    f"[MUJOCO_VLA] inference step={step_index} chunk={len(chunk)}",
                    flush=True,
                )

            proposed = action_queue.popleft()
            current_target, clipped = clamp_policy_action(model, bindings, proposed)
            if clipped:
                print(
                    f"[MUJOCO_VLA] warning: clipped out-of-range action at step {step_index}",
                    flush=True,
                )
            executed_actions.append(current_target.copy())
            if args.debug_actions:
                print(
                    f"[MUJOCO_VLA] action[{step_index}]="
                    f"{np.array2string(current_target, precision=4)}",
                    flush=True,
                )
            step_kinematic_robot(
                model,
                data,
                bindings,
                current_target,
                upper_body_pose,
                1.0 / args.frequency,
            )
            step_index += 1
            success_count = success_count + 1 if die_is_placed(model, data) else 0
            if args.success_hold_steps > 0 and success_count >= args.success_hold_steps:
                print(f"[MUJOCO_VLA] success at step {step_index}", flush=True)
                break
            if not args.no_realtime:
                remaining = 1.0 / args.frequency - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if writer is not None:
            writer.close()
        if args.save_actions:
            save_actions(args.save_actions, executed_actions)
        if viewer is not None:
            viewer.close()
        if show_head:
            cv2.destroyWindow(HEAD_WINDOW)
        renderer.close()
        policy.close()

    success = success_count >= args.success_hold_steps > 0
    print(
        f"[MUJOCO_VLA] finished steps={step_index} success={success} aborted={aborted}",
        flush=True,
    )
    return 0 if success or aborted else 2


if __name__ == "__main__":
    raise SystemExit(main())
