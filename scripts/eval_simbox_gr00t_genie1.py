#!/usr/bin/env python3
"""Roll out a GR00T/VLA policy in the SimBox Genie1 lab task.

This is the Genie1 counterpart of ``eval_simbox_gr00t.py``.  It keeps the
SimBox/Nimbus scene setup unchanged and only swaps the planner trajectory for
actions returned by a GR00T policy server.

Genie1 uses a 16-D state/action layout:

    [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]

Example:
    python aloha_lerobot/eval_simbox_gr00t_genie1.py \
        --config configs/simbox/de_lab_task_genie1.yaml \
        --policy-host localhost \
        --policy-port 5555
    python aloha_lerobot/eval_simbox_gr00t_genie1.py \
        --config configs/simbox/de_lab_task_genie1.yaml \
        --save-video logs/vla_genie1_left.mp4 \
        --save-video-camera hand_left
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM = 16
ARM_DIM = 7


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_local_paths() -> None:
    root = _repo_root()
    local_lerobot_src = root / "aloha_lerobot" / "lerobot" / "src"
    for path in (root, local_lerobot_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _patch_scipy_rotation_scalar_first() -> None:
    """Match launcher.py's compatibility patch for older scipy builds."""
    try:
        from scipy.spatial.transform import Rotation as _R

        _R.from_quat([1, 0, 0, 0], scalar_first=True)
    except TypeError:
        import scipy.spatial.transform as _sst
        from scipy.spatial.transform import Rotation as _OrigR

        class Rotation(_OrigR):
            @classmethod
            def from_quat(cls, quat, *args, scalar_first=False, **kwargs):
                quat = np.asarray(quat, dtype=float)
                if scalar_first:
                    if quat.ndim == 1:
                        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
                    else:
                        quat = np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)
                return super().from_quat(quat, *args, **kwargs)

            def as_quat(self, *args, scalar_first=False, **kwargs):
                quat = super().as_quat(*args, **kwargs)
                if scalar_first:
                    if quat.ndim == 1:
                        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
                    else:
                        quat = np.concatenate([quat[..., 3:], quat[..., :3]], axis=-1)
                return quat

        _sst.Rotation = Rotation
        sys.modules["scipy.spatial.transform._rotation"].Rotation = Rotation
        sys.modules["scipy.spatial.transform"].Rotation = Rotation


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image)) if image.size else 1.0
        if max_value <= 1.0:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _split_flat_action(flat_action: np.ndarray) -> dict[str, np.ndarray]:
    flat_action = np.asarray(flat_action, dtype=np.float32).reshape(-1)
    if flat_action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}-D Genie1 action, got {flat_action.shape[0]}")
    return {
        "left_arm": flat_action[0:7],
        "left_gripper": flat_action[7:8],
        "right_arm": flat_action[8:15],
        "right_gripper": flat_action[15:16],
    }


def _format_action_debug(step_idx: int, flat_action: np.ndarray) -> str:
    parts = _split_flat_action(flat_action)
    formatted = {
        key: np.array2string(value, precision=5, suppress_small=False)
        for key, value in parts.items()
    }
    return (
        f"[VLA_ACTION] step={step_idx} "
        f"left_arm={formatted['left_arm']} "
        f"left_gripper={formatted['left_gripper']} "
        f"right_arm={formatted['right_arm']} "
        f"right_gripper={formatted['right_gripper']}"
    )


def _save_flat_actions(save_path: Path, flat_actions: list[np.ndarray]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    actions = (
        np.stack([np.asarray(action, dtype=np.float32).reshape(ACTION_DIM) for action in flat_actions])
        if flat_actions
        else np.empty((0, ACTION_DIM), dtype=np.float32)
    )
    suffix = save_path.suffix.lower()
    if suffix == ".csv":
        header = ",".join(
            [
                *[f"left_arm_{idx}" for idx in range(ARM_DIM)],
                "left_gripper",
                *[f"right_arm_{idx}" for idx in range(ARM_DIM)],
                "right_gripper",
            ]
        )
        np.savetxt(save_path, actions, delimiter=",", header=header, comments="")
    elif suffix == ".npz":
        parts = {
            key: np.stack([_split_flat_action(action)[key] for action in flat_actions])
            if flat_actions
            else np.empty((0, 0), dtype=np.float32)
            for key in ("left_arm", "left_gripper", "right_arm", "right_gripper")
        }
        np.savez_compressed(save_path, flat_action=actions, **parts)
    else:
        np.save(save_path, actions)
    print(f"[VLA_EVAL] saved flat actions: {save_path} shape={actions.shape}")


def _save_rgb_video(save_path: Path, frames: list[np.ndarray], fps: float, camera_key: str) -> None:
    if not frames:
        print(f"[VLA_EVAL] no camera frames to save: camera={camera_key} path={save_path}")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    imageio.mimsave(save_path, [np.asarray(frame, dtype=np.uint8) for frame in frames], fps=max(1, int(round(fps))))
    print(f"[VLA_EVAL] saved camera video: camera={camera_key} path={save_path} frames={len(frames)}")


class _StreamingRgbVideoWriter:
    """Encode RGB frames incrementally with PyAV instead of retaining them in RAM."""

    def __init__(self, save_path: Path, fps: float):
        import av

        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._av = av
        self._container = av.open(str(save_path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=max(1, int(round(fps))))
        self._initialized = False
        self._closed = False

    def append_data(self, image: np.ndarray) -> None:
        image = _as_uint8_rgb(image)
        height, width = image.shape[:2]
        if not self._initialized:
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv420p"
            self._initialized = True
        elif (width, height) != (self._stream.width, self._stream.height):
            raise ValueError(
                f"Video frame size changed from {(self._stream.width, self._stream.height)} "
                f"to {(width, height)}"
            )

        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        if self._initialized:
            for packet in self._stream.encode():
                self._container.mux(packet)
        self._container.close()
        self._closed = True


def _open_rgb_video_writer(save_path: Path, fps: float) -> _StreamingRgbVideoWriter:
    """Open a streaming writer so multi-camera rollouts do not retain raw frames in RAM."""
    return _StreamingRgbVideoWriter(save_path, fps)


def _state_from_robot_obs(robot_obs: dict[str, Any]) -> np.ndarray:
    pieces = [
        robot_obs["states.left_joint.position"],
        robot_obs["states.left_gripper.position"],
        robot_obs["states.right_joint.position"],
        robot_obs["states.right_gripper.position"],
    ]
    return np.concatenate([np.asarray(piece, dtype=np.float32).reshape(-1) for piece in pieces])


def _make_policy_observation(
    obs: dict[str, Any],
    robot_name: str,
    language: str,
    camera_keys: tuple[str, str, str],
) -> dict[str, Any]:
    robot_obs = obs["robots"][robot_name]
    state = _state_from_robot_obs(robot_obs)
    if state.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected Genie1 state dim {ACTION_DIM}, got {state.shape[0]}")

    head_key, left_key, right_key = camera_keys
    cameras = obs.get("cameras", {})
    missing = [key for key in camera_keys if key not in cameras]
    if missing:
        raise KeyError(f"Missing camera observation(s): {missing}. Available: {sorted(cameras)}")

    left_arm = state[0:7]
    left_hand = state[7:8]
    right_arm = state[8:15]
    right_hand = state[15:16]

    return {
        "video": {
            "ego_view": _as_uint8_rgb(cameras[head_key]["color_image"])[np.newaxis, np.newaxis, :],
            "hand_left": _as_uint8_rgb(cameras[left_key]["color_image"])[np.newaxis, np.newaxis, :],
            "hand_right": _as_uint8_rgb(cameras[right_key]["color_image"])[np.newaxis, np.newaxis, :],
        },
        "state": {
            "left_arm": left_arm.reshape(1, 1, -1).astype(np.float32),
            "left_gripper": left_hand.reshape(1, 1, -1).astype(np.float32),
            "right_arm": right_arm.reshape(1, 1, -1).astype(np.float32),
            "right_gripper": right_hand.reshape(1, 1, -1).astype(np.float32),
        },
        "language": {
            "task_description": [[language]],
        },
    }


def _extract_array(action: dict[str, Any], key: str) -> np.ndarray:
    if key not in action:
        raise KeyError(f"Policy action missing key '{key}'. Available keys: {sorted(action)}")
    return np.asarray(action[key], dtype=np.float32)


def _extract_array_alias(action: dict[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in action:
            return np.asarray(action[key], dtype=np.float32)
    raise KeyError(f"Policy action missing any of {keys}. Available keys: {sorted(action)}")


def _normalize_action_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    if arr.ndim == 2:
        return arr.reshape(1, *arr.shape)
    return arr


def _iter_action_chunk(action: dict[str, Any]):
    """Yield flat 16-D Genie1 actions from common GR00T output layouts."""
    if "action" in action:
        flat = _normalize_action_array(np.asarray(action["action"], dtype=np.float32))
        if flat.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected flat Genie1 action dim {ACTION_DIM}, got shape {flat.shape}")
        for idx in range(flat.shape[1]):
            yield flat[0, idx]
        return

    left_arm = _extract_array(action, "left_arm")
    right_arm = _extract_array(action, "right_arm")
    left_hand = _extract_array_alias(action, "left_gripper", "left_hand")
    right_hand = _extract_array_alias(action, "right_gripper", "right_hand")

    arrays = [_normalize_action_array(arr) for arr in (left_arm, right_arm, left_hand, right_hand)]
    horizon = arrays[0].shape[1]
    if any(arr.shape[0] != 1 or arr.shape[1] != horizon for arr in arrays):
        raise ValueError(
            "Expected action arrays shaped (1, H, D) with the same horizon; "
            f"got {[arr.shape for arr in arrays]}"
        )
    for idx in range(horizon):
        flat = np.concatenate(
            [
                arrays[0][0, idx].reshape(-1),
                arrays[2][0, idx].reshape(-1),
                arrays[1][0, idx].reshape(-1),
                arrays[3][0, idx].reshape(-1),
            ]
        ).astype(np.float32)
        if flat.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected structured Genie1 action dim {ACTION_DIM}, got {flat.shape[0]}")
        yield flat


def _gripper_target(value: float, mode: str) -> float:
    value = float(value)
    if mode == "joint":
        return float(np.clip(value, 0.0, 1.0))
    if mode == "state_position":
        return float(np.clip(value * 0.5, 0.0, 1.0))
    if mode == "binary_openness":
        return 1.0 if value >= 0.5 else 0.0
    return float(np.clip(value, 0.0, 1.0))


def _sim_action_from_flat(
    flat_action: np.ndarray,
    robot_name: str,
    robot,
    gripper_action_mode: str,
) -> dict[str, dict[str, np.ndarray]]:
    flat_action = np.asarray(flat_action, dtype=np.float32).reshape(-1)
    if flat_action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM}-D Genie1 action, got {flat_action.shape[0]}")

    resolver = getattr(robot, "_resolve_gripper_indices_by_name", None)
    if callable(resolver):
        resolver()

    joint_positions = np.concatenate(
        [
            flat_action[0:7],
            flat_action[8:15],
            np.array(
                [
                    _gripper_target(flat_action[7], gripper_action_mode),
                    _gripper_target(flat_action[15], gripper_action_mode),
                ],
                dtype=np.float32,
            ),
        ]
    )
    joint_indices = np.asarray(
        robot.left_joint_indices
        + robot.right_joint_indices
        + robot.left_gripper_indices
        + robot.right_gripper_indices,
        dtype=np.int64,
    )
    return {
        robot_name: {
            "joint_positions": joint_positions,
            "joint_indices": joint_indices,
        }
    }


def _task_language(wf, fallback: str) -> str:
    if fallback:
        return fallback
    data_cfg = wf.task_cfg.get("data", {})
    return (
        data_cfg.get("language_instruction")
        or data_cfg.get("detailed_language_instruction")
        or getattr(wf.task, "language_instruction", "")
        or ""
    )


def _pick_tube_success(wf) -> bool | None:
    getter = getattr(wf.task, "get_failure_diagnostics", None)
    if not callable(getter):
        return None
    diagnostics = getter() or {}
    details = diagnostics.get("details", {})
    lift = details.get("tube_lift_from_anchor_z")
    if lift is None:
        return None
    threshold = float(wf.task_cfg.get("skills", [{}])[0].get("success_lift_th", 0.15))
    for skill_group in wf.task_cfg.get("skills", []):
        for robot_skills in skill_group.values():
            for lr_skill_dict in robot_skills:
                for skill_list in lr_skill_dict.values():
                    for skill_cfg in skill_list:
                        threshold = float(skill_cfg.get("lift_th", threshold))
                        break
    return float(lift) >= threshold


def _build_scene(config_path: str, cli_overrides: list[str], random_seed: int | None):
    from nimbus.utils.config_processor import ConfigProcessor
    from nimbus.utils.flags import set_random_seed
    from nimbus_extension.components.load.env_loader import EnvLoader
    from nimbus_extension.components.load.env_randomizer import EnvRandomizer

    if random_seed is not None:
        set_random_seed(int(random_seed))

    config = ConfigProcessor().process_config(config_path, cli_args=cli_overrides)
    load_cfg = config["load_stage"]
    scene_loader_cfg = load_cfg["scene_loader"]["args"]
    randomizer_cfg = load_cfg.get("layout_random_generator", {}).get("args", {})

    loader = EnvLoader(None, **scene_loader_cfg)
    randomizer = EnvRandomizer(
        loader,
        random_num=1,
        strict_mode=False,
        input_dir=randomizer_cfg.get("input_dir"),
    )
    scene = next(randomizer)
    return scene


def eval_policy(args: argparse.Namespace) -> int:
    _add_local_paths()
    _patch_scipy_rotation_scalar_first()

    from nimbus.utils.utils import init_env

    init_env()

    from aloha_lerobot.gr00t_utils.server_client import PolicyClient

    scene = _build_scene(args.config, args.config_override, args.random_seed)
    wf = scene.wf
    robot = wf.task.robots[args.robot_name]
    camera_keys = (args.head_camera, args.left_camera, args.right_camera)
    language = _task_language(wf, args.language)

    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        strict=False,
    )
    policy.reset({"language": language})

    action_queue: deque[np.ndarray] = deque()
    executed_flat_actions: list[np.ndarray] = []
    save_video_camera = args.save_video_camera or args.head_camera
    video_writer = _open_rgb_video_writer(args.save_video, args.frequency) if args.save_video else None
    head_video_writer = (
        _open_rgb_video_writer(args.save_head_video, args.frequency) if args.save_head_video else None
    )
    video_frame_count = 0
    head_video_frame_count = 0
    step_idx = 0
    success = None
    print(f"[VLA_EVAL] language={language!r}")
    print(f"[VLA_EVAL] cameras={camera_keys} robot={args.robot_name}")
    print(f"[VLA_EVAL] action_dim={ACTION_DIM} gripper_action_mode={args.gripper_action_mode}")
    print(
        f"[VLA_EVAL] video_outputs primary={args.save_video} "
        f"primary_camera={save_video_camera} head={args.save_head_video}",
        flush=True,
    )

    try:
        for _ in range(args.warmup_steps):
            wf.world.get_observations()
            wf.world.step(render=args.render)

        while step_idx < args.max_steps:
            loop_start = time.perf_counter()
            obs = wf.world.get_observations()
            if args.save_video:
                cameras = obs.get("cameras", {})
                if save_video_camera not in cameras:
                    raise KeyError(
                        f"Cannot save video: missing camera '{save_video_camera}'. "
                        f"Available: {sorted(cameras)}"
                    )
                video_writer.append_data(_as_uint8_rgb(cameras[save_video_camera]["color_image"]))
                video_frame_count += 1
            if args.save_head_video:
                cameras = obs.get("cameras", {})
                if args.head_camera not in cameras:
                    raise KeyError(
                        f"Cannot save head video: missing camera '{args.head_camera}'. "
                        f"Available: {sorted(cameras)}"
                    )
                head_video_writer.append_data(_as_uint8_rgb(cameras[args.head_camera]["color_image"]))
                head_video_frame_count += 1

            if not action_queue:
                policy_obs = _make_policy_observation(obs, args.robot_name, language, camera_keys)
                action, info = policy.get_action(policy_obs)
                del info
                action_queue.extend(_iter_action_chunk(action))
                if not action_queue:
                    raise RuntimeError("Policy returned an empty action horizon.")

            flat_action = action_queue.popleft()
            executed_flat_actions.append(np.asarray(flat_action, dtype=np.float32).reshape(ACTION_DIM).copy())
            if args.debug_actions:
                print(_format_action_debug(step_idx, flat_action), flush=True)

            sim_action = _sim_action_from_flat(flat_action, args.robot_name, robot, args.gripper_action_mode)
            wf.task.apply_action(sim_action)
            wf.world.step(render=args.render)

            step_idx += 1
            if args.success_check_interval > 0 and step_idx % args.success_check_interval == 0:
                success = _pick_tube_success(wf)
                if success:
                    print(f"[VLA_EVAL] success detected at step {step_idx}")
                    break

            sleep_time = (1.0 / args.frequency) - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if success is None:
            try:
                success = _pick_tube_success(wf)
            except Exception:
                success = None
        try:
            policy.socket.close()
            policy.context.term()
        except Exception:
            pass
        if args.save_path:
            try:
                _save_flat_actions(args.save_path, executed_flat_actions)
            except Exception as exc:
                print(f"[VLA_EVAL] failed to save flat actions to {args.save_path}: {exc}", flush=True)
        if video_writer is not None:
            try:
                video_writer.close()
                print(
                    f"[VLA_EVAL] saved camera video: camera={save_video_camera} "
                    f"path={args.save_video} frames={video_frame_count}"
                )
            except Exception as exc:
                print(
                    f"[VLA_EVAL] failed to save camera video "
                    f"camera={save_video_camera} path={args.save_video}: {exc}",
                    flush=True,
                )
        if head_video_writer is not None:
            try:
                head_video_writer.close()
                print(
                    f"[VLA_EVAL] saved camera video: camera={args.head_camera} "
                    f"path={args.save_head_video} frames={head_video_frame_count}"
                )
            except Exception as exc:
                print(
                    f"[VLA_EVAL] failed to save head camera video "
                    f"camera={args.head_camera} path={args.save_head_video}: {exc}",
                    flush=True,
                )
        if not args.keep_app_open:
            scene.simulation_app.close()

    print(f"[VLA_EVAL] finished steps={step_idx} success={success}")
    return 0 if success is not False else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simbox/de_lab_task_genie1.yaml")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Nimbus config override. Unknown --a.b=1 arguments are also forwarded.",
    )
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--policy-timeout-ms", type=int, default=30000)
    parser.add_argument("--language", default="", help="Override task language. Empty means use the task config.")
    parser.add_argument("--robot-name", default="genie1")
    parser.add_argument("--head-camera", default="head")
    parser.add_argument("--left-camera", default="hand_left")
    parser.add_argument("--right-camera", default="hand_right")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--frequency", type=float, default=15.0)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--gripper-action-mode",
        choices=("openness", "binary_openness", "state_position", "joint"),
        default="state_position",
        help=(
            "How to map the policy's gripper scalar to Genie1 joint targets. "
            "'state_position' maps SimBox logged gripper state qpos*2 back to qpos and "
            "matches convert_genie1_lmdb_to_lerobot.py's default action schema."
        ),
    )
    parser.add_argument("--success-check-interval", type=int, default=10)
    parser.add_argument(
        "--debug-actions",
        "--debug",
        dest="debug_actions",
        action="store_true",
        help="Print each executed flat action split into left/right arm and gripper components.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Save executed flat_action sequence. Supported suffixes: .npy, .npz, .csv.",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Save the executed rollout's RGB frames as a video, e.g. logs/vla_genie1_head.mp4.",
    )
    parser.add_argument(
        "--save-video-camera",
        default=None,
        help="Camera key to save with --save-video. Defaults to --head-camera.",
    )
    parser.add_argument(
        "--save-head-video",
        type=Path,
        default=None,
        help="Also save the head camera RGB frames to this video path.",
    )
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--keep-app-open", action="store_true")
    parser.set_defaults(render=True)
    args, unknown = parser.parse_known_args()
    args.config_override.extend(unknown)
    return args


def main() -> int:
    return eval_policy(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
