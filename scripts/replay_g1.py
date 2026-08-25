#!/usr/bin/env python3
"""Replay fixed_spine3_to_g1 joint trajectories in native MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_g1_to_mjcf import (  # noqa: E402
    DEFAULT_MJCF,
    DEFAULT_URDF,
    convert_urdf_to_mjcf,
)
from scripts.convert_a2d_to_mjcf import (  # noqa: E402
    A2D_ARM_JOINT_NAMES,
    DEFAULT_A2D_MJCF,
    DEFAULT_A2D_URDF,
    convert_a2d_urdf_to_mjcf,
)


DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "fixed_spine3_to_g1"
DEFAULT_EPISODE = DEFAULT_DATASET_DIR / "episode_000000.npz"
DEFAULT_SUMMARY = DEFAULT_DATASET_DIR / "retarget_summary.json"

GRIPPER_MAIN_JOINTS = (
    ("idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"),
    ("idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"),
)
GRIPPER_PASSIVE_JOINTS = (
    "idx32_gripper_l_inner_joint3",
    "idx33_gripper_l_inner_joint4",
    "idx39_gripper_l_inner_joint0",
    "idx42_gripper_l_outer_joint3",
    "idx43_gripper_l_outer_joint4",
    "idx49_gripper_l_outer_joint0",
    "idx72_gripper_r_inner_joint3",
    "idx73_gripper_r_inner_joint4",
    "idx79_gripper_r_inner_joint0",
    "idx82_gripper_r_outer_joint3",
    "idx83_gripper_r_outer_joint4",
    "idx89_gripper_r_outer_joint0",
)
A2D_GRIPPER_JOINTS = tuple(
    f"{side}_{finger}{link}_joint"
    for side in ("left", "right")
    for finger in ("narrow", "wide")
    for link in (1, 2, 3, 4)
)


@dataclass(frozen=True)
class RobotReplayConfig:
    key: str
    default_urdf: Path
    default_mjcf: Path
    model_joint_names: tuple[str, ...] | None
    arm_base_body: str
    eef_body_names: tuple[str, str]
    gripper_mode: str


G1_ROBOT_CONFIG = RobotReplayConfig(
    key="g1",
    default_urdf=DEFAULT_URDF,
    default_mjcf=DEFAULT_MJCF,
    model_joint_names=None,
    arm_base_body="arm_base_link",
    eef_body_names=("arm_l_end_link", "arm_r_end_link"),
    gripper_mode="g1_mimic",
)
A2D_ROBOT_CONFIG = RobotReplayConfig(
    key="a2d",
    default_urdf=DEFAULT_A2D_URDF,
    default_mjcf=DEFAULT_A2D_MJCF,
    model_joint_names=A2D_ARM_JOINT_NAMES,
    arm_base_body="link-arm",
    eef_body_names=("Link7_l", "Link7_r"),
    gripper_mode="a2d_neutral",
)
ROBOT_CONFIGS = {config.key: config for config in (G1_ROBOT_CONFIG, A2D_ROBOT_CONFIG)}


@dataclass(frozen=True)
class Trajectory:
    times_s: np.ndarray
    joint_positions: np.ndarray
    joint_names: tuple[str, ...]
    target_eef_wxyz: np.ndarray
    achieved_eef_wxyz: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.joint_positions.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])


@dataclass(frozen=True)
class JointBindings:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray


def load_trajectory(episode_path: Path, summary_path: Path) -> Trajectory:
    episode_path = episode_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    joint_names = tuple(summary["joint_order"])

    required = {
        "local_timestamps_ns",
        "action_joint_position",
        "target_eef_wxyz",
        "achieved_eef_wxyz",
    }
    with np.load(episode_path, allow_pickle=False) as episode:
        missing = sorted(required.difference(episode.files))
        if missing:
            raise ValueError(f"Episode is missing arrays: {missing}")
        timestamps_ns = np.asarray(episode["local_timestamps_ns"], dtype=np.int64)
        joint_positions = np.asarray(episode["action_joint_position"], dtype=float)
        target_eef = np.asarray(episode["target_eef_wxyz"], dtype=float)
        achieved_eef = np.asarray(episode["achieved_eef_wxyz"], dtype=float)

    if timestamps_ns.ndim != 1 or timestamps_ns.size < 2:
        raise ValueError("local_timestamps_ns must contain at least two timestamps")
    frames = timestamps_ns.size
    if joint_positions.shape != (frames, len(joint_names)):
        raise ValueError(
            "action_joint_position shape does not match timestamps and joint_order: "
            f"{joint_positions.shape} vs ({frames}, {len(joint_names)})"
        )
    if target_eef.shape != (frames, 14) or achieved_eef.shape != (frames, 14):
        raise ValueError("EEF arrays must have shape (frames, 14)")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("local_timestamps_ns must be strictly increasing")
    if not all(
        np.isfinite(array).all()
        for array in (joint_positions, target_eef, achieved_eef)
    ):
        raise ValueError("Trajectory contains NaN or infinite values")

    times_s = (timestamps_ns - timestamps_ns[0]).astype(float) * 1e-9
    return Trajectory(
        times_s=times_s,
        joint_positions=joint_positions,
        joint_names=joint_names,
        target_eef_wxyz=target_eef,
        achieved_eef_wxyz=achieved_eef,
    )


def bind_joints(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    model_joint_names: tuple[str, ...] | None = None,
) -> JointBindings:
    joint_ids: list[int] = []
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    missing: list[str] = []

    names_to_bind = model_joint_names if model_joint_names is not None else joint_names
    if len(names_to_bind) != len(joint_names):
        raise ValueError(
            "model_joint_names and trajectory joint_names must have the same length"
        )

    for source_name, name in zip(joint_names, names_to_bind, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            missing.append(f"{source_name} -> {name}")
            continue
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Replay joint {name!r} is not a scalar joint")
        joint_ids.append(joint_id)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))

    if missing:
        raise ValueError(f"Model is missing replay joints: {missing}")
    return JointBindings(
        joint_ids=np.asarray(joint_ids, dtype=int),
        qpos_addresses=np.asarray(qpos_addresses, dtype=int),
        dof_addresses=np.asarray(dof_addresses, dtype=int),
    )


def validate_joint_limits(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    *,
    tolerance: float = 1e-8,
) -> None:
    violations: list[str] = []
    for column, (name, joint_id) in enumerate(
        zip(trajectory.joint_names, bindings.joint_ids, strict=True)
    ):
        if not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        observed_min = float(np.min(trajectory.joint_positions[:, column]))
        observed_max = float(np.max(trajectory.joint_positions[:, column]))
        if observed_min < lower - tolerance or observed_max > upper + tolerance:
            violations.append(
                f"{name}: observed [{observed_min:.6f}, {observed_max:.6f}], "
                f"limit [{lower:.6f}, {upper:.6f}]"
            )
    if violations:
        raise ValueError("Joint limit violations:\n" + "\n".join(violations))


def _joint_address(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Model is missing joint {joint_name!r}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def set_gripper_opening(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    normalized_opening: float,
    robot_config: RobotReplayConfig = G1_ROBOT_CONFIG,
) -> None:
    if not 0.0 <= normalized_opening <= 1.0:
        raise ValueError("gripper_open must be in [0, 1]")
    if robot_config.gripper_mode == "a2d_neutral":
        # A2D.urdf exposes the eight links of each physical four-bar gripper as
        # independent tree joints and contains no loop/mimic constraints. Keep
        # its authored neutral pose rather than tearing the mechanism apart.
        for name in A2D_GRIPPER_JOINTS:
            qpos_address, dof_address = _joint_address(model, name)
            data.qpos[qpos_address] = 0.0
            data.qvel[dof_address] = 0.0
        return
    if robot_config.gripper_mode != "g1_mimic":
        raise ValueError(f"Unknown gripper mode: {robot_config.gripper_mode}")

    angle = normalized_opening * np.pi / 4.0
    for inner_name, outer_name in GRIPPER_MAIN_JOINTS:
        inner_qpos, inner_dof = _joint_address(model, inner_name)
        outer_qpos, outer_dof = _joint_address(model, outer_name)
        data.qpos[inner_qpos] = -angle
        data.qpos[outer_qpos] = angle
        data.qvel[inner_dof] = 0.0
        data.qvel[outer_dof] = 0.0

    for name in GRIPPER_PASSIVE_JOINTS:
        qpos_address, dof_address = _joint_address(model, name)
        data.qpos[qpos_address] = 0.0
        data.qvel[dof_address] = 0.0


def interpolate_joint_state(
    trajectory: Trajectory, trajectory_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    time_s = float(np.clip(trajectory_time_s, 0.0, trajectory.duration_s))
    right = int(np.searchsorted(trajectory.times_s, time_s, side="right"))
    left = max(0, min(right - 1, trajectory.frames - 2))
    right = left + 1
    interval = trajectory.times_s[right] - trajectory.times_s[left]
    alpha = float(np.clip((time_s - trajectory.times_s[left]) / interval, 0.0, 1.0))
    q0 = trajectory.joint_positions[left]
    q1 = trajectory.joint_positions[right]
    return (1.0 - alpha) * q0 + alpha * q1, (q1 - q0) / interval


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    return result / np.linalg.norm(result)


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float).copy()
    result[1:] *= -1.0
    return result


def _quat_interpolate(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path normalized interpolation for nearby unit quaternions."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.dot(first, second) < 0.0:
        second = -second
    result = (1.0 - alpha) * first + alpha * second
    return result / np.linalg.norm(result)


def _pose_in_parent_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    parent_body_name: str,
    child_body_name: str,
) -> np.ndarray:
    parent_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, parent_body_name
    )
    child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_body_name)
    if parent_id < 0 or child_id < 0:
        raise ValueError(f"Missing FK body: {parent_body_name} or {child_body_name}")

    parent_rotation = data.xmat[parent_id].reshape(3, 3)
    local_position = parent_rotation.T @ (data.xpos[child_id] - data.xpos[parent_id])
    local_quaternion = _quat_multiply(
        _quat_conjugate(data.xquat[parent_id]), data.xquat[child_id]
    )
    return np.concatenate((local_position, local_quaternion))


def _set_target_mocap_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_local_pose: np.ndarray,
    target_body_name: str,
    arm_base_body_name: str,
) -> None:
    arm_base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, arm_base_body_name
    )
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
    )
    if arm_base_id < 0 or target_body_id < 0:
        raise ValueError(
            f"Model is missing {arm_base_body_name} or a target mocap body"
        )
    mocap_id = int(model.body_mocapid[target_body_id])
    if mocap_id < 0:
        raise ValueError(f"Body {target_body_name!r} is not a mocap body")

    arm_base_rotation = data.xmat[arm_base_id].reshape(3, 3)
    data.mocap_pos[mocap_id] = (
        data.xpos[arm_base_id] + arm_base_rotation @ target_local_pose[:3]
    )
    data.mocap_quat[mocap_id] = _quat_multiply(
        data.xquat[arm_base_id], target_local_pose[3:]
    )


def set_target_visibility(model: mujoco.MjModel, visible: bool) -> None:
    alpha = 0.9 if visible else 0.0
    for name in ("left_eef_target", "right_eef_target"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            model.site_rgba[site_id, 3] = alpha


def apply_kinematic_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: Trajectory,
    bindings: JointBindings,
    trajectory_time_s: float,
    *,
    gripper_open: float,
    show_target: bool,
    robot_config: RobotReplayConfig = G1_ROBOT_CONFIG,
) -> None:
    joint_positions, joint_velocities = interpolate_joint_state(
        trajectory, trajectory_time_s
    )
    data.qpos[bindings.qpos_addresses] = joint_positions
    data.qvel[:] = 0.0
    data.qvel[bindings.dof_addresses] = joint_velocities
    set_gripper_opening(model, data, gripper_open, robot_config)
    data.time = float(np.clip(trajectory_time_s, 0.0, trajectory.duration_s))
    mujoco.mj_forward(model, data)

    if show_target:
        right = int(
            np.clip(
                np.searchsorted(trajectory.times_s, data.time, side="left"),
                0,
                trajectory.frames - 1,
            )
        )
        left = max(0, right - 1)
        if left == right:
            target = trajectory.target_eef_wxyz[right]
        else:
            interval = trajectory.times_s[right] - trajectory.times_s[left]
            alpha = (data.time - trajectory.times_s[left]) / interval
            target = trajectory.target_eef_wxyz[left].copy()
            target[:3] = (
                (1.0 - alpha) * trajectory.target_eef_wxyz[left, :3]
                + alpha * trajectory.target_eef_wxyz[right, :3]
            )
            target[7:10] = (
                (1.0 - alpha) * trajectory.target_eef_wxyz[left, 7:10]
                + alpha * trajectory.target_eef_wxyz[right, 7:10]
            )
            target[3:7] = _quat_interpolate(
                trajectory.target_eef_wxyz[left, 3:7],
                trajectory.target_eef_wxyz[right, 3:7],
                alpha,
            )
            target[10:14] = _quat_interpolate(
                trajectory.target_eef_wxyz[left, 10:14],
                trajectory.target_eef_wxyz[right, 10:14],
                alpha,
            )
        _set_target_mocap_pose(
            model,
            data,
            target[:7],
            "left_eef_target_body",
            robot_config.arm_base_body,
        )
        _set_target_mocap_pose(
            model,
            data,
            target[7:],
            "right_eef_target_body",
            robot_config.arm_base_body,
        )
        mujoco.mj_forward(model, data)


def _quaternion_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(cosine)))


def evaluate_trajectory(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    *,
    gripper_open: float,
    robot_config: RobotReplayConfig = G1_ROBOT_CONFIG,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    position_errors: dict[str, list[float]] = {"left": [], "right": []}
    orientation_errors: dict[str, list[float]] = {"left": [], "right": []}
    max_contacts = 0

    for frame, trajectory_time_s in enumerate(trajectory.times_s):
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            float(trajectory_time_s),
            gripper_open=gripper_open,
            show_target=False,
            robot_config=robot_config,
        )
        max_contacts = max(max_contacts, int(data.ncon))
        for side, body_name, pose_slice in (
            ("left", robot_config.eef_body_names[0], slice(0, 7)),
            ("right", robot_config.eef_body_names[1], slice(7, 14)),
        ):
            actual = _pose_in_parent_frame(
                model, data, robot_config.arm_base_body, body_name
            )
            expected = trajectory.achieved_eef_wxyz[frame, pose_slice]
            position_errors[side].append(float(np.linalg.norm(actual[:3] - expected[:3])))
            orientation_errors[side].append(
                _quaternion_error_degrees(actual[3:], expected[3:])
            )

    return {
        "frames": trajectory.frames,
        "duration_s": trajectory.duration_s,
        "source_fps": (trajectory.frames - 1) / trajectory.duration_s,
        "fk_position_error_m": {
            side: {
                "mean": float(np.mean(errors)),
                "max": float(np.max(errors)),
            }
            for side, errors in position_errors.items()
        },
        "fk_orientation_error_deg": {
            side: {
                "mean": float(np.mean(errors)),
                "max": float(np.max(errors)),
            }
            for side, errors in orientation_errors.items()
        },
        "max_contacts": max_contacts,
    }


def ensure_model(
    model_path: Path,
    urdf_path: Path,
    rebuild: bool,
    robot_config: RobotReplayConfig = G1_ROBOT_CONFIG,
) -> Path:
    model_path = model_path.expanduser().resolve()
    urdf_path = urdf_path.expanduser().resolve()
    needs_rebuild = rebuild or not model_path.exists()
    if model_path.exists() and urdf_path.stat().st_mtime > model_path.stat().st_mtime:
        needs_rebuild = True
    if needs_rebuild:
        if robot_config.key == "g1":
            result = convert_urdf_to_mjcf(urdf_path, model_path)
        elif robot_config.key == "a2d":
            result = convert_a2d_urdf_to_mjcf(urdf_path, model_path)
        else:
            raise ValueError(f"Unsupported robot: {robot_config.key}")
        print(f"Generated MuJoCo model: {result.output_path}")
        print(f"Repaired inertials: {', '.join(result.repaired_inertials)}")
    return model_path


def replay_in_viewer(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    *,
    speed: float,
    loop: bool,
    gripper_open: float,
    show_target: bool,
    show_collision: bool,
    robot_config: RobotReplayConfig = G1_ROBOT_CONFIG,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    set_target_visibility(model, show_target)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.05, 0.0, 0.8)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.opt.geomgroup[0] = int(show_collision)
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1

        wall_start = time.monotonic()
        while viewer.is_running():
            iteration_start = time.monotonic()
            elapsed = (iteration_start - wall_start) * speed
            if loop:
                trajectory_time_s = elapsed % trajectory.duration_s
            else:
                trajectory_time_s = min(elapsed, trajectory.duration_s)

            apply_kinematic_pose(
                model,
                data,
                trajectory,
                bindings,
                trajectory_time_s,
                gripper_open=gripper_open,
                show_target=show_target,
                robot_config=robot_config,
            )
            viewer.sync()

            if not loop and elapsed >= trajectory.duration_s:
                break
            remaining = 1.0 / 120.0 - (time.monotonic() - iteration_start)
            if remaining > 0:
                time.sleep(remaining)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--robot",
        choices=tuple(ROBOT_CONFIGS),
        default="g1",
        help="Robot description used for visualization",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument(
        "--no-loop", action="store_false", dest="loop", help="Play once and exit"
    )
    parser.set_defaults(loop=True)
    parser.add_argument(
        "--hide-target",
        action="store_false",
        dest="show_target",
        help="Hide the red target EEF markers",
    )
    parser.set_defaults(show_target=True)
    parser.add_argument(
        "--show-collision", action="store_true", help="Show convex collision geoms"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run FK validation without opening a viewer",
    )
    parser.add_argument("--rebuild-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    if not 0.0 <= args.gripper_open <= 1.0:
        raise ValueError("gripper-open must be in [0, 1]")

    robot_config = ROBOT_CONFIGS[args.robot]
    model_argument = args.model or robot_config.default_mjcf
    urdf_argument = args.urdf or robot_config.default_urdf
    model_path = ensure_model(
        model_argument,
        urdf_argument,
        args.rebuild_model,
        robot_config,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    trajectory = load_trajectory(args.episode, args.summary)
    bindings = bind_joints(
        model,
        trajectory.joint_names,
        robot_config.model_joint_names,
    )
    validate_joint_limits(model, trajectory, bindings)

    print(
        f"Loaded {trajectory.frames} frames, {trajectory.duration_s:.6f} s, "
        f"{(trajectory.frames - 1) / trajectory.duration_s:.3f} Hz"
    )
    print(f"Robot: {robot_config.key}")
    print(f"Model: {model_path}")
    if robot_config.gripper_mode == "a2d_neutral":
        print("A2D gripper: neutral URDF pose (trajectory has no gripper channel)")

    if args.headless:
        report = evaluate_trajectory(
            model,
            trajectory,
            bindings,
            gripper_open=args.gripper_open,
            robot_config=robot_config,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    replay_in_viewer(
        model,
        trajectory,
        bindings,
        speed=args.speed,
        loop=args.loop,
        gripper_open=args.gripper_open,
        show_target=args.show_target,
        show_collision=args.show_collision,
        robot_config=robot_config,
    )


if __name__ == "__main__":
    main()
