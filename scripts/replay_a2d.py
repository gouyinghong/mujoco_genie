#!/usr/bin/env python3
"""Replay A2D arm and gripper actions with a deterministic dice pick-and-place."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_a2d_to_mjcf import (  # noqa: E402
    A2D_ARM_JOINT_NAMES,
    DEFAULT_A2D_MJCF,
    DEFAULT_A2D_ROBOT_ONLY_MJCF,
    DEFAULT_A2D_URDF,
    convert_a2d_urdf_to_mjcf,
)


DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "fixed_spine3_to_g1_add_effector"
DEFAULT_EPISODE = DEFAULT_DATASET_DIR / "episode_000000.npz"
DEFAULT_SUMMARY = DEFAULT_DATASET_DIR / "retarget_summary.json"
ARM_BASE_BODY = "link-arm"
EEF_BODY_NAMES = ("Link7_l", "Link7_r")
A2D_GRIPPER_JOINTS = tuple(
    f"{side}_{finger}{link}_joint"
    for side in ("left", "right")
    for finger in ("narrow", "wide")
    for link in (1, 2, 3, 4)
)
GRIPPER_LINK_ORDER = (1, 3, 4, 2)
GRIPPER_OPENNESS_GRID = np.linspace(0.0, 1.0, 9)
# Four-bar closure solution for the wide finger, ordered as links 1, 3, 4, 2.
# The narrow finger uses the negated values.
GRIPPER_WIDE_JOINT_POSITIONS = np.array(
    [
        (0.0000, 0.0000, 0.0000, 0.0000),
        (0.0982, -0.0124, 0.0697, 0.1674),
        (0.1963, -0.0667, 0.2279, 0.2932),
        (0.2945, -0.1200, 0.3732, 0.4229),
        (0.3927, -0.1647, 0.4823, 0.5569),
        (0.4909, -0.2046, 0.5604, 0.6908),
        (0.5890, -0.2430, 0.6150, 0.8222),
        (0.6872, -0.2816, 0.6525, 0.9500),
        (0.7853981633974483, -0.3215, 0.6780, 1.0740),
    ],
    dtype=float,
)
GRIPPER_CENTER_LOCAL_POS = np.array((0.0, 0.0, 0.14308), dtype=float)
DICE_TABLE_CENTER_Z = 0.8248
A2D_UPPER_BODY_POSE = (
    ("joint_head_yaw", np.deg2rad(0.0)),
    ("joint_head_pitch", np.deg2rad(25.00167804031422)),
    ("joint_body_pitch", 0.3087556226039414),
    ("joint_lift_body", 0.24924583435058595),
)


@dataclass(frozen=True)
class Trajectory:
    times_s: np.ndarray
    joint_positions: np.ndarray
    joint_names: tuple[str, ...]
    target_eef_wxyz: np.ndarray
    achieved_eef_wxyz: np.ndarray
    effector_positions: np.ndarray | None

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


@dataclass(frozen=True)
class DiceReplayPlan:
    side: str
    side_index: int
    grasp_start_frame: int
    release_frame: int
    grasp_start_s: float
    release_s: float
    initial_position: np.ndarray
    initial_quaternion: np.ndarray
    grasp_gripper_quaternion: np.ndarray
    release_position: np.ndarray
    release_quaternion: np.ndarray
    landing_position: np.ndarray
    drop_duration_s: float


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
        effector_positions = (
            np.asarray(episode["action_effector"], dtype=float)
            if "action_effector" in episode.files
            else None
        )

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
    if effector_positions is not None:
        if effector_positions.shape != (frames, 2):
            raise ValueError(
                f"action_effector must have shape ({frames}, 2), "
                f"got {effector_positions.shape}"
            )
        if np.any((effector_positions < 0.0) | (effector_positions > 1.0)):
            raise ValueError("action_effector values must be in [0, 1]")
    if np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("local_timestamps_ns must be strictly increasing")
    if not all(
        np.isfinite(array).all()
        for array in (
            joint_positions,
            target_eef,
            achieved_eef,
            *(() if effector_positions is None else (effector_positions,)),
        )
    ):
        raise ValueError("Trajectory contains NaN or infinite values")

    times_s = (timestamps_ns - timestamps_ns[0]).astype(float) * 1e-9
    return Trajectory(
        times_s=times_s,
        joint_positions=joint_positions,
        joint_names=joint_names,
        target_eef_wxyz=target_eef,
        achieved_eef_wxyz=achieved_eef,
        effector_positions=effector_positions,
    )


def bind_joints(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> JointBindings:
    """Bind the 14 source columns to A2D joints by side and joint index."""

    if len(joint_names) != len(A2D_ARM_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(A2D_ARM_JOINT_NAMES)} arm columns, got {len(joint_names)}"
        )

    joint_ids: list[int] = []
    missing: list[str] = []
    for source_name, model_name in zip(
        joint_names, A2D_ARM_JOINT_NAMES, strict=True
    ):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, model_name
        )
        if joint_id < 0:
            missing.append(f"{source_name} -> {model_name}")
            continue
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Replay joint {model_name!r} is not a scalar joint")
        joint_ids.append(joint_id)

    if missing:
        raise ValueError(f"Model is missing replay joints: {missing}")
    ids = np.asarray(joint_ids, dtype=int)
    return JointBindings(
        joint_ids=ids,
        qpos_addresses=model.jnt_qposadr[ids].astype(int),
        dof_addresses=model.jnt_dofadr[ids].astype(int),
    )


def validate_joint_limits(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    *,
    tolerance: float = 1e-8,
) -> None:
    violations: list[str] = []
    for column, (source_name, joint_id) in enumerate(
        zip(trajectory.joint_names, bindings.joint_ids, strict=True)
    ):
        if not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        observed_min = float(np.min(trajectory.joint_positions[:, column]))
        observed_max = float(np.max(trajectory.joint_positions[:, column]))
        if observed_min < lower - tolerance or observed_max > upper + tolerance:
            model_name = A2D_ARM_JOINT_NAMES[column]
            violations.append(
                f"{source_name} -> {model_name}: observed "
                f"[{observed_min:.6f}, {observed_max:.6f}], "
                f"limit [{lower:.6f}, {upper:.6f}]"
            )
    if violations:
        raise ValueError("Joint limit violations:\n" + "\n".join(violations))


def _joint_address(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Model is missing joint {joint_name!r}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def set_gripper_neutral(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Hold A2D's four-bar gripper joints at their authored zero pose."""

    for name in A2D_GRIPPER_JOINTS:
        qpos_address, dof_address = _joint_address(model, name)
        data.qpos[qpos_address] = 0.0
        data.qvel[dof_address] = 0.0


def gripper_joint_positions(openness: float) -> np.ndarray:
    """Map one normalized gripper command to the wide-finger four-bar joints."""

    openness = float(np.clip(openness, 0.0, 1.0))
    return np.array(
        [
            np.interp(
                openness,
                GRIPPER_OPENNESS_GRID,
                GRIPPER_WIDE_JOINT_POSITIONS[:, column],
            )
            for column in range(GRIPPER_WIDE_JOINT_POSITIONS.shape[1])
        ],
        dtype=float,
    )


def set_gripper_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    effector_positions: np.ndarray,
    effector_velocities: np.ndarray | None = None,
) -> None:
    """Apply normalized [left, right] openness commands to both four-bar grippers."""

    commands = np.asarray(effector_positions, dtype=float)
    if commands.shape != (2,) or np.any((commands < 0.0) | (commands > 1.0)):
        raise ValueError("effector_positions must contain two values in [0, 1]")
    velocities = (
        np.zeros(2, dtype=float)
        if effector_velocities is None
        else np.asarray(effector_velocities, dtype=float)
    )
    if velocities.shape != (2,):
        raise ValueError("effector_velocities must contain two values")

    epsilon = 1e-5
    for side_index, side in enumerate(("left", "right")):
        command = float(commands[side_index])
        wide_positions = gripper_joint_positions(command)
        lower = gripper_joint_positions(max(0.0, command - epsilon))
        upper = gripper_joint_positions(min(1.0, command + epsilon))
        denominator = min(1.0, command + epsilon) - max(0.0, command - epsilon)
        slopes = (upper - lower) / denominator
        wide_velocities = slopes * velocities[side_index]

        for finger, sign in (("wide", 1.0), ("narrow", -1.0)):
            for link, position, velocity in zip(
                GRIPPER_LINK_ORDER,
                sign * wide_positions,
                sign * wide_velocities,
                strict=True,
            ):
                qpos_address, dof_address = _joint_address(
                    model, f"{side}_{finger}{link}_joint"
                )
                data.qpos[qpos_address] = position
                data.qvel[dof_address] = velocity


def set_upper_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: tuple[tuple[str, float], ...] = A2D_UPPER_BODY_POSE,
) -> None:
    """Hold the head and torso at the requested replay pose."""

    for name, position in pose:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Model is missing upper-body joint {name!r}")
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"Upper-body joint {name!r} is not scalar")
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if not lower <= position <= upper:
                raise ValueError(
                    f"Upper-body joint {name!r} position {position} is outside "
                    f"[{lower}, {upper}]"
                )
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_address] = position
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


def interpolate_effector_state(
    trajectory: Trajectory, trajectory_time_s: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate normalized gripper openness and its velocity."""

    if trajectory.effector_positions is None:
        return None
    time_s = float(np.clip(trajectory_time_s, 0.0, trajectory.duration_s))
    right = int(np.searchsorted(trajectory.times_s, time_s, side="right"))
    left = max(0, min(right - 1, trajectory.frames - 2))
    right = left + 1
    interval = trajectory.times_s[right] - trajectory.times_s[left]
    alpha = float(np.clip((time_s - trajectory.times_s[left]) / interval, 0.0, 1.0))
    e0 = trajectory.effector_positions[left]
    e1 = trajectory.effector_positions[right]
    return (1.0 - alpha) * e0 + alpha * e1, (e1 - e0) / interval


def infer_grasp_frames(trajectory: Trajectory) -> tuple[int, int, int]:
    """Infer active side, final sustained closing onset, and release onset."""

    if trajectory.effector_positions is None:
        raise ValueError("Trajectory does not contain action_effector")
    ranges = np.ptp(trajectory.effector_positions, axis=0)
    side_index = int(np.argmax(ranges))
    values = trajectory.effector_positions[:, side_index]
    closed_frame = int(np.argmin(values))
    close_span = float(values[: closed_frame + 1].max() - values[closed_frame])
    if close_span < 0.1:
        raise ValueError("No substantial gripper closing event was found")

    minimum_required_drop = max(0.1, 0.25 * float(ranges[side_index]))
    candidates: list[int] = []
    for frame in range(1, closed_frame):
        window = values[max(0, frame - 3) : min(closed_frame + 1, frame + 4)]
        if (
            values[frame] >= float(np.max(window)) - 1e-6
            and values[frame] - values[closed_frame] >= minimum_required_drop
        ):
            candidates.append(frame)
    grasp_start_frame = candidates[-1] if candidates else 0

    release_frame = trajectory.frames - 1
    for frame in range(closed_frame + 1, trajectory.frames):
        if values[frame] - values[frame - 1] > 0.005:
            release_frame = frame
            break
    if release_frame <= closed_frame:
        raise ValueError("No gripper release event was found after closing")
    return side_index, grasp_start_frame, release_frame


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
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.dot(first, second) < 0.0:
        second = -second
    result = (1.0 - alpha) * first + alpha * second
    return result / np.linalg.norm(result)


def _pose_in_parent_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    child_body_name: str,
) -> np.ndarray:
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM_BASE_BODY)
    child_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, child_body_name
    )
    if parent_id < 0 or child_id < 0:
        raise ValueError(f"Missing FK body: {ARM_BASE_BODY} or {child_body_name}")
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
) -> None:
    arm_base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, ARM_BASE_BODY
    )
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
    )
    if arm_base_id < 0 or target_body_id < 0:
        raise ValueError(f"Model is missing {ARM_BASE_BODY} or a target mocap body")
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


def _interpolate_target(trajectory: Trajectory, time_s: float) -> np.ndarray:
    right = int(
        np.clip(
            np.searchsorted(trajectory.times_s, time_s, side="left"),
            0,
            trajectory.frames - 1,
        )
    )
    left = max(0, right - 1)
    if left == right:
        return trajectory.target_eef_wxyz[right]

    interval = trajectory.times_s[right] - trajectory.times_s[left]
    alpha = (time_s - trajectory.times_s[left]) / interval
    target = trajectory.target_eef_wxyz[left].copy()
    for start in (0, 7):
        target[start : start + 3] = (
            (1.0 - alpha) * trajectory.target_eef_wxyz[left, start : start + 3]
            + alpha * trajectory.target_eef_wxyz[right, start : start + 3]
        )
        target[start + 3 : start + 7] = _quat_interpolate(
            trajectory.target_eef_wxyz[left, start + 3 : start + 7],
            trajectory.target_eef_wxyz[right, start + 3 : start + 7],
            alpha,
        )
    return target


def _grasp_frame_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_link"
    )
    if base_body_id < 0:
        raise ValueError(f"Model is missing {side!r} gripper base body")
    rotation = data.xmat[base_body_id].reshape(3, 3)
    position = data.xpos[base_body_id] + rotation @ GRIPPER_CENTER_LOCAL_POS
    return position.copy(), data.xquat[base_body_id].copy()


def _set_dice_replay_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: DiceReplayPlan,
    trajectory_time_s: float,
) -> None:
    dice_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    if dice_joint_id < 0:
        return
    qpos_address = int(model.jnt_qposadr[dice_joint_id])
    dof_address = int(model.jnt_dofadr[dice_joint_id])

    if trajectory_time_s < plan.grasp_start_s:
        position = plan.initial_position
        quaternion = plan.initial_quaternion
    elif trajectory_time_s < plan.release_s:
        position, gripper_quaternion = _grasp_frame_pose(model, data, plan.side)
        gripper_to_dice = _quat_multiply(
            _quat_conjugate(plan.grasp_gripper_quaternion),
            plan.initial_quaternion,
        )
        quaternion = _quat_multiply(gripper_quaternion, gripper_to_dice)
    else:
        drop_time = max(0.0, trajectory_time_s - plan.release_s)
        if plan.drop_duration_s <= 0.0:
            alpha = 1.0
        else:
            alpha = float(np.clip(drop_time / plan.drop_duration_s, 0.0, 1.0))
        position = plan.release_position.copy()
        position[2] = max(
            plan.landing_position[2],
            plan.release_position[2] - 0.5 * 9.81 * drop_time**2,
        )
        if alpha >= 1.0:
            position = plan.landing_position
        quaternion = _quat_interpolate(
            plan.release_quaternion,
            np.array((1.0, 0.0, 0.0, 0.0)),
            alpha,
        )

    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
    data.qvel[dof_address : dof_address + 6] = 0.0


def apply_kinematic_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: Trajectory,
    bindings: JointBindings,
    trajectory_time_s: float,
    *,
    show_target: bool,
    dice_plan: DiceReplayPlan | None = None,
    upper_body_pose: tuple[tuple[str, float], ...] = A2D_UPPER_BODY_POSE,
) -> None:
    """Write interpolated arm, gripper, and dice states into MuJoCo."""

    joint_positions, joint_velocities = interpolate_joint_state(
        trajectory, trajectory_time_s
    )
    data.qpos[bindings.qpos_addresses] = joint_positions
    data.qvel[:] = 0.0
    data.qvel[bindings.dof_addresses] = joint_velocities
    set_upper_body_pose(model, data, upper_body_pose)
    effector_state = interpolate_effector_state(trajectory, trajectory_time_s)
    if effector_state is None:
        set_gripper_neutral(model, data)
    else:
        set_gripper_command(model, data, *effector_state)
    data.time = float(np.clip(trajectory_time_s, 0.0, trajectory.duration_s))
    mujoco.mj_forward(model, data)

    if show_target:
        target = _interpolate_target(trajectory, data.time)
        _set_target_mocap_pose(model, data, target[:7], "left_eef_target_body")
        _set_target_mocap_pose(model, data, target[7:], "right_eef_target_body")
    if dice_plan is not None:
        _set_dice_replay_pose(model, data, dice_plan, data.time)
    if show_target or dice_plan is not None:
        mujoco.mj_forward(model, data)


def build_dice_replay_plan(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    *,
    upper_body_pose: tuple[tuple[str, float], ...] = A2D_UPPER_BODY_POSE,
) -> DiceReplayPlan | None:
    """Build the deterministic grasp, carry, release, and table landing plan."""

    if trajectory.effector_positions is None:
        return None
    if (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint")
        < 0
    ):
        return None

    side_index, grasp_start_frame, release_frame = infer_grasp_frames(trajectory)
    side = ("left", "right")[side_index]
    probe = mujoco.MjData(model)
    probe.qpos[:] = model.qpos0

    grasp_start_s = float(trajectory.times_s[grasp_start_frame])
    apply_kinematic_pose(
        model,
        probe,
        trajectory,
        bindings,
        grasp_start_s,
        show_target=False,
        upper_body_pose=upper_body_pose,
    )
    initial_position, grasp_gripper_quaternion = _grasp_frame_pose(
        model, probe, side
    )
    # Start with the cube upright so its collision box rests flat on the table.
    initial_quaternion = np.array((1.0, 0.0, 0.0, 0.0), dtype=float)

    release_s = float(trajectory.times_s[release_frame])
    apply_kinematic_pose(
        model,
        probe,
        trajectory,
        bindings,
        release_s,
        show_target=False,
        upper_body_pose=upper_body_pose,
    )
    release_position, release_gripper_quaternion = _grasp_frame_pose(
        model, probe, side
    )
    gripper_to_dice = _quat_multiply(
        _quat_conjugate(grasp_gripper_quaternion), initial_quaternion
    )
    release_quaternion = _quat_multiply(
        release_gripper_quaternion, gripper_to_dice
    )
    landing_position = release_position.copy()
    landing_position[2] = DICE_TABLE_CENTER_Z
    drop_height = max(0.0, release_position[2] - landing_position[2])
    drop_duration_s = float(np.sqrt(2.0 * drop_height / 9.81))

    return DiceReplayPlan(
        side=side,
        side_index=side_index,
        grasp_start_frame=grasp_start_frame,
        release_frame=release_frame,
        grasp_start_s=grasp_start_s,
        release_s=release_s,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        grasp_gripper_quaternion=grasp_gripper_quaternion,
        release_position=release_position,
        release_quaternion=release_quaternion,
        landing_position=landing_position,
        drop_duration_s=drop_duration_s,
    )


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
    upper_body_pose: tuple[tuple[str, float], ...] = A2D_UPPER_BODY_POSE,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    dice_plan = build_dice_replay_plan(
        model, trajectory, bindings, upper_body_pose=upper_body_pose
    )
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
            show_target=False,
            dice_plan=dice_plan,
            upper_body_pose=upper_body_pose,
        )
        max_contacts = max(max_contacts, int(data.ncon))
        for side, body_name, pose_slice in (
            ("left", EEF_BODY_NAMES[0], slice(0, 7)),
            ("right", EEF_BODY_NAMES[1], slice(7, 14)),
        ):
            actual = _pose_in_parent_frame(model, data, body_name)
            expected = trajectory.achieved_eef_wxyz[frame, pose_slice]
            position_errors[side].append(float(np.linalg.norm(actual[:3] - expected[:3])))
            orientation_errors[side].append(
                _quaternion_error_degrees(actual[3:], expected[3:])
            )

    result = {
        "frames": trajectory.frames,
        "duration_s": trajectory.duration_s,
        "source_fps": (trajectory.frames - 1) / trajectory.duration_s,
        "fk_position_error_m": {
            side: {"mean": float(np.mean(errors)), "max": float(np.max(errors))}
            for side, errors in position_errors.items()
        },
        "fk_orientation_error_deg": {
            side: {"mean": float(np.mean(errors)), "max": float(np.max(errors))}
            for side, errors in orientation_errors.items()
        },
        "max_contacts": max_contacts,
    }
    if trajectory.effector_positions is not None:
        result["action_effector_range"] = {
            side: {
                "min": float(np.min(trajectory.effector_positions[:, index])),
                "max": float(np.max(trajectory.effector_positions[:, index])),
            }
            for index, side in enumerate(("left", "right"))
        }
    if dice_plan is not None:
        result["dice_replay"] = {
            "gripper": dice_plan.side,
            "grasp_start_frame": dice_plan.grasp_start_frame,
            "grasp_start_s": dice_plan.grasp_start_s,
            "release_frame": dice_plan.release_frame,
            "release_s": dice_plan.release_s,
            "initial_position": dice_plan.initial_position.tolist(),
            "landing_position": dice_plan.landing_position.tolist(),
        }
    return result


def ensure_model(
    model_path: Path = DEFAULT_A2D_MJCF,
    urdf_path: Path = DEFAULT_A2D_URDF,
    rebuild: bool = False,
    include_table: bool | None = None,
) -> Path:
    model_path = model_path.expanduser().resolve()
    urdf_path = urdf_path.expanduser().resolve()
    if include_table is None:
        include_table = model_path != DEFAULT_A2D_ROBOT_ONLY_MJCF.resolve()
    needs_rebuild = rebuild or not model_path.exists()
    if model_path.exists() and urdf_path.stat().st_mtime > model_path.stat().st_mtime:
        needs_rebuild = True
    if needs_rebuild:
        result = convert_a2d_urdf_to_mjcf(
            urdf_path,
            model_path,
            include_table=include_table,
        )
        print(f"Generated MuJoCo model: {result.output_path}")
        print(f"Repaired inertials: {', '.join(result.repaired_inertials) or 'none'}")
    return model_path


def replay_in_viewer(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    dice_plan: DiceReplayPlan | None,
    *,
    speed: float,
    loop: bool,
    wait_for_start: bool,
    upper_body_pose: tuple[tuple[str, float], ...],
    show_target: bool,
    show_collision: bool,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    set_target_visibility(model, show_target)
    apply_kinematic_pose(
        model,
        data,
        trajectory,
        bindings,
        0.0,
        show_target=show_target,
        dice_plan=dice_plan,
        upper_body_pose=upper_body_pose,
    )

    playback_toggle_requested = threading.Event()

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            playback_toggle_requested.set()

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        viewer.cam.lookat[:] = (0.05, 0.0, 0.8)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.opt.geomgroup[0] = int(show_collision)
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1

        paused = wait_for_start
        if paused:
            print(
                "Replay paused at frame 0. Adjust the camera in the viewer, "
                "then focus the viewer window and press SPACE to start."
            )
        else:
            print("Press SPACE in the viewer to pause/resume replay.")

        elapsed = 0.0
        previous_wall_time = time.monotonic()
        while viewer.is_running():
            iteration_start = time.monotonic()
            wall_delta = iteration_start - previous_wall_time
            previous_wall_time = iteration_start
            if playback_toggle_requested.is_set():
                playback_toggle_requested.clear()
                paused = not paused
                wall_delta = 0.0
                state = "paused" if paused else "resumed"
                print(
                    f"Replay {state} at {elapsed:.3f} s. "
                    "Press SPACE to toggle playback.",
                    flush=True,
                )
            if not paused:
                elapsed += wall_delta * speed
            trajectory_time_s = (
                elapsed % trajectory.duration_s
                if loop
                else min(elapsed, trajectory.duration_s)
            )
            apply_kinematic_pose(
                model,
                data,
                trajectory,
                bindings,
                trajectory_time_s,
                show_target=show_target,
                dice_plan=dice_plan,
                upper_body_pose=upper_body_pose,
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
    parser.add_argument("--model", type=Path, default=DEFAULT_A2D_MJCF)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_A2D_URDF)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--body-lift-m",
        type=float,
        default=dict(A2D_UPPER_BODY_POSE)["joint_lift_body"],
        help="Fixed torso lift used by replay and dice-position inference",
    )
    parser.add_argument(
        "--no-loop",
        action="store_false",
        dest="loop",
        help="Pause at frame 0 until SPACE is pressed, then play once",
    )
    parser.set_defaults(loop=True)
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Do not wait for SPACE before a one-shot replay",
    )
    parser.add_argument(
        "--hide-target",
        action="store_false",
        dest="show_target",
        help="Hide the red target EEF markers",
    )
    parser.set_defaults(show_target=True)
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument(
        "--headless", action="store_true", help="Run FK diagnostics without a viewer"
    )
    parser.add_argument("--rebuild-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.speed <= 0:
        raise ValueError("speed must be positive")

    model_path = ensure_model(args.model, args.urdf, args.rebuild_model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    trajectory = load_trajectory(args.episode, args.summary)
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)
    upper_body_pose = tuple(
        (name, args.body_lift_m if name == "joint_lift_body" else position)
        for name, position in A2D_UPPER_BODY_POSE
    )
    dice_plan = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=upper_body_pose,
    )

    print(
        f"Loaded {trajectory.frames} frames, {trajectory.duration_s:.6f} s, "
        f"{(trajectory.frames - 1) / trajectory.duration_s:.3f} Hz"
    )
    print(f"Model: {model_path}")
    print(f"Torso lift: {args.body_lift_m:.6f} m")
    if trajectory.effector_positions is None:
        print("A2D gripper: neutral URDF pose (trajectory has no gripper channel)")
    else:
        print("A2D gripper: replaying action_effector (0=closed, 1=open)")
    if dice_plan is not None:
        print(
            f"Dice: {dice_plan.side} gripper closes from frame "
            f"{dice_plan.grasp_start_frame}, releases at frame "
            f"{dice_plan.release_frame}; initial="
            f"{np.array2string(dice_plan.initial_position, precision=4)}"
        )

    if args.headless:
        print(
            json.dumps(
                evaluate_trajectory(
                    model,
                    trajectory,
                    bindings,
                    upper_body_pose=upper_body_pose,
                ),
                indent=2,
            )
        )
        return

    replay_in_viewer(
        model,
        trajectory,
        bindings,
        dice_plan,
        speed=args.speed,
        loop=args.loop,
        wait_for_start=not args.loop and not args.start_immediately,
        upper_body_pose=upper_body_pose,
        show_target=args.show_target,
        show_collision=args.show_collision,
    )


if __name__ == "__main__":
    main()
