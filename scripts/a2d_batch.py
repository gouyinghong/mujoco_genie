"""Batch preparation utilities for fixed-torso A2D dataset replay."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from scripts.replay_a2d import (
    A2D_UPPER_BODY_POSE,
    DICE_HALF_EXTENT_M,
    JointBindings,
    Trajectory,
    _grasp_frame_pose,
    apply_kinematic_pose,
    bind_joints,
    build_dice_replay_plan,
    load_trajectory,
    set_cardboard_box_pose,
    validate_joint_limits,
)


LAYOUT_SCHEMA = "a2d_fixed_torso_batch_replay.v1"
DEFAULT_BODY_LIFT_M = 0.264496
DEFAULT_BODY_PITCH_RAD = 0.387295
BOX_INNER_HALF_SIZE_M = np.array((0.116, 0.076), dtype=float)
TABLE_X_BOUNDS_M = (0.45, 1.35)
TABLE_Y_BOUNDS_M = (-0.70, 0.70)
BOX_CONTACT_TOLERANCE_M = 5e-4
TABLE_PENETRATION_TOLERANCE_M = 5e-4
IK_TOLERANCE_M = 0.002
PLACEMENT_LIFTS_M = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)


def fixed_upper_body_pose(
    body_lift_m: float, body_pitch_rad: float
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (
            name,
            body_lift_m
            if name == "joint_lift_body"
            else body_pitch_rad
            if name == "joint_body_pitch"
            else value,
        )
        for name, value in A2D_UPPER_BODY_POSE
    )


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def robot_table_metrics(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    upper_body_pose: tuple[tuple[str, float], ...],
) -> dict[str, Any]:
    """Measure robot/table penetrations over an entire replay trajectory."""

    robot_root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
    )
    table_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "table"
    )
    if min(robot_root_id, table_body_id) < 0:
        raise ValueError("Model must contain base_link and table bodies")
    table_geom_ids = set(
        np.flatnonzero(model.geom_bodyid == table_body_id).astype(int).tolist()
    )
    data = mujoco.MjData(model)
    contact_frames: set[int] = set()
    max_penetration = 0.0
    first_frame: int | None = None
    deepest_frame: int | None = None
    deepest_body: str | None = None
    for frame, time_s in enumerate(trajectory.times_s):
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            float(time_s),
            show_target=False,
            upper_body_pose=upper_body_pose,
        )
        for contact in data.contact:
            geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
            if geom0 in table_geom_ids:
                other_geom = geom1
            elif geom1 in table_geom_ids:
                other_geom = geom0
            else:
                continue
            other_body = int(model.geom_bodyid[other_geom])
            if not _is_descendant(model, other_body, robot_root_id):
                continue
            penetration = max(0.0, -float(contact.dist))
            if penetration <= TABLE_PENETRATION_TOLERANCE_M:
                continue
            contact_frames.add(frame)
            if first_frame is None:
                first_frame = frame
            if penetration > max_penetration:
                max_penetration = penetration
                deepest_frame = frame
                deepest_body = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, other_body
                ) or f"body_{other_body}"
    return {
        "contact_frames": len(contact_frames),
        "first_contact_frame": first_frame,
        "deepest_contact_frame": deepest_frame,
        "deepest_body": deepest_body,
        "max_penetration_m": max_penetration,
    }


def adjustment_path(episode_path: Path) -> Path:
    return episode_path.with_name(f"{episode_path.stem}_gripper_adjustment.json")


def load_gripper_boundaries(episode_path: Path) -> dict[str, int]:
    path = adjustment_path(episode_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing gripper adjustment metadata: {path}")
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    required = ("close_start_frame", "close_end_frame", "open_start_frame", "open_end_frame")
    boundaries = document.get("boundaries", {})
    missing = [name for name in required if name not in boundaries]
    if missing:
        raise ValueError(f"{path} is missing boundary fields: {missing}")
    return {name: int(boundaries[name]) for name in required}


def load_layout_overrides(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    local_path = dataset_dir / "replay_layout_overrides.json"
    shared_path = dataset_dir.parent / "replay_layout_overrides.json"
    if local_path.is_file():
        path = local_path
        with path.open("r", encoding="utf-8") as stream:
            overrides = json.load(stream)
    elif shared_path.is_file():
        path = shared_path
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError(f"{path} must contain a JSON object")
        overrides = document.get(dataset_dir.name, {})
    else:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return overrides


def choose_dice_center_frame(boundaries: dict[str, int], frames: int) -> int:
    """Choose the same relative closing phase as tuned frame 37 of episode 0."""

    start = boundaries["close_start_frame"]
    end = boundaries["close_end_frame"]
    span = max(1, end - start)
    frame = end - max(1, int(round(0.4 * span)))
    return int(np.clip(frame, start, min(end, frames - 1)))


def _right_gripper_center_at_frame(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    upper_body_pose: tuple[tuple[str, float], ...],
    frame: int,
) -> np.ndarray:
    data = mujoco.MjData(model)
    apply_kinematic_pose(
        model,
        data,
        trajectory,
        bindings,
        float(trajectory.times_s[frame]),
        show_target=False,
        upper_body_pose=upper_body_pose,
    )
    position, _ = _grasp_frame_pose(model, data, "right")
    return position


def translate_right_arm_trajectory(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    upper_body_pose: tuple[tuple[str, float], ...],
    z_offset_m: float | np.ndarray,
    *,
    tolerance_m: float = 2e-4,
    max_iterations: int = 80,
) -> tuple[Trajectory, dict[str, float]]:
    """Translate the right gripper path vertically using position-only DLS IK."""

    offsets = np.asarray(z_offset_m, dtype=float)
    if offsets.ndim == 0:
        offsets = np.full(trajectory.frames, float(offsets), dtype=float)
    if offsets.shape != (trajectory.frames,):
        raise ValueError(
            f"z_offset_m must be scalar or shape ({trajectory.frames},), "
            f"got {offsets.shape}"
        )
    if float(np.max(np.abs(offsets))) < 1e-12:
        return trajectory, {
            "max_position_error_m": 0.0,
            "mean_position_error_m": 0.0,
            "max_joint_step_rad": float(
                np.max(np.abs(np.diff(trajectory.joint_positions[:, 7:], axis=0)))
            ),
        }

    corrected = trajectory.joint_positions.copy()
    data = mujoco.MjData(model)
    base_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "right_base_link"
    )
    if base_body_id < 0:
        raise ValueError("Model is missing right_base_link")
    joint_ids = bindings.joint_ids[7:]
    qpos_addresses = bindings.qpos_addresses[7:]
    dof_addresses = bindings.dof_addresses[7:]
    lower = np.where(
        model.jnt_limited[joint_ids], model.jnt_range[joint_ids, 0], -np.inf
    )
    upper = np.where(
        model.jnt_limited[joint_ids], model.jnt_range[joint_ids, 1], np.inf
    )
    jacobian_position = np.zeros((3, model.nv), dtype=float)
    jacobian_rotation = np.zeros((3, model.nv), dtype=float)
    previous_correction = np.zeros(7, dtype=float)
    errors: list[float] = []

    for frame, time_s in enumerate(trajectory.times_s):
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            float(time_s),
            show_target=False,
            upper_body_pose=upper_body_pose,
        )
        original = data.qpos[qpos_addresses].copy()
        target, _ = _grasp_frame_pose(model, data, "right")
        target[2] += offsets[frame]
        qpos = np.clip(original + previous_correction, lower, upper)

        for _ in range(max_iterations):
            data.qpos[qpos_addresses] = qpos
            mujoco.mj_forward(model, data)
            position, _ = _grasp_frame_pose(model, data, "right")
            error = target - position
            if np.linalg.norm(error) <= tolerance_m:
                break
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                position,
                base_body_id,
            )
            jacobian = jacobian_position[:, dof_addresses]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-5 * np.eye(3), error
            )
            norm = float(np.linalg.norm(delta))
            if norm > 0.1:
                delta *= 0.1 / norm
            qpos = np.clip(qpos + delta, lower, upper)

        data.qpos[qpos_addresses] = qpos
        mujoco.mj_forward(model, data)
        position, _ = _grasp_frame_pose(model, data, "right")
        errors.append(float(np.linalg.norm(target - position)))
        corrected[frame, 7:] = qpos
        previous_correction = qpos - original

    result = replace(trajectory, joint_positions=corrected)
    validate_joint_limits(model, result, bindings)
    return result, {
        "max_position_error_m": float(np.max(errors)),
        "mean_position_error_m": float(np.mean(errors)),
        "max_joint_step_rad": float(
            np.max(np.abs(np.diff(corrected[:, 7:], axis=0)))
        ),
    }


def placement_lift_profile(
    frames: int,
    boundaries: dict[str, int],
    lift_m: float,
) -> np.ndarray:
    """Smoothly raise the right arm after grasp and lower it after release."""

    profile = np.zeros(frames, dtype=float)
    ramp_start = min(frames - 1, boundaries["close_end_frame"])
    open_start = min(frames - 1, boundaries["open_start_frame"])
    open_end = min(frames - 1, boundaries["open_end_frame"])
    ramp_end = max(ramp_start + 1, open_start - 5)
    ramp_end = min(ramp_end, open_start)
    lower_end = min(frames - 1, open_end + 10)

    for frame in range(ramp_start, ramp_end + 1):
        alpha = (frame - ramp_start) / max(1, ramp_end - ramp_start)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        profile[frame] = lift_m * smooth
    profile[ramp_end : open_end + 1] = lift_m
    for frame in range(open_end, lower_end + 1):
        alpha = (frame - open_end) / max(1, lower_end - open_end)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        profile[frame] = lift_m * (1.0 - smooth)
    return profile


def _box_wall_ids(model: mujoco.MjModel) -> set[int]:
    return {
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"cardboard_box_collision_wall_{axis}_{side}",
        )
        for axis in ("x", "y")
        for side in ("negative", "positive")
    }


def _box_pose_candidates(landing_xy: np.ndarray):
    yaw_order = [0.0]
    for magnitude in range(10, 91, 10):
        yaw_order.extend((float(magnitude), float(-magnitude)))
    for yaw_deg in yaw_order:
        yaw = np.deg2rad(yaw_deg)
        rotation = np.array(
            ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw))),
            dtype=float,
        )
        dice_projection = DICE_HALF_EXTENT_M * (
            abs(np.cos(yaw)) + abs(np.sin(yaw))
        )
        margins = BOX_INNER_HALF_SIZE_M - dice_projection - 0.005
        fractions = (0.0, 1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0, -1.0)
        offsets = sorted(
            (
                (margins[0] * fraction_x, margins[1] * fraction_y)
                for fraction_x in fractions
                for fraction_y in fractions
            ),
            key=lambda offset: offset[0] ** 2 + offset[1] ** 2,
        )
        for local_offset in offsets:
            local_offset_array = np.asarray(local_offset, dtype=float)
            center = landing_xy - rotation @ local_offset_array
            outer_projection = np.abs(rotation) @ np.array((0.12, 0.08))
            if not (
                TABLE_X_BOUNDS_M[0] <= center[0] - outer_projection[0]
                and center[0] + outer_projection[0] <= TABLE_X_BOUNDS_M[1]
                and TABLE_Y_BOUNDS_M[0] <= center[1] - outer_projection[1]
                and center[1] + outer_projection[1] <= TABLE_Y_BOUNDS_M[1]
            ):
                continue
            yield float(center[0]), float(center[1]), yaw_deg


def find_collision_free_box_pose(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    upper_body_pose: tuple[tuple[str, float], ...],
    dice_center_frame: int,
    *,
    dice_xy_offset_m: tuple[float, float] = (0.0, 0.0),
    yaw_zero_only: bool = False,
) -> tuple[dict[str, float], Any, dict[str, float]]:
    """Search a table-valid box pose that has no wall contacts."""

    provisional = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=upper_body_pose,
        dice_on_table=True,
        align_dice_to_gripper=True,
        dice_center_frame=dice_center_frame,
        dice_xy_offset_m=dice_xy_offset_m,
    )
    if provisional is None:
        raise ValueError("Model or trajectory cannot produce a dice replay plan")
    wall_ids = _box_wall_ids(model)
    if min(wall_ids) < 0:
        raise ValueError("Model is missing cardboard-box wall collision geoms")
    best: tuple[int, float, dict[str, float], Any] | None = None

    candidates = _box_pose_candidates(provisional.landing_position[:2])
    if yaw_zero_only:
        x, y = provisional.landing_position[:2]
        candidates = iter(((float(x), float(y), 0.0),))
    for x, y, yaw_deg in candidates:
        set_cardboard_box_pose(model, x=x, y=y, yaw_deg=yaw_deg)
        plan = build_dice_replay_plan(
            model,
            trajectory,
            bindings,
            upper_body_pose=upper_body_pose,
            dice_on_table=True,
            align_dice_to_gripper=True,
            dice_center_frame=dice_center_frame,
            dice_xy_offset_m=dice_xy_offset_m,
        )
        assert plan is not None
        data = mujoco.MjData(model)
        contacts = 0
        max_penetration = 0.0
        collision_found = False
        for time_s in trajectory.times_s:
            apply_kinematic_pose(
                model,
                data,
                trajectory,
                bindings,
                float(time_s),
                show_target=False,
                dice_plan=plan,
                upper_body_pose=upper_body_pose,
            )
            for contact in data.contact:
                if int(contact.geom[0]) in wall_ids or int(contact.geom[1]) in wall_ids:
                    penetration = max(0.0, -float(contact.dist))
                    if penetration <= BOX_CONTACT_TOLERANCE_M:
                        continue
                    contacts += 1
                    max_penetration = max(max_penetration, penetration)
                    collision_found = True
                    break
            if collision_found:
                break
        pose = {"x": x, "y": y, "z": 0.8, "yaw_deg": yaw_deg}
        score = (contacts, max_penetration)
        if best is None or score < best[:2]:
            best = (contacts, max_penetration, pose, plan)
        if contacts == 0:
            return pose, plan, {
                "box_wall_contacts": 0,
                "max_box_wall_penetration_m": 0.0,
            }

    if best is None:
        raise ValueError("No box pose candidate fits on the table")
    contacts, penetration, pose, plan = best
    return pose, plan, {
        "box_wall_contacts": int(contacts),
        "max_box_wall_penetration_m": float(penetration),
    }


def load_corrected_trajectory(
    episode_path: Path, summary_path: Path, cache_path: Path | None
) -> Trajectory:
    trajectory = load_trajectory(episode_path, summary_path)
    if cache_path is None:
        return trajectory
    with np.load(cache_path, allow_pickle=False) as cache:
        joint_positions = np.asarray(cache["joint_positions"], dtype=float)
    if joint_positions.shape != trajectory.joint_positions.shape:
        raise ValueError(
            f"Cached joints have shape {joint_positions.shape}, expected "
            f"{trajectory.joint_positions.shape}"
        )
    return replace(trajectory, joint_positions=joint_positions)


def prepare_dataset_layouts(
    model_path: Path,
    dataset_dir: Path,
    output_path: Path,
    *,
    body_lift_m: float | None = None,
    body_pitch_rad: float | None = None,
    max_episodes: int | None = None,
    episode_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    summary_path = dataset_dir / "retarget_summary.json"
    all_episodes = sorted(dataset_dir.glob("episode_*.npz"))
    if not all_episodes:
        raise FileNotFoundError(f"No episode_*.npz files found in {dataset_dir}")
    if episode_names and max_episodes is not None:
        raise ValueError("episode_names and max_episodes cannot be used together")

    existing_document: dict[str, Any] | None = None
    if episode_names:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Incremental update requires an existing manifest: {output_path}. "
                "Run a full preparation first."
            )
        with output_path.open("r", encoding="utf-8") as stream:
            existing_document = json.load(stream)
        if existing_document.get("schema") != LAYOUT_SCHEMA:
            raise ValueError(f"Cannot merge unsupported manifest: {output_path}")
        if Path(existing_document.get("dataset_dir", "")).resolve() != dataset_dir:
            raise ValueError(
                "Existing manifest belongs to a different dataset; run a full preparation"
            )
        if Path(existing_document.get("model", "")).resolve() != model_path:
            raise ValueError(
                "Existing manifest uses a different model; run a full preparation"
            )
        existing_torso = existing_document.get("fixed_torso", {})
        if body_lift_m is None:
            body_lift_m = float(existing_torso["body_lift_m"])
        if body_pitch_rad is None:
            body_pitch_rad = float(existing_torso["body_pitch_rad"])
        if not (
            np.isclose(existing_torso.get("body_lift_m", np.nan), body_lift_m)
            and np.isclose(
                existing_torso.get("body_pitch_rad", np.nan), body_pitch_rad
            )
        ):
            raise ValueError(
                "Existing manifest uses a different torso pose; run a full preparation"
            )

        requested_names = tuple(Path(name).name for name in episode_names)
        available = {path.name: path for path in all_episodes}
        missing = [name for name in requested_names if name not in available]
        if missing:
            raise FileNotFoundError(
                f"Requested episodes are missing from {dataset_dir}: {missing}"
            )
        requested_set = set(requested_names)
        episodes = [path for path in all_episodes if path.name in requested_set]
    else:
        episodes = all_episodes
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
    if body_lift_m is None:
        body_lift_m = DEFAULT_BODY_LIFT_M
    if body_pitch_rad is None:
        body_pitch_rad = DEFAULT_BODY_PITCH_RAD
    model = mujoco.MjModel.from_xml_path(str(model_path))
    upper_body_pose = fixed_upper_body_pose(body_lift_m, body_pitch_rad)
    layout_overrides = load_layout_overrides(dataset_dir)
    cache_dir = output_path.parent / ".replay_cache" / dataset_dir.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    reference_episode = all_episodes[0]
    reference_trajectory = load_trajectory(reference_episode, summary_path)
    reference_bindings = bind_joints(model, reference_trajectory.joint_names)
    reference_boundaries = load_gripper_boundaries(reference_episode)
    reference_center_frame = choose_dice_center_frame(
        reference_boundaries, reference_trajectory.frames
    )
    reference_grasp_z = float(
        _right_gripper_center_at_frame(
            model,
            reference_trajectory,
            reference_bindings,
            upper_body_pose,
            reference_center_frame,
        )[2]
    )

    records: list[dict[str, Any]] = []
    for episode_index, episode_path in enumerate(episodes):
        print(f"[{episode_index + 1}/{len(episodes)}] {episode_path.name}", flush=True)
        record: dict[str, Any] = {
            "episode": episode_path.name,
            "status": "failed",
        }
        try:
            trajectory = load_trajectory(episode_path, summary_path)
            bindings = bind_joints(model, trajectory.joint_names)
            boundaries = load_gripper_boundaries(episode_path)
            center_frame = choose_dice_center_frame(boundaries, trajectory.frames)
            layout_override = layout_overrides.get(episode_path.name, {})
            center_frame = int(layout_override.get("dice_center_frame", center_frame))
            dice_xy_offset_m = tuple(
                float(value)
                for value in layout_override.get("dice_xy_offset_m", (0.0, 0.0))
            )
            grasp_z = float(
                _right_gripper_center_at_frame(
                    model,
                    trajectory,
                    bindings,
                    upper_body_pose,
                    center_frame,
                )[2]
            )
            z_offset_m = reference_grasp_z - grasp_z
            selected = None
            last_table_safe = None
            last_rejected_metrics = None
            for placement_lift_m in PLACEMENT_LIFTS_M:
                offsets = z_offset_m + placement_lift_profile(
                    trajectory.frames, boundaries, placement_lift_m
                )
                corrected, ik_metrics = translate_right_arm_trajectory(
                    model,
                    trajectory,
                    bindings,
                    upper_body_pose,
                    offsets,
                )
                table_metrics = robot_table_metrics(
                    model, corrected, bindings, upper_body_pose
                )
                last_rejected_metrics = {
                    **ik_metrics,
                    "table": table_metrics,
                }
                if (
                    ik_metrics["max_position_error_m"] > IK_TOLERANCE_M
                    or table_metrics["contact_frames"] != 0
                ):
                    continue
                last_table_safe = (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    table_metrics,
                )
                box_pose, dice_plan, collision_metrics = find_collision_free_box_pose(
                    model,
                    corrected,
                    bindings,
                    upper_body_pose,
                    center_frame,
                    dice_xy_offset_m=dice_xy_offset_m,
                    yaw_zero_only=True,
                )
                selected = (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    table_metrics,
                    box_pose,
                    dice_plan,
                    collision_metrics,
                )
                if collision_metrics["box_wall_contacts"] == 0:
                    break

            if selected is None and last_table_safe is None:
                record.update(
                    {
                        "reason": "No IK-valid trajectory clears the table",
                        "metrics": last_rejected_metrics or {},
                    }
                )
                records.append(record)
                continue
            if selected is None:
                assert last_table_safe is not None
                (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    table_metrics,
                ) = last_table_safe
                box_pose, dice_plan, collision_metrics = find_collision_free_box_pose(
                    model,
                    corrected,
                    bindings,
                    upper_body_pose,
                    center_frame,
                    dice_xy_offset_m=dice_xy_offset_m,
                )
                selected = (
                    placement_lift_m,
                    corrected,
                    ik_metrics,
                    table_metrics,
                    box_pose,
                    dice_plan,
                    collision_metrics,
                )
            (
                placement_lift_m,
                corrected,
                ik_metrics,
                table_metrics,
                box_pose,
                dice_plan,
                collision_metrics,
            ) = selected
            if collision_metrics["box_wall_contacts"] != 0:
                box_pose, dice_plan, collision_metrics = find_collision_free_box_pose(
                    model,
                    corrected,
                    bindings,
                    upper_body_pose,
                    center_frame,
                    dice_xy_offset_m=dice_xy_offset_m,
                )
            cache_path = cache_dir / f"{episode_path.stem}_joints.npz"
            np.savez_compressed(cache_path, joint_positions=corrected.joint_positions)
            status = (
                "ok"
                if ik_metrics["max_position_error_m"] <= IK_TOLERANCE_M
                and table_metrics["contact_frames"] == 0
                and collision_metrics["box_wall_contacts"] == 0
                else "failed"
            )
            record.update(
                {
                    "status": status,
                    "frames": corrected.frames,
                    "boundaries": boundaries,
                    "dice_center_frame": center_frame,
                    "dice_xy_offset_m": list(dice_xy_offset_m),
                    "right_arm_z_offset_m": z_offset_m,
                    "placement_lift_m": placement_lift_m,
                    "cache": str(cache_path.relative_to(output_path.parent)),
                    "dice": {
                        "initial_position": dice_plan.initial_position.tolist(),
                        "initial_yaw_deg": float(np.degrees(dice_plan.initial_yaw_rad)),
                        "grasp_frame": dice_plan.grasp_frame,
                        "release_frame": dice_plan.release_frame,
                        "landing_position": dice_plan.landing_position.tolist(),
                    },
                    "box": box_pose,
                    "metrics": {
                        **ik_metrics,
                        **collision_metrics,
                        "table": table_metrics,
                    },
                }
            )
            if status != "ok":
                record["reason"] = (
                    "IK error, robot-table penetration, or box-wall contact "
                    "exceeds tolerance"
                )
        except Exception as error:  # Keep processing the remaining dataset.
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)

    if existing_document is not None:
        existing_records = {
            record["episode"]: record
            for record in existing_document.get("episodes", [])
            if "episode" in record
        }
        updated_records = {record["episode"]: record for record in records}
        records = [
            (
                updated_records[path.name]
                if path.name in updated_records
                else existing_records[path.name]
            )
            for path in all_episodes
            if path.name in updated_records or path.name in existing_records
        ]

    document = {
        "schema": LAYOUT_SCHEMA,
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "summary": str(summary_path),
        "fixed_torso": {
            "body_lift_m": body_lift_m,
            "body_pitch_rad": body_pitch_rad,
        },
        "layout_overrides": layout_overrides,
        "reference": {
            "episode": reference_episode.name,
            "dice_center_frame": reference_center_frame,
            "right_gripper_center_z_m": reference_grasp_z,
        },
        "episodes": records,
        "counts": {
            "total": len(records),
            "ok": sum(record["status"] == "ok" for record in records),
            "failed": sum(record["status"] != "ok" for record in records),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return document
