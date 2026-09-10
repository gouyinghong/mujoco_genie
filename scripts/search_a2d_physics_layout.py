#!/usr/bin/env python3
"""Search a separate, physics-only dice pose for one prepared A2D episode."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (  # noqa: E402
    fixed_upper_body_pose,
    load_corrected_trajectory,
)
from scripts.a2d_closed_loop import (  # noqa: E402
    ClosedLoopGripper, add_gripper_arguments, load_physics_model, loop_error_m, lower_grasp_trajectory,
)
from scripts.replay_a2d import (  # noqa: E402
    _grasp_frame_pose,
    bind_joints,
    build_dice_replay_plan,
    set_cardboard_box_pose,
    validate_joint_limits,
)
from scripts.replay_a2d_physics import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PHYSICS_LAYOUT,
    PHYSICS_LAYOUT_SCHEMA,
    choose_record,
    configure_dice_dynamics,
    load_manifest,
    set_initial_dice_pose,
    set_robot_target,
    successful_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gripper_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_PHYSICS_LAYOUT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode", help="For example episode_000000.npz")
    parser.add_argument("--objective", choices=("retention", "low-slip"), default="retention",
                        help="low-slip checks final placement and ranks relative motion, not only retention")
    parser.add_argument("--xy-range-m", type=float, default=0.03)
    parser.add_argument("--xy-step-m", type=float, default=0.01)
    parser.add_argument("--yaw-range-deg", type=float, default=30.0)
    parser.add_argument("--yaw-step-deg", type=float, default=15.0)
    parser.add_argument("--settle-time-s", type=float, default=0.3)
    parser.add_argument("--min-gripper-openness", type=float, default=0.0)
    parser.add_argument("--dice-mass-kg", type=float)
    parser.add_argument("--dice-sliding-friction", type=float)
    parser.add_argument("--dice-linear-damping", type=float, default=0.0)
    parser.add_argument("--dice-angular-damping", type=float, default=0.0)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run a second half-step search around the best coarse candidate",
    )
    return parser.parse_args()


def inclusive_grid(radius: float, step: float) -> np.ndarray:
    count = int(np.floor((2.0 * radius) / step + 1e-9))
    values = -radius + np.arange(count + 1, dtype=float) * step
    if values[-1] < radius - 1e-9:
        values = np.append(values, radius)
    return values


def candidate_rank(metrics: dict[str, Any], objective: str) -> tuple:
    if objective == "retention":
        return (metrics["success"], metrics["score"])
    if objective != "low-slip":
        raise ValueError(f"Unknown search objective: {objective}")
    # Normalize each displacement by its explicit acceptance tolerance. A
    # successful placement outranks a motionless die left on the table.
    motion = max(
        metrics["max_dice_translation_in_gripper_m"] / metrics["low_slip_translation_tolerance_m"],
        metrics["max_dice_rotation_in_gripper_deg"] / metrics["low_slip_rotation_tolerance_deg"],
    )
    return (metrics["low_slip_pick_and_place_success"], metrics["pick_and_place_success"], -motion)


def candidate_metrics(
    model: mujoco.MjModel,
    trajectory,
    bindings,
    upper_body_pose,
    dice_plan,
    position: np.ndarray,
    yaw_deg: float,
    *,
    settle_time_s: float,
    min_gripper_openness: float,
    gripper: ClosedLoopGripper | None = None,
    post_rollout_s: float | None = None,
    initial_quaternion: np.ndarray | None = None,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    set_robot_target(
        model,
        data,
        trajectory,
        bindings,
        0.0,
        upper_body_pose,
        min_gripper_openness,
        moving=False,
        gripper=gripper,
        initialize_gripper=True,
    )
    yaw_rad = np.deg2rad(yaw_deg)
    quaternion = np.array(
        (np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)),
        dtype=float,
    )
    if initial_quaternion is not None:
        quaternion = np.asarray(initial_quaternion, dtype=float)
    set_initial_dice_pose(model, data, position, quaternion)
    mujoco.mj_forward(model, data)

    timestep = float(model.opt.timestep)
    for _ in range(int(round(settle_time_s / timestep))):
        set_robot_target(
            model,
            data,
            trajectory,
            bindings,
            0.0,
            upper_body_pose,
            min_gripper_openness,
            moving=False,
            gripper=gripper,
        )
        mujoco.mj_step(model, data)

    dice_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "dice"
    )
    dice_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dice_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dice_collision"
    )
    narrow_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        f"{dice_plan.side}_narrow_fingertip_collision",
    )
    wide_ids = {
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{dice_plan.side}_wide_fingertip_{pad}_collision",
        )
        for pad in ("lower", "upper")
    }
    dof_address = int(model.jnt_dofadr[dice_joint_id])
    first_narrow_s: float | None = None
    first_wide_s: float | None = None
    simultaneous_steps = 0
    closing_contact_steps = 0
    carry_heights: list[float] = []
    carry_distances: list[float] = []
    carry_bilateral: list[bool] = []
    carry_supported: list[bool] = []
    retained: list[bool] = []
    loss_steps = 0
    max_loss_steps = 0
    base_id = model.body(f"{dice_plan.side}_base_link").id
    reference_position = None
    reference_rotation = None
    max_relative_translation = 0.0
    max_relative_rotation = 0.0
    max_relative_drop = 0.0
    max_speed = 0.0
    max_loop_error = 0.0
    max_torque = 0.0
    max_robot_support_penetration = 0.0
    support_body_ids = {model.body("table").id, model.body("cardboard_box").id}
    contact_force = np.zeros(6)
    time_s = 0.0
    # Linear interpolation starts opening at the sample BEFORE the first
    # increased command. Check retention right up to that point.
    opening_start_s = float(trajectory.times_s[max(0, dice_plan.release_frame - 1)])
    evaluation_end_s = max(dice_plan.grasp_s, opening_start_s)

    while time_s < evaluation_end_s:
        set_robot_target(
            model,
            data,
            trajectory,
            bindings,
            time_s,
            upper_body_pose,
            min_gripper_openness,
            moving=True,
            gripper=gripper,
        )
        mujoco.mj_step(model, data)
        time_s += timestep

        narrow_contact = False
        wide_contact = False
        support_contact = False
        for contact_index, contact in enumerate(data.contact):
            geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
            if dice_geom_id not in (geom0, geom1):
                bodies = {int(model.geom_bodyid[geom0]), int(model.geom_bodyid[geom1])}
                if bodies & support_body_ids and bodies - support_body_ids:
                    max_robot_support_penetration = max(max_robot_support_penetration, -float(contact.dist))
                continue
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            if contact_force[0] <= 0.001:
                continue
            other = geom1 if geom0 == dice_geom_id else geom0
            narrow_contact = narrow_contact or other == narrow_id
            wide_contact = wide_contact or other in wide_ids
            support_contact = support_contact or model.geom_bodyid[other] in support_body_ids
        if narrow_contact and first_narrow_s is None:
            first_narrow_s = time_s
        if wide_contact and first_wide_s is None:
            first_wide_s = time_s
        if dice_plan.grasp_start_s <= time_s <= dice_plan.grasp_s + 0.15:
            closing_contact_steps += 1
            simultaneous_steps += int(narrow_contact and wide_contact)

        speed = float(
            np.linalg.norm(data.qvel[dof_address : dof_address + 3])
        )
        max_speed = max(max_speed, speed)
        if gripper is not None:
            max_loop_error = max(max_loop_error, loop_error_m(model, data))
            max_torque = max(max_torque, float(np.max(np.abs(data.actuator_force[gripper.actuator_ids]))))
        if dice_plan.grasp_s + 0.15 <= time_s <= opening_start_s:
            retained.append(narrow_contact and wide_contact)
            loss_steps = 0 if retained[-1] else loss_steps + 1
            max_loss_steps = max(max_loss_steps, loss_steps)
        if dice_plan.grasp_s <= time_s <= opening_start_s:
            base_rotation = data.xmat[base_id].reshape(3, 3)
            relative_position = base_rotation.T @ (data.xpos[dice_body_id] - data.xpos[base_id])
            relative_rotation = base_rotation.T @ data.xmat[dice_body_id].reshape(3, 3)
            if reference_position is None:
                reference_position = relative_position.copy()
                reference_rotation = relative_rotation.copy()
            displacement = relative_position - reference_position
            max_relative_translation = max(max_relative_translation, float(np.linalg.norm(displacement)))
            max_relative_drop = max(max_relative_drop, float(-(base_rotation @ displacement)[2]))
            angle = np.arccos(np.clip((np.trace(reference_rotation.T @ relative_rotation) - 1) / 2, -1, 1))
            max_relative_rotation = max(max_relative_rotation, float(np.degrees(angle)))
        if dice_plan.grasp_s + 0.15 <= time_s <= dice_plan.release_s - 0.15:
            carry_heights.append(float(data.xpos[dice_body_id, 2]))
            carry_bilateral.append(narrow_contact and wide_contact)
            carry_supported.append(support_contact)
            gripper_position, _ = _grasp_frame_pose(
                model, data, dice_plan.side
            )
            carry_distances.append(
                float(np.linalg.norm(data.xpos[dice_body_id] - gripper_position))
            )

    heights = np.asarray(carry_heights, dtype=float)
    distances = np.asarray(carry_distances, dtype=float)
    if not heights.size:
        raise ValueError("Episode has no carry interval after closure and before release")
    carry_height_p10 = float(np.quantile(heights, 0.1))
    carry_height_median = float(np.median(heights))
    carry_distance_p90 = float(np.quantile(distances, 0.9))
    release_error = float(
        np.linalg.norm(data.xpos[dice_body_id] - dice_plan.release_position)
    )
    contact_delta = (
        None
        if first_narrow_s is None or first_wide_s is None
        else abs(first_narrow_s - first_wide_s)
    )
    simultaneous_fraction = (
        simultaneous_steps / closing_contact_steps
        if closing_contact_steps
        else 0.0
    )
    both_contacted = contact_delta is not None
    legacy_success = bool(
        both_contacted
        and contact_delta <= 0.05
        and carry_height_p10 >= 0.86
        and carry_distance_p90 <= 0.10
        and release_error <= 0.12
        and max_speed <= 2.0
    )
    # A slowly closing gripper may touch one side long before the other. Judge
    # actual carrying: loaded contacts on BOTH fingers, no table/box support,
    # meaningful lift and continued proximity through the carry interval.
    success = bool(
        np.mean(carry_bilateral) >= 0.95
        and np.mean(carry_supported) <= 0.05
        and heights.max() - position[2] >= 0.08
        and carry_distance_p90 <= 0.06
        and release_error <= 0.08
        and max_speed <= 2.0
        and max_robot_support_penetration <= 0.001
        and bool(retained) and all(retained)
    )
    score = (
        600.0 * (carry_height_p10 - 0.83)
        + 300.0 * (carry_height_median - 0.83)
        - 20.0 * (contact_delta if contact_delta is not None else 1.0)
        - 12.0 * carry_distance_p90
        - 8.0 * release_error
        - 0.25 * max_speed
        + 2.0 * simultaneous_fraction
    )
    result = {
        "success": success,
        "legacy_layout_success": legacy_success,
        "score": float(score),
        "first_narrow_contact_s": first_narrow_s,
        "first_wide_contact_s": first_wide_s,
        "contact_time_delta_s": contact_delta,
        "simultaneous_contact_fraction": float(simultaneous_fraction),
        "carry_height_p10_m": carry_height_p10,
        "carry_height_median_m": carry_height_median,
        "carry_height_max_m": float(heights.max()),
        "carry_bilateral_contact_fraction": float(np.mean(carry_bilateral)),
        "opening_start_s": opening_start_s,
        "retention_bilateral_contact_fraction": float(np.mean(retained)) if retained else 0.0,
        "max_preopening_contact_loss_s": max_loss_steps * timestep,
        "max_dice_translation_in_gripper_m": max_relative_translation,
        "max_dice_rotation_in_gripper_deg": max_relative_rotation,
        "max_dice_drop_relative_to_gripper_m": max_relative_drop,
        "carry_support_contact_fraction": float(np.mean(carry_supported)),
        "carry_gripper_distance_p90_m": carry_distance_p90,
        "release_position_error_m": release_error,
        "max_dice_speed_m_s": float(max_speed),
        "final_position": data.xpos[dice_body_id].tolist(),
        "max_loop_error_m": max_loop_error if gripper is not None else None,
        "max_drive_torque_nm": max_torque if gripper is not None else None,
        "max_robot_support_penetration_m": max_robot_support_penetration,
    }
    if post_rollout_s is not None:
        last_release_contact_s = None
        release_contact_steps = 0
        while time_s < trajectory.duration_s + post_rollout_s:
            set_robot_target(
                model, data, trajectory, bindings, min(time_s, trajectory.duration_s),
                upper_body_pose, min_gripper_openness,
                moving=time_s < trajectory.duration_s, gripper=gripper,
            )
            mujoco.mj_step(model, data)
            time_s += timestep
            if gripper is not None:
                max_loop_error = max(max_loop_error, loop_error_m(model, data))
                max_torque = max(max_torque, float(np.max(np.abs(data.actuator_force[gripper.actuator_ids]))))
            for contact in data.contact:
                geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
                if dice_geom_id in (geom0, geom1):
                    continue
                bodies = {int(model.geom_bodyid[geom0]), int(model.geom_bodyid[geom1])}
                if bodies & support_body_ids and bodies - support_body_ids:
                    max_robot_support_penetration = max(max_robot_support_penetration, -float(contact.dist))
            loaded_finger_contact = False
            for contact_index, contact in enumerate(data.contact):
                geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
                if dice_geom_id not in (geom0, geom1):
                    continue
                other = geom1 if geom0 == dice_geom_id else geom0
                if other != narrow_id and other not in wide_ids:
                    continue
                mujoco.mj_contactForce(model, data, contact_index, contact_force)
                loaded_finger_contact |= contact_force[0] > 0.001
            if loaded_finger_contact:
                last_release_contact_s = time_s
                release_contact_steps += 1
        result["last_finger_contact_after_opening_s"] = last_release_contact_s
        result["postopening_finger_contact_duration_s"] = release_contact_steps * timestep
        result["last_finger_contact_delay_from_opening_s"] = (
            max(0.0, last_release_contact_s - opening_start_s)
            if last_release_contact_s is not None else 0.0
        )
        mujoco.mj_forward(model, data)
        result["post_rollout_position"] = data.xpos[dice_body_id].tolist()
        box_id = model.body("cardboard_box").id
        box_rotation = data.xmat[box_id].reshape(3, 3)
        local_position = box_rotation.T @ (data.xpos[dice_body_id] - data.xpos[box_id])
        # Conservative bounds of the rotated 6 cm cube, not only its center.
        half_extent = np.abs(box_rotation.T @ data.xmat[dice_body_id].reshape(3, 3)) @ np.full(3, 0.03)
        inner_half = np.array([
            model.geom(f"cardboard_box_collision_wall_{axis}_positive").pos[i]
            - model.geom(f"cardboard_box_collision_wall_{axis}_positive").size[i]
            for i, axis in enumerate(("x", "y"))
        ])
        result["landed_in_box"] = bool(
            np.all(np.abs(local_position[:2]) + half_extent[:2] <= inner_half + 0.001)
            and -0.001 <= local_position[2] - half_extent[2] <= 0.01
            and np.linalg.norm(data.qvel[dof_address:dof_address + 3]) < 0.01
        )
    result["physics_warnings"] = int(sum(w.number for w in data.warning))
    result["max_loop_error_m"] = max_loop_error if gripper is not None else None
    result["max_drive_torque_nm"] = max_torque if gripper is not None else None
    result["max_robot_support_penetration_m"] = max_robot_support_penetration
    result["success"] = bool(result["success"] and result["physics_warnings"] == 0
                             and (gripper is None or max_loop_error < 0.001)
                             and max_robot_support_penetration <= 0.001)
    # Retention alone does not imply a steady grasp: a die can slide or rotate
    # while remaining in loaded contact with both fingers. These are explicit
    # engineering tolerances, measured from closure completion, not zero-slip.
    result["low_slip_translation_tolerance_m"] = 0.002
    result["low_slip_rotation_tolerance_deg"] = 5.0
    result["low_slip_grasp_success"] = bool(
        result["success"] and reference_position is not None
        and max_relative_translation <= result["low_slip_translation_tolerance_m"]
        and max_relative_rotation <= result["low_slip_rotation_tolerance_deg"]
    )
    if post_rollout_s is not None:
        result["pick_and_place_success"] = result["success"] and result["landed_in_box"]
        result["low_slip_pick_and_place_success"] = result["low_slip_grasp_success"] and result["landed_in_box"]
    return result


def save_result(
    path: Path,
    dataset_dir: Path,
    episode: str,
    record: dict[str, Any],
) -> None:
    path = path.expanduser().resolve()
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("schema") != PHYSICS_LAYOUT_SCHEMA:
            raise ValueError(f"Unsupported existing physics-layout schema in {path}")
    else:
        document = {"schema": PHYSICS_LAYOUT_SCHEMA, "datasets": {}}
    document.setdefault("datasets", {}).setdefault(dataset_dir.name, {})[
        episode
    ] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.xy_range_m < 0.0 or args.yaw_range_deg < 0.0:
        raise ValueError("Search ranges must be non-negative")
    if args.xy_step_m <= 0.0 or args.yaw_step_deg <= 0.0:
        raise ValueError("Search steps must be positive")
    if not 0.0 <= args.min_gripper_openness <= 1.0:
        raise ValueError("min-gripper-openness must be in [0, 1]")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    records = successful_records(manifest)
    _, record = choose_record(records, args.episode_index, args.episode)
    dataset_dir = Path(manifest["dataset_dir"])
    model = load_physics_model(
        manifest["model"], gripper_control=args.gripper_control,
        gripper_kp=args.gripper_kp, gripper_kv=args.gripper_kv,
        gripper_max_torque=args.gripper_max_torque,
        gripper_sliding_friction=args.gripper_sliding_friction,
        arm_contact_mode=args.arm_contact_mode,
        physics_timestep=args.physics_timestep,
        contact_impratio=args.contact_impratio,
    )
    gripper = (ClosedLoopGripper(model, args.gripper_close_bias, args.gripper_release_mode)
               if args.gripper_control == "closed-loop" else None)
    configure_dice_dynamics(
        model,
        mass_kg=args.dice_mass_kg,
        sliding_friction=args.dice_sliding_friction,
        linear_damping=args.dice_linear_damping,
        angular_damping=args.dice_angular_damping,
    )
    torso = manifest["fixed_torso"]
    upper_body_pose = fixed_upper_body_pose(
        float(torso["body_lift_m"]), float(torso["body_pitch_rad"])
    )
    cache_value = record.get("cache")
    cache_path = (
        None
        if cache_value is None
        else (manifest_path.parent / cache_value).resolve()
    )
    trajectory = load_corrected_trajectory(
        dataset_dir / record["episode"], Path(manifest["summary"]), cache_path
    )
    bindings = bind_joints(model, trajectory.joint_names)
    trajectory = lower_grasp_trajectory(model, trajectory, bindings, upper_body_pose, args.grasp_lower_m)
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
        raise ValueError(f"Could not build a dice plan for {record['episode']}")

    evaluated: dict[tuple[float, float, float], dict[str, Any]] = {}
    success_field = "low_slip_pick_and_place_success" if args.objective == "low-slip" else "success"

    def evaluate_grid(
        center_xy: np.ndarray,
        center_yaw_deg: float,
        xy_radius: float,
        xy_step: float,
        yaw_radius: float,
        yaw_step: float,
    ) -> None:
        dx_values = inclusive_grid(xy_radius, xy_step)
        dy_values = inclusive_grid(xy_radius, xy_step)
        yaw_values = inclusive_grid(yaw_radius, yaw_step)
        total = len(dx_values) * len(dy_values) * len(yaw_values)
        count = 0
        for dx in dx_values:
            for dy in dy_values:
                for yaw_delta in yaw_values:
                    position = dice_plan.initial_position.copy()
                    position[:2] = center_xy + (dx, dy)
                    yaw_deg = center_yaw_deg + yaw_delta
                    key = (
                        round(float(position[0]), 7),
                        round(float(position[1]), 7),
                        round(float(yaw_deg), 5),
                    )
                    if key not in evaluated:
                        metrics = candidate_metrics(
                            model,
                            trajectory,
                            bindings,
                            upper_body_pose,
                            dice_plan,
                            position,
                            yaw_deg,
                            settle_time_s=args.settle_time_s,
                            min_gripper_openness=args.min_gripper_openness,
                            gripper=gripper,
                            post_rollout_s=1.0 if args.objective == "low-slip" else None,
                        )
                        evaluated[key] = {
                            "position": position.tolist(),
                            "yaw_deg": float(yaw_deg),
                            "metrics": metrics,
                        }
                    count += 1
                    if count % 25 == 0 or count == total:
                        best_now = max(
                            evaluated.values(), key=lambda item: candidate_rank(item["metrics"], args.objective)
                        )
                        print(
                            f"  {count}/{total}: best score="
                            f"{best_now['metrics']['score']:.3f}, "
                            f"{args.objective} success={best_now['metrics'][success_field]}",
                            flush=True,
                        )

    base_xy = dice_plan.initial_position[:2].copy()
    base_yaw_deg = float(np.degrees(dice_plan.initial_yaw_rad))
    print(
        f"Searching {record['episode']} around xy={base_xy}, "
        f"yaw={base_yaw_deg:.3f} deg",
        flush=True,
    )
    evaluate_grid(
        base_xy,
        base_yaw_deg,
        args.xy_range_m,
        args.xy_step_m,
        args.yaw_range_deg,
        args.yaw_step_deg,
    )
    best = max(evaluated.values(), key=lambda item: candidate_rank(item["metrics"], args.objective))
    if args.refine:
        evaluate_grid(
            np.asarray(best["position"][:2]),
            float(best["yaw_deg"]),
            args.xy_step_m,
            args.xy_step_m / 2.0,
            args.yaw_step_deg,
            args.yaw_step_deg / 2.0,
        )
        best = max(evaluated.values(), key=lambda item: candidate_rank(item["metrics"], args.objective))

    metrics = best["metrics"]
    output_record = {
        "status": "ok" if metrics[success_field] else "best_effort",
        "objective": args.objective,
        "score": float(metrics["score"]),
        "dice_position": best["position"],
        "dice_yaw_deg": float(best["yaw_deg"]),
        "deterministic_position": dice_plan.initial_position.tolist(),
        "deterministic_yaw_deg": base_yaw_deg,
        "xy_offset_from_deterministic_m": (
            np.asarray(best["position"][:2]) - base_xy
        ).tolist(),
        "yaw_offset_from_deterministic_deg": float(best["yaw_deg"] - base_yaw_deg),
        "min_gripper_openness": float(args.min_gripper_openness),
        "gripper_control": args.gripper_control,
        "grasp_lower_m": args.grasp_lower_m,
        "arm_contact_mode": args.arm_contact_mode,
        "physics_timestep_s": float(model.opt.timestep),
        "contact_impratio": float(model.opt.impratio),
        "gripper_parameters": {
            "kp": args.gripper_kp, "kv": args.gripper_kv,
            "max_torque_nm": args.gripper_max_torque, "close_bias": args.gripper_close_bias,
            "sliding_friction_override": args.gripper_sliding_friction,
            "release_mode": args.gripper_release_mode,
        },
        "dice_dynamics": {
            "mass_kg": args.dice_mass_kg,
            "sliding_friction": args.dice_sliding_friction,
            "linear_damping": float(args.dice_linear_damping),
            "angular_damping": float(args.dice_angular_damping),
        },
        "metrics": metrics,
        "model": str(Path(manifest["model"]).resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_result(
        args.output, dataset_dir, str(record["episode"]), output_record
    )
    print(json.dumps(output_record, indent=2), flush=True)
    print(f"Saved independent physics layout: {args.output}", flush=True)
    if not metrics[success_field]:
        print(
            f"WARNING: no grasp passed the {args.objective} objective; the saved pose "
            "is the best candidate and is not suitable for training yet.",
            flush=True,
        )


if __name__ == "__main__":
    main()
