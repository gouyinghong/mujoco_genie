#!/usr/bin/env python3
"""Replay one prepared A2D episode while the dice is fully physics-driven.

The arms follow the recorded trajectory kinematically. By default each gripper
has one finite-torque drive and mechanical linkage constraints. The dice pose is written only once
at reset; grasping, carrying, release, and landing are then produced solely by
MuJoCo contacts, friction, inertia, and gravity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.a2d_batch import (  # noqa: E402
    DEFAULT_PHYSICS_LAYOUT,
    DEFAULT_REPLAY_MANIFEST,
    LAYOUT_SCHEMA,
    fixed_upper_body_pose,
    load_corrected_trajectory,
)
from scripts.a2d_closed_loop import (  # noqa: E402
    ClosedLoopGripper, add_gripper_arguments, load_physics_model, lower_grasp_trajectory,
    update_prescribed_arm_constraints,
)
from scripts.replay_a2d import (  # noqa: E402
    JointBindings,
    Trajectory,
    apply_texture_gamma,
    bind_joints,
    build_dice_replay_plan,
    interpolate_effector_state,
    interpolate_joint_state,
    set_cardboard_box_pose,
    set_gripper_command,
    set_target_visibility,
    set_upper_body_pose,
    trajectory_frame_at_time,
    validate_joint_limits,
)


DEFAULT_MANIFEST = DEFAULT_REPLAY_MANIFEST
PHYSICS_LAYOUT_SCHEMA = "a2d_physics_replay_layouts.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_gripper_arguments(parser)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--metrics-output", type=Path,
                        help="Write deterministic validation metrics to a new JSON file")
    parser.add_argument("--exit-when-finished", action="store_true",
                        help="Close the viewer after one complete rollout (for batch replay)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--physics-layout", type=Path,
        help="Defaults to the physics_layout stored in the prepared manifest",
    )
    parser.add_argument(
        "--ignore-physics-layout",
        action="store_true",
        help="Use the original deterministic layout instead of a searched layout",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Zero-based index among successful records in replay_layouts.json",
    )
    parser.add_argument(
        "--episode",
        help="Select by filename, for example episode_000012.npz",
    )
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument(
        "--min-gripper-openness",
        type=float,
        default=0.0,
        help=(
            "Clamp normalized gripper commands to this minimum; 0 is fully "
            "closed and larger values reduce closing strength/travel"
        ),
    )
    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=0.4,
        help="Physics settling time before trajectory playback",
    )
    parser.add_argument(
        "--post-rollout-s",
        type=float,
        default=1.0,
        help="Continue physics after the final trajectory sample",
    )
    parser.add_argument(
        "--dice-mass-kg",
        type=float,
        help="Override dice mass while preserving its inertia ratios",
    )
    parser.add_argument(
        "--dice-sliding-friction",
        type=float,
        help="Override the dice sliding-friction coefficient",
    )
    parser.add_argument(
        "--dice-linear-damping",
        type=float,
        default=0.0,
        help="Viscous damping for the three translational free-joint DOFs",
    )
    parser.add_argument(
        "--dice-angular-damping",
        type=float,
        default=0.0,
        help="Viscous damping for the three rotational free-joint DOFs",
    )
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--box-texture-gamma", type=float, default=0.65)
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


def successful_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [record for record in manifest["episodes"] if record.get("status") == "ok"]


def load_physics_layout(
    path: Path, dataset_dir: Path, episode: str
) -> dict[str, Any] | None:
    path = path.expanduser().resolve()
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema") != PHYSICS_LAYOUT_SCHEMA:
        raise ValueError(f"Unsupported physics-layout schema in {path}")
    return document.get("datasets", {}).get(dataset_dir.name, {}).get(episode)


def choose_record(
    records: list[dict[str, Any]], episode_index: int, episode: str | None
) -> tuple[int, dict[str, Any]]:
    if episode is not None:
        matches = [
            (index, record)
            for index, record in enumerate(records)
            if record["episode"] == episode
        ]
        if not matches:
            raise ValueError(f"No successful manifest record named {episode!r}")
        return matches[0]
    if not 0 <= episode_index < len(records):
        raise ValueError(f"episode-index must be in [0, {len(records) - 1}]")
    return episode_index, records[episode_index]


def set_robot_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: Trajectory,
    bindings: JointBindings,
    trajectory_time_s: float,
    upper_body_pose: tuple[tuple[str, float], ...],
    min_gripper_openness: float,
    *,
    moving: bool,
    gripper: ClosedLoopGripper | None = None,
    initialize_gripper: bool = False,
) -> None:
    """Write only robot DOFs, leaving the free dice state untouched."""

    joint_positions, joint_velocities = interpolate_joint_state(
        trajectory, trajectory_time_s
    )
    data.qpos[bindings.qpos_addresses] = joint_positions
    data.qvel[bindings.dof_addresses] = joint_velocities if moving else 0.0
    set_upper_body_pose(model, data, upper_body_pose)
    update_prescribed_arm_constraints(model, data)

    effector_state = interpolate_effector_state(trajectory, trajectory_time_s)
    if effector_state is None:
        raise ValueError("Physical replay requires action_effector")
    openness, openness_velocity = effector_state
    limited = np.maximum(openness, min_gripper_openness)
    # A clamped gripper must stop instead of retaining the source closing speed.
    limited_velocity = openness_velocity.copy()
    limited_velocity[
        (openness <= min_gripper_openness) & (openness_velocity < 0.0)
    ] = 0.0
    if not moving:
        limited_velocity[:] = 0.0
    if gripper is None:
        set_gripper_command(model, data, limited, limited_velocity)
    else:
        gripper.command(data, openness, minimum=min_gripper_openness,
                        initialize=initialize_gripper)


def set_initial_dice_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    if joint_id < 0:
        raise ValueError("Model is missing dice_free_joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
    data.qvel[dof_address : dof_address + 6] = 0.0


def configure_dice_dynamics(
    model: mujoco.MjModel,
    *,
    mass_kg: float | None = None,
    sliding_friction: float | None = None,
    linear_damping: float = 0.0,
    angular_damping: float = 0.0,
) -> None:
    """Apply optional runtime-only dice dynamics overrides."""

    if mass_kg is not None and mass_kg <= 0.0:
        raise ValueError("dice-mass-kg must be positive")
    if sliding_friction is not None and sliding_friction < 0.0:
        raise ValueError("dice-sliding-friction must be non-negative")
    if linear_damping < 0.0 or angular_damping < 0.0:
        raise ValueError("dice damping values must be non-negative")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dice_collision"
    )
    if min(body_id, joint_id, geom_id) < 0:
        raise ValueError("Model is missing the dynamic dice definitions")

    if mass_kg is not None:
        mass_scale = mass_kg / float(model.body_mass[body_id])
        model.body_mass[body_id] = mass_kg
        model.body_inertia[body_id] *= mass_scale
    if sliding_friction is not None:
        model.geom_friction[geom_id, 0] = sliding_friction

    dof_address = int(model.jnt_dofadr[joint_id])
    model.dof_damping[dof_address : dof_address + 3] = linear_damping
    model.dof_damping[dof_address + 3 : dof_address + 6] = angular_damping


def dice_diagnostics(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dof_address = int(model.jnt_dofadr[joint_id])
    speed = float(np.linalg.norm(data.qvel[dof_address : dof_address + 3]))
    position = data.xpos[body_id]
    return (
        f"dice=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m, "
        f"speed={speed:.4f} m/s, contacts={data.ncon}"
    )


def configure_collision_debug(model: mujoco.MjModel) -> None:
    """Make robot/object collision geoms visible as translucent red overlays."""

    debug_mask = (
        np.isin(model.geom_group, (0, 3))
        & (model.geom_contype != 0)
        & (model.geom_contype != 8)
    )
    model.geom_rgba[debug_mask] = (1.0, 0.1, 0.1, 0.25)


def first_dice_object_contact(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[str, float] | None:
    """Return the first die contact that is not table/box support contact."""

    dice_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dice_collision"
    )
    for contact in data.contact:
        geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
        if dice_geom_id not in (geom0, geom1):
            continue
        other_geom_id = geom1 if geom0 == dice_geom_id else geom0
        other_body_id = int(model.geom_bodyid[other_geom_id])
        other_body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, other_body_id
        )
        if other_body in ("table", "cardboard_box"):
            continue
        return other_body or f"body_{other_body_id}", float(contact.dist)
    return None


def main() -> None:
    args = parse_args()
    if args.metrics_output is not None and args.metrics_output.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {args.metrics_output}")
    if args.speed <= 0.0:
        raise ValueError("speed must be positive")
    if not 0.0 <= args.min_gripper_openness <= 1.0:
        raise ValueError("min-gripper-openness must be in [0, 1]")
    if args.settle_time_s < 0.0 or args.post_rollout_s < 0.0:
        raise ValueError("settle-time-s and post-rollout-s must be non-negative")
    if args.box_texture_gamma <= 0.0:
        raise ValueError("box-texture-gamma must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    args.physics_layout = args.physics_layout or Path(
        manifest.get("physics_layout", DEFAULT_PHYSICS_LAYOUT)
    )
    records = successful_records(manifest)
    record_index, record = choose_record(records, args.episode_index, args.episode)
    dataset_dir = Path(manifest["dataset_dir"])
    summary_path = Path(manifest["summary"])
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
    apply_texture_gamma(model, "cardboard_box_texture", args.box_texture_gamma)
    set_target_visibility(model, False)
    if args.show_collision:
        configure_collision_debug(model)

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
        dataset_dir / record["episode"], summary_path, cache_path
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

    physics_layout = (
        None
        if args.ignore_physics_layout
        else load_physics_layout(
            args.physics_layout, dataset_dir, str(record["episode"])
        )
    )
    initial_dice_position = dice_plan.initial_position.copy()
    initial_dice_quaternion = dice_plan.initial_quaternion.copy()
    if physics_layout is not None:
        initial_dice_position = np.asarray(
            physics_layout["dice_position"], dtype=float
        )
        yaw_rad = np.deg2rad(float(physics_layout["dice_yaw_deg"]))
        initial_dice_quaternion = np.array(
            (np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)),
            dtype=float,
        )

    if physics_layout is not None and 'dice_quaternion_wxyz' in physics_layout:
        initial_dice_quaternion = np.asarray(physics_layout['dice_quaternion_wxyz'], dtype=float)

    data = mujoco.MjData(model)

    def reset() -> None:
        mujoco.mj_resetData(model, data)
        set_robot_target(
            model,
            data,
            trajectory,
            bindings,
            0.0,
            upper_body_pose,
            args.min_gripper_openness,
            moving=False,
            gripper=gripper,
            initialize_gripper=True,
        )
        set_initial_dice_pose(
            model, data, initial_dice_position, initial_dice_quaternion
        )
        mujoco.mj_forward(model, data)
        settle_steps = int(round(args.settle_time_s / float(model.opt.timestep)))
        for _ in range(settle_steps):
            set_robot_target(
                model,
                data,
                trajectory,
                bindings,
                0.0,
                upper_body_pose,
                args.min_gripper_openness,
                moving=False,
                gripper=gripper,
            )
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

    reset()
    print(
        f"Physical replay {record_index + 1}/{len(records)}: {record['episode']}; "
        f"frames={trajectory.frames}, grasp={dice_plan.grasp_frame}, "
        f"release={dice_plan.release_frame}, timestep={model.opt.timestep:.4f} s"
    )
    print(
        "Dice is dynamic after reset; no grasp attachment or scripted fall is used."
    )
    print(f"Gripper control: {args.gripper_control}")
    print(f"Right arm downward offset (runtime only): {args.grasp_lower_m:g} m")
    if gripper is not None:
        print(f"One drive per hand: kp={args.gripper_kp:g}, kv={args.gripper_kv:g}, "
              f"torque limit={args.gripper_max_torque:g} N m, close bias={args.gripper_close_bias:g}")
    print(
        f"Minimum gripper openness: {args.min_gripper_openness:.3f} "
        "(0=fully closed, increase if the grip is too tight)"
    )
    dice_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "dice"
    )
    print(f"Fingertip sliding friction: {model.geom_friction[model.geom('right_narrow_fingertip_collision').id, 0]:g}")
    print(f"Arm contact mode: {args.arm_contact_mode}; contact impratio={model.opt.impratio:g}")
    print(f"Gripper release mode: {args.gripper_release_mode}")
    print(
        f"Dice dynamics: mass={model.body_mass[dice_body_id]:.5f} kg, "
        f"linear_damping={args.dice_linear_damping:g}, "
        f"angular_damping={args.dice_angular_damping:g}"
    )
    if physics_layout is None:
        print("Physical layout: original deterministic dice pose")
    else:
        print(
            f"Physical layout: {args.physics_layout} "
            f"[{physics_layout.get('status', 'unknown')}], "
            f"score={float(physics_layout.get('score', 0.0)):.3f}"
        )
    print(f"Initial {dice_diagnostics(model, data)}")

    def validate_rollout() -> dict[str, Any]:
        from scripts.search_a2d_physics_layout import candidate_metrics

        # Same stepping/control path used by the layout evaluator, including reset.
        yaw = np.rad2deg(2 * np.arctan2(initial_dice_quaternion[3], initial_dice_quaternion[0]))
        metrics = candidate_metrics(
            model, trajectory, bindings, upper_body_pose, dice_plan,
            initial_dice_position, float(yaw), settle_time_s=args.settle_time_s,
            min_gripper_openness=args.min_gripper_openness, gripper=gripper,
            post_rollout_s=args.post_rollout_s, initial_quaternion=initial_dice_quaternion,
        )
        print(json.dumps(metrics, indent=2))
        if args.metrics_output is not None:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.metrics_output.with_suffix(args.metrics_output.suffix + ".tmp")
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    json.dump(metrics, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                # Publish a complete file atomically without replacing an existing result.
                os.link(temporary, args.metrics_output)
            finally:
                temporary.unlink(missing_ok=True)
        return metrics

    if args.headless:
        validate_rollout()
        return

    from mujoco import viewer as mujoco_viewer

    command_lock = threading.Lock()
    commands: list[str] = []

    def key_callback(keycode: int) -> None:
        command = {ord(" "): "toggle", 257: "restart"}.get(keycode)
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

        paused = not args.start_immediately
        trajectory_time_s = 0.0
        wall_accumulator_s = 0.0
        previous_wall_time = time.monotonic()
        final_reported = False
        first_contact_reported = False
        print(
            "Controls: SPACE pause/resume and print current frame, "
            "ENTER reset episode."
        )

        total_duration_s = trajectory.duration_s + args.post_rollout_s
        timestep = float(model.opt.timestep)
        while viewer.is_running():
            iteration_start = time.monotonic()
            wall_delta = iteration_start - previous_wall_time
            previous_wall_time = iteration_start
            with command_lock:
                pending = commands.copy()
                commands.clear()

            for command in pending:
                if command == "restart":
                    reset()
                    trajectory_time_s = 0.0
                    wall_accumulator_s = 0.0
                    final_reported = False
                    first_contact_reported = False
                    paused = not args.start_immediately
                    print(f"Reset: {record['episode']}; {dice_diagnostics(model, data)}")
                else:
                    paused = not paused
                    frame = trajectory_frame_at_time(
                        trajectory, min(trajectory_time_s, trajectory.duration_s)
                    )
                    print(
                        f"{'Paused' if paused else 'Resumed'}: frame "
                        f"{frame}/{trajectory.frames - 1}, "
                        f"trajectory_time={trajectory_time_s:.3f} s, "
                        f"{dice_diagnostics(model, data)}",
                        flush=True,
                    )
                wall_delta = 0.0

            if not paused and trajectory_time_s < total_duration_s:
                wall_accumulator_s += wall_delta * args.speed
                while wall_accumulator_s >= timestep:
                    command_time_s = min(trajectory_time_s, trajectory.duration_s)
                    set_robot_target(
                        model,
                        data,
                        trajectory,
                        bindings,
                        command_time_s,
                        upper_body_pose,
                        args.min_gripper_openness,
                        moving=trajectory_time_s < trajectory.duration_s,
                        gripper=gripper,
                    )
                    mujoco.mj_step(model, data)
                    trajectory_time_s = min(
                        trajectory_time_s + timestep, total_duration_s
                    )
                    wall_accumulator_s -= timestep
                    if not first_contact_reported:
                        contact = first_dice_object_contact(model, data)
                        if contact is not None:
                            body_name, distance = contact
                            frame = trajectory_frame_at_time(
                                trajectory,
                                min(trajectory_time_s, trajectory.duration_s),
                            )
                            print(
                                "First robot/dice contact: "
                                f"frame={frame}, time={trajectory_time_s:.3f} s, "
                                f"body={body_name}, penetration="
                                f"{max(0.0, -distance) * 1000.0:.3f} mm",
                                flush=True,
                            )
                            first_contact_reported = True
                    if trajectory_time_s >= total_duration_s:
                        break

            if trajectory_time_s >= total_duration_s and not final_reported:
                paused = True
                final_reported = True
                print(f"Finished: {dice_diagnostics(model, data)}", flush=True)
                if args.exit_when_finished:
                    break

            viewer.opt.geomgroup[0] = int(args.show_collision)
            viewer.opt.geomgroup[1] = 1
            viewer.opt.geomgroup[2] = 1
            if args.show_collision:
                viewer.opt.geomgroup[3] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.sync()
            remaining = 1.0 / 120.0 - (time.monotonic() - iteration_start)
            if remaining > 0.0:
                time.sleep(remaining)

    if args.exit_when_finished and not final_reported:
        # Closing a batch viewer early means stop the batch, not silently skip.
        raise SystemExit(130)
    if args.metrics_output is not None:
        print("Validating a fresh deterministic rollout after viewer playback.")
        validate_rollout()


if __name__ == "__main__":
    main()
