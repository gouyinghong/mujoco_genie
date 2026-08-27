#!/usr/bin/env python3
"""Interactively tune the A2D torso pose while replaying arms and grippers.

The dice remains fixed on the tabletop. Press SPACE in the MuJoCo window to
start or restart the trajectory.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_a2d import (  # noqa: E402
    A2D_UPPER_BODY_POSE,
    DEFAULT_A2D_MJCF,
    DEFAULT_A2D_URDF,
    DEFAULT_EPISODE,
    DEFAULT_SUMMARY,
    JointBindings,
    Trajectory,
    apply_kinematic_pose,
    bind_joints,
    ensure_model,
    load_trajectory,
    set_target_visibility,
    validate_joint_limits,
)


JOINT_NAMES = (
    "joint_head_yaw",
    "joint_head_pitch",
    "joint_body_pitch",
    "joint_lift_body",
)
DEFAULT_POSE = dict(A2D_UPPER_BODY_POSE)
DICE_POSITION = np.array((0.75, 0.0, 0.8248), dtype=float)


def override_effector_commands(
    trajectory: Trajectory,
    *,
    left: float | None = None,
    right: float | None = None,
) -> Trajectory:
    """Return a trajectory with optional constant left/right gripper commands."""

    if left is None and right is None:
        return trajectory
    if trajectory.effector_positions is None:
        raise ValueError("Trajectory has no action_effector channel to override")
    for side, value in (("left", left), ("right", right)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{side} effector override must be in [0, 1]")

    commands = trajectory.effector_positions.copy()
    if left is not None:
        commands[:, 0] = left
    if right is not None:
        commands[:, 1] = right
    return replace(trajectory, effector_positions=commands)


class PoseController:
    """Thread-safe upper-body pose edited by the viewer key callback."""

    def __init__(self, model: mujoco.MjModel, initial: dict[str, float]) -> None:
        self._lock = threading.Lock()
        self._positions = dict(initial)
        self._limits: dict[str, tuple[float, float]] = {}
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Model is missing upper-body joint {name!r}")
            lower, upper = model.jnt_range[joint_id]
            self._limits[name] = (float(lower), float(upper))
            position = self._positions[name]
            if not lower <= position <= upper:
                raise ValueError(
                    f"Initial {name} position {position} is outside "
                    f"[{lower}, {upper}]"
                )

    def pose(self) -> tuple[tuple[str, float], ...]:
        with self._lock:
            return tuple((name, self._positions[name]) for name in JOINT_NAMES)

    def adjust(self, name: str, delta: float) -> None:
        with self._lock:
            lower, upper = self._limits[name]
            self._positions[name] = float(
                np.clip(self._positions[name] + delta, lower, upper)
            )
        self.print_pose()

    def print_pose(self) -> None:
        values = dict(self.pose())
        print(
            "Pose [head_yaw_deg, head_pitch_deg, body_pitch_rad, body_lift_m] = "
            f"[{np.degrees(values['joint_head_yaw']):.6f}, "
            f"{np.degrees(values['joint_head_pitch']):.6f}, "
            f"{values['joint_body_pitch']:.9f}, "
            f"{values['joint_lift_body']:.6f}]",
            flush=True,
        )


def set_static_dice_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Keep the free dice upright at a fixed point on the tabletop."""

    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    if joint_id < 0:
        return
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 3] = DICE_POSITION
    data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_address : dof_address + 6] = 0.0


def replay(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    pose_controller: PoseController,
    *,
    speed: float,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    set_target_visibility(model, False)
    replay_requested = threading.Event()

    adjustments = {
        ord("W"): ("joint_body_pitch", np.deg2rad(1.0)),
        ord("S"): ("joint_body_pitch", -np.deg2rad(1.0)),
        ord("R"): ("joint_lift_body", 0.01),
        ord("F"): ("joint_lift_body", -0.01),
        ord("I"): ("joint_head_pitch", np.deg2rad(1.0)),
        ord("K"): ("joint_head_pitch", -np.deg2rad(1.0)),
        ord("J"): ("joint_head_yaw", np.deg2rad(1.0)),
        ord("L"): ("joint_head_yaw", -np.deg2rad(1.0)),
    }

    def key_callback(keycode: int) -> None:
        if keycode == ord(" "):
            replay_requested.set()
        elif keycode in adjustments:
            pose_controller.adjust(*adjustments[keycode])

    def apply(time_s: float) -> None:
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            time_s,
            show_target=False,
            dice_plan=None,
            upper_body_pose=pose_controller.pose(),
        )
        set_static_dice_pose(model, data)
        mujoco.mj_forward(model, data)

    apply(0.0)
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        viewer.cam.lookat[:] = (0.05, 0.0, 0.8)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.opt.geomgroup[0] = 0
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1

        print("Controls: W/S torso pitch, R/F torso lift, I/K head pitch, J/L head yaw")
        print("Adjust the pose, then press SPACE in the viewer to start/restart replay.")
        pose_controller.print_pose()

        playing = False
        current_time_s = 0.0
        wall_start = 0.0
        while viewer.is_running():
            iteration_start = time.monotonic()
            if replay_requested.is_set():
                replay_requested.clear()
                playing = True
                current_time_s = 0.0
                wall_start = iteration_start
                print("Replay started.", flush=True)

            if playing:
                current_time_s = (iteration_start - wall_start) * speed
                if current_time_s >= trajectory.duration_s:
                    current_time_s = trajectory.duration_s
                    playing = False
                    print(
                        "Replay finished. Adjust the pose or press SPACE to replay.",
                        flush=True,
                    )

            apply(current_time_s)
            viewer.sync()
            remaining = 1.0 / 120.0 - (time.monotonic() - iteration_start)
            if remaining > 0.0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_EPISODE.parent,
        help=(
            "Dataset directory containing episode_000000.npz and "
            "retarget_summary.json"
        ),
    )
    parser.add_argument(
        "--episode",
        type=Path,
        help="Episode path; overrides <dataset-dir>/episode_000000.npz",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary path; overrides <dataset-dir>/retarget_summary.json",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_A2D_MJCF)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_A2D_URDF)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--left-effector",
        type=float,
        help="Override every left-gripper frame (0=closed, 1=open)",
    )
    parser.add_argument(
        "--right-effector",
        type=float,
        help="Override every right-gripper frame (0=closed, 1=open)",
    )
    parser.add_argument("--head-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--head-pitch-deg",
        type=float,
        default=float(np.degrees(DEFAULT_POSE["joint_head_pitch"])),
    )
    parser.add_argument(
        "--body-pitch-rad",
        type=float,
        default=DEFAULT_POSE["joint_body_pitch"],
    )
    parser.add_argument(
        "--body-lift-m", type=float, default=DEFAULT_POSE["joint_lift_body"]
    )
    parser.add_argument("--rebuild-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("speed must be positive")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    episode_path = (
        args.episode
        if args.episode is not None
        else dataset_dir / DEFAULT_EPISODE.name
    )
    summary_path = (
        args.summary
        if args.summary is not None
        else dataset_dir / DEFAULT_SUMMARY.name
    )

    model_path = ensure_model(args.model, args.urdf, args.rebuild_model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    trajectory = load_trajectory(episode_path, summary_path)
    trajectory = override_effector_commands(
        trajectory,
        left=args.left_effector,
        right=args.right_effector,
    )
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)
    initial = {
        "joint_head_yaw": float(np.deg2rad(args.head_yaw_deg)),
        "joint_head_pitch": float(np.deg2rad(args.head_pitch_deg)),
        "joint_body_pitch": args.body_pitch_rad,
        "joint_lift_body": args.body_lift_m,
    }
    pose_controller = PoseController(model, initial)

    print(
        f"Loaded {trajectory.frames} frames from {episode_path}; "
        "arms and action_effector will replay, dice remains on the table."
    )
    if args.left_effector is not None:
        print(f"Left gripper override: {args.left_effector:.6f}")
    if args.right_effector is not None:
        print(f"Right gripper override: {args.right_effector:.6f}")
    replay(model, trajectory, bindings, pose_controller, speed=args.speed)


if __name__ == "__main__":
    main()
