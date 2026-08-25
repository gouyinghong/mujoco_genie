#!/usr/bin/env python3
"""Show and hold the final frame of a retargeted trajectory in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_g1 import (  # noqa: E402
    DEFAULT_EPISODE,
    DEFAULT_SUMMARY,
    ROBOT_CONFIGS,
    JointBindings,
    RobotReplayConfig,
    Trajectory,
    apply_kinematic_pose,
    bind_joints,
    ensure_model,
    load_trajectory,
    set_target_visibility,
    validate_joint_limits,
)


def prepare_last_frame(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    robot_config: RobotReplayConfig,
    *,
    gripper_open: float,
    show_target: bool,
) -> mujoco.MjData:
    """Create MjData positioned exactly at action_joint_position[-1]."""

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    set_target_visibility(model, show_target)
    apply_kinematic_pose(
        model,
        data,
        trajectory,
        bindings,
        trajectory.duration_s,
        gripper_open=gripper_open,
        show_target=show_target,
        robot_config=robot_config,
    )
    return data


def show_last_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    show_collision: bool,
    show_joint_axes: bool = False,
) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.05, 0.0, 0.8)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.opt.geomgroup[0] = int(show_collision)
        viewer.opt.geomgroup[1] = 1
        viewer.opt.geomgroup[2] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = int(show_joint_axes)

        while viewer.is_running():
            # No mj_step(): this is a kinematic snapshot and must not fall under
            # gravity or move in response to contacts.
            viewer.sync()
            time.sleep(1.0 / 60.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        choices=tuple(ROBOT_CONFIGS),
        default="a2d",
        help="Robot description used for visualization (default: a2d)",
    )
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--show-target", action="store_true")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--rebuild-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.gripper_open <= 1.0:
        raise ValueError("gripper-open must be in [0, 1]")

    robot_config = ROBOT_CONFIGS[args.robot]
    model_path = ensure_model(
        args.model or robot_config.default_mjcf,
        args.urdf or robot_config.default_urdf,
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
    data = prepare_last_frame(
        model,
        trajectory,
        bindings,
        robot_config,
        gripper_open=args.gripper_open,
        show_target=args.show_target,
    )

    np.testing.assert_allclose(
        data.qpos[bindings.qpos_addresses],
        trajectory.joint_positions[-1],
        atol=1e-12,
    )
    print(f"Robot: {robot_config.key}")
    print(f"Model: {model_path}")
    print(
        f"Holding final frame {trajectory.frames - 1} "
        f"at t={trajectory.duration_s:.6f} s; close the viewer to exit."
    )
    if robot_config.gripper_mode == "a2d_neutral":
        print("A2D gripper: neutral URDF pose (trajectory has no gripper channel)")

    show_last_frame(model, data, show_collision=args.show_collision)


if __name__ == "__main__":
    main()
