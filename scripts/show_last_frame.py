#!/usr/bin/env python3
"""Show and hold the final A2D trajectory frame in MuJoCo."""

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

from scripts.convert_a2d_to_mjcf import DEFAULT_A2D_MJCF, DEFAULT_A2D_URDF  # noqa: E402
from scripts.replay_a2d import (  # noqa: E402
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


def prepare_last_frame(
    model: mujoco.MjModel,
    trajectory: Trajectory,
    bindings: JointBindings,
    *,
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
        show_target=show_target,
    )
    return data


def show_static_pose(
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
            # This is a kinematic snapshot: do not call mj_step.
            viewer.sync()
            time.sleep(1.0 / 60.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_A2D_MJCF)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_A2D_URDF)
    parser.add_argument("--show-target", action="store_true")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--show-joint-axes", action="store_true")
    parser.add_argument("--rebuild-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = ensure_model(args.model, args.urdf, args.rebuild_model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    trajectory = load_trajectory(args.episode, args.summary)
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)
    data = prepare_last_frame(
        model,
        trajectory,
        bindings,
        show_target=args.show_target,
    )

    np.testing.assert_allclose(
        data.qpos[bindings.qpos_addresses],
        trajectory.joint_positions[-1],
        atol=1e-12,
    )
    print(f"Model: {model_path}")
    print(
        f"Holding final frame {trajectory.frames - 1} "
        f"at t={trajectory.duration_s:.6f} s; close the viewer to exit."
    )
    print("A2D gripper: neutral URDF pose")
    show_static_pose(
        model,
        data,
        show_collision=args.show_collision,
        show_joint_axes=args.show_joint_axes,
    )


if __name__ == "__main__":
    main()
