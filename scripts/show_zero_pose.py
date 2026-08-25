#!/usr/bin/env python3
"""Visualize a robot with all 14 arm joints held at zero radians."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_g1_to_mjcf import ARM_JOINT_NAMES  # noqa: E402
from scripts.replay_g1 import (  # noqa: E402
    ROBOT_CONFIGS,
    RobotReplayConfig,
    ensure_model,
    set_gripper_opening,
    set_target_visibility,
)
from scripts.show_last_frame import show_last_frame  # noqa: E402


def arm_joint_names(robot_config: RobotReplayConfig) -> tuple[str, ...]:
    return robot_config.model_joint_names or ARM_JOINT_NAMES


def prepare_zero_pose(
    model: mujoco.MjModel,
    robot_config: RobotReplayConfig,
    *,
    gripper_open: float,
) -> tuple[mujoco.MjData, np.ndarray]:
    """Create a static state with every left/right arm joint at exactly zero."""

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    qpos_addresses: list[int] = []
    for name in arm_joint_names(robot_config):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Model is missing arm joint {name!r}")
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_address] = 0.0
        data.qvel[dof_address] = 0.0
        qpos_addresses.append(qpos_address)

    set_gripper_opening(model, data, gripper_open, robot_config)
    set_target_visibility(model, False)
    data.time = 0.0
    mujoco.mj_forward(model, data)
    return data, np.asarray(qpos_addresses, dtype=int)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        choices=tuple(ROBOT_CONFIGS),
        default="a2d",
        help="Robot description used for visualization (default: a2d)",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.0,
        help="G1 gripper opening in [0, 1]; A2D keeps its authored neutral pose",
    )
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument(
        "--show-joint-axes",
        action="store_true",
        help="Draw MuJoCo joint coordinate axes for zero-frame comparison",
    )
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
    data, qpos_addresses = prepare_zero_pose(
        model,
        robot_config,
        gripper_open=args.gripper_open,
    )
    np.testing.assert_allclose(data.qpos[qpos_addresses], 0.0, atol=0.0)

    print(f"Robot: {robot_config.key}")
    print(f"Model: {model_path}")
    print("Holding 14 arm joints at exactly 0 rad; close the viewer to exit.")
    for name in arm_joint_names(robot_config):
        print(f"  {name}: 0.0 rad")
    if robot_config.gripper_mode == "a2d_neutral":
        print("A2D gripper: neutral URDF pose")

    show_last_frame(
        model,
        data,
        show_collision=args.show_collision,
        show_joint_axes=args.show_joint_axes,
    )


if __name__ == "__main__":
    main()
