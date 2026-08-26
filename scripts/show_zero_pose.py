#!/usr/bin/env python3
"""Visualize A2D with all 14 arm joints held at zero radians."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_a2d_to_mjcf import (  # noqa: E402
    A2D_ARM_JOINT_NAMES,
    DEFAULT_A2D_MJCF,
    DEFAULT_A2D_URDF,
)
from scripts.replay_a2d import (  # noqa: E402
    ensure_model,
    set_gripper_neutral,
    set_target_visibility,
    set_upper_body_pose,
)
from scripts.show_last_frame import show_static_pose  # noqa: E402


def prepare_zero_pose(model: mujoco.MjModel) -> tuple[mujoco.MjData, np.ndarray]:
    """Create a static state with every A2D arm joint at exactly zero."""

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    qpos_addresses: list[int] = []
    for name in A2D_ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Model is missing arm joint {name!r}")
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_address] = 0.0
        data.qvel[dof_address] = 0.0
        qpos_addresses.append(qpos_address)

    set_upper_body_pose(model, data)
    set_gripper_neutral(model, data)
    set_target_visibility(model, False)
    data.time = 0.0
    mujoco.mj_forward(model, data)
    return data, np.asarray(qpos_addresses, dtype=int)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_A2D_MJCF)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_A2D_URDF)
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
    model_path = ensure_model(args.model, args.urdf, args.rebuild_model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data, qpos_addresses = prepare_zero_pose(model)
    np.testing.assert_allclose(data.qpos[qpos_addresses], 0.0, atol=0.0)

    print(f"Model: {model_path}")
    print("Holding 14 A2D arm joints at exactly 0 rad; close the viewer to exit.")
    for name in A2D_ARM_JOINT_NAMES:
        print(f"  {name}: 0.0 rad")
    print("A2D gripper: neutral URDF pose")
    show_static_pose(
        model,
        data,
        show_collision=args.show_collision,
        show_joint_axes=args.show_joint_axes,
    )


if __name__ == "__main__":
    main()
