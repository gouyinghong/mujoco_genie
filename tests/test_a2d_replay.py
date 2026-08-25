from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.convert_a2d_to_mjcf import (
    A2D_ARM_JOINT_NAMES,
    DEFAULT_A2D_URDF,
    convert_a2d_urdf_to_mjcf,
)
from scripts.replay_a2d import (
    A2D_GRIPPER_JOINTS,
    DEFAULT_EPISODE,
    DEFAULT_SUMMARY,
    bind_joints,
    interpolate_joint_state,
    load_trajectory,
    validate_joint_limits,
)
from scripts.show_last_frame import prepare_last_frame
from scripts.show_zero_pose import prepare_zero_pose


@pytest.fixture(scope="module")
def converted_a2d_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, mujoco.MjModel]:
    output_path = tmp_path_factory.mktemp("a2d_mjcf") / "a2d.xml"
    result = convert_a2d_urdf_to_mjcf(DEFAULT_A2D_URDF, output_path)
    return result.output_path, mujoco.MjModel.from_xml_path(str(result.output_path))


def test_a2d_conversion_and_joint7_zero_calibration(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    output_path, model = converted_a2d_model
    assert output_path.is_file()
    assert model.nq == 32
    assert model.nv == 32
    assert model.njnt == 32
    assert model.nmocap == 2
    assert np.count_nonzero((model.geom_group == 1) & (model.geom_contype == 0)) == 39
    assert np.count_nonzero((model.geom_group == 0) & (model.geom_contype != 0)) == 39

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for parent_name, child_name, original_rpy, zero_offset in (
        ("Link6_l", "Link7_l", (-np.pi / 2, 0.0, -np.pi), np.pi / 2),
        ("Link6_r", "Link7_r", (-np.pi / 2, 0.0, np.pi), -np.pi / 2),
    ):
        parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_name)
        child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_name)
        parent_rotation = data.xmat[parent_id].reshape(3, 3)
        actual_rotation = parent_rotation.T @ data.xmat[child_id].reshape(3, 3)
        expected_rotation = (
            Rotation.from_euler("xyz", original_rpy).as_matrix()
            @ Rotation.from_rotvec((0.0, 0.0, zero_offset)).as_matrix()
        )
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-6)


def test_joint_binding_and_limits(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)
    actual_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in bindings.joint_ids
    )
    assert actual_names == A2D_ARM_JOINT_NAMES


def test_interpolation_uses_dataset_timestamps() -> None:
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    midpoint = (trajectory.times_s[10] + trajectory.times_s[11]) / 2.0
    qpos, qvel = interpolate_joint_state(trajectory, midpoint)
    np.testing.assert_allclose(
        qpos,
        (trajectory.joint_positions[10] + trajectory.joint_positions[11]) / 2.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        qvel,
        (trajectory.joint_positions[11] - trajectory.joint_positions[10])
        / (trajectory.times_s[11] - trajectory.times_s[10]),
        atol=1e-12,
    )


def test_last_frame_uses_final_action_joint(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    data = prepare_last_frame(model, trajectory, bindings, show_target=False)

    np.testing.assert_allclose(
        data.qpos[bindings.qpos_addresses],
        trajectory.joint_positions[-1],
        atol=1e-12,
    )
    for joint_name in A2D_GRIPPER_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert data.qpos[model.jnt_qposadr[joint_id]] == 0.0
    assert data.time == pytest.approx(trajectory.duration_s)


def test_zero_pose_sets_all_arm_joints_to_zero(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    data, qpos_addresses = prepare_zero_pose(model)

    assert qpos_addresses.shape == (14,)
    np.testing.assert_array_equal(data.qpos[qpos_addresses], np.zeros(14))
    np.testing.assert_array_equal(data.qvel, np.zeros(model.nv))
    assert data.time == 0.0
