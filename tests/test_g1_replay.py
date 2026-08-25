from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.convert_g1_to_mjcf import DEFAULT_URDF, convert_urdf_to_mjcf
from scripts.convert_a2d_to_mjcf import (
    A2D_ARM_JOINT_NAMES,
    DEFAULT_A2D_URDF,
    convert_a2d_urdf_to_mjcf,
)
from scripts.replay_g1 import (
    A2D_ROBOT_CONFIG,
    DEFAULT_EPISODE,
    DEFAULT_SUMMARY,
    bind_joints,
    evaluate_trajectory,
    interpolate_joint_state,
    load_trajectory,
    set_gripper_opening,
    validate_joint_limits,
)
from scripts.show_last_frame import prepare_last_frame
from scripts.show_zero_pose import prepare_zero_pose


@pytest.fixture(scope="module")
def converted_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, mujoco.MjModel]:
    output_path = tmp_path_factory.mktemp("g1_mjcf") / "g1.xml"
    result = convert_urdf_to_mjcf(DEFAULT_URDF, output_path)
    return result.output_path, mujoco.MjModel.from_xml_path(str(result.output_path))


@pytest.fixture(scope="module")
def converted_a2d_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, mujoco.MjModel]:
    output_path = tmp_path_factory.mktemp("a2d_mjcf") / "a2d.xml"
    result = convert_a2d_urdf_to_mjcf(DEFAULT_A2D_URDF, output_path)
    return result.output_path, mujoco.MjModel.from_xml_path(str(result.output_path))


def test_conversion_preserves_visuals_collisions_and_names(
    converted_model: tuple[Path, mujoco.MjModel],
) -> None:
    output_path, model = converted_model
    assert output_path.is_file()
    assert model.nq == 34
    assert model.nv == 34
    assert model.njnt == 34
    assert model.nmocap == 2
    assert model.neq == 2
    assert np.count_nonzero((model.geom_group == 1) & (model.geom_contype == 0)) == 61
    assert np.count_nonzero((model.geom_group == 0) & (model.geom_contype != 0)) == 32
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "arm_base_link"
    ) >= 0
    assert mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_eef_actual"
    ) >= 0


def test_joint_binding_is_name_based_and_limits_are_valid(
    converted_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    validate_joint_limits(model, trajectory, bindings)

    assert bindings.qpos_addresses[:7].tolist() == list(range(2, 9))
    # Left gripper joints occupy qpos 9..16, so the right arm is not contiguous
    # with the left arm in the imported model.
    assert bindings.qpos_addresses[7:].tolist() == list(range(17, 24))


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


def test_gripper_opening_sets_mimic_pair(
    converted_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_model
    data = mujoco.MjData(model)
    set_gripper_opening(model, data, 1.0)
    for inner_name, outer_name in (
        ("idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"),
        ("idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"),
    ):
        inner_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, inner_name)
        outer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, outer_name)
        assert data.qpos[model.jnt_qposadr[inner_id]] == pytest.approx(-np.pi / 4)
        assert data.qpos[model.jnt_qposadr[outer_id]] == pytest.approx(np.pi / 4)


def test_fk_matches_retarget_output(
    converted_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    report = evaluate_trajectory(model, trajectory, bindings, gripper_open=1.0)

    for side in ("left", "right"):
        assert report["fk_position_error_m"][side]["max"] < 5e-4
        assert report["fk_orientation_error_deg"][side]["max"] < 0.1


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

    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names, A2D_ARM_JOINT_NAMES)
    validate_joint_limits(model, trajectory, bindings)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for parent_name, child_name, original_rpy, zero_offset in (
        ("Link6_l", "Link7_l", (-np.pi / 2, 0.0, -np.pi), np.pi / 2),
        ("Link6_r", "Link7_r", (-np.pi / 2, 0.0, np.pi), -np.pi / 2),
    ):
        parent_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, parent_name
        )
        child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_name)
        parent_rotation = data.xmat[parent_id].reshape(3, 3)
        actual_rotation = parent_rotation.T @ data.xmat[child_id].reshape(3, 3)
        expected_rotation = (
            Rotation.from_euler("xyz", original_rpy).as_matrix()
            @ Rotation.from_rotvec((0.0, 0.0, zero_offset)).as_matrix()
        )
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-6)


def test_a2d_last_frame_uses_final_action_joint(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names, A2D_ARM_JOINT_NAMES)
    data = prepare_last_frame(
        model,
        trajectory,
        bindings,
        A2D_ROBOT_CONFIG,
        gripper_open=1.0,
        show_target=False,
    )

    np.testing.assert_allclose(
        data.qpos[bindings.qpos_addresses],
        trajectory.joint_positions[-1],
        atol=1e-12,
    )
    assert data.time == pytest.approx(trajectory.duration_s)


def test_a2d_zero_pose_sets_all_arm_joints_to_zero(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    data, qpos_addresses = prepare_zero_pose(
        model,
        A2D_ROBOT_CONFIG,
        gripper_open=0.0,
    )

    assert qpos_addresses.shape == (14,)
    np.testing.assert_array_equal(data.qpos[qpos_addresses], np.zeros(14))
    np.testing.assert_array_equal(data.qvel, np.zeros(model.nv))
    assert data.time == 0.0
