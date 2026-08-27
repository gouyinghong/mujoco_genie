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
    A2D_UPPER_BODY_POSE,
    DEFAULT_EPISODE,
    DEFAULT_SUMMARY,
    GRIPPER_LINK_ORDER,
    apply_kinematic_pose,
    bind_joints,
    build_dice_replay_plan,
    gripper_joint_positions,
    infer_grasp_frames,
    interpolate_effector_state,
    interpolate_joint_state,
    load_trajectory,
    validate_joint_limits,
)
from scripts.replay_a2d_torso_adjust import (
    PoseController,
    override_effector_commands,
    set_static_dice_pose,
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
    assert model.nq == 41
    assert model.nv == 40
    assert model.njnt == 35
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


def test_table_dimensions_position_and_color(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    table_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "table"
    )
    table_top_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"
    )

    assert table_body_id >= 0
    assert table_top_id >= 0
    np.testing.assert_allclose(model.body_pos[table_body_id], (0.90, 0.0, 0.0))
    np.testing.assert_allclose(model.geom_pos[table_top_id], (0.0, 0.0, 0.77))
    np.testing.assert_allclose(model.geom_size[table_top_id], (0.45, 0.70, 0.03))
    np.testing.assert_allclose(model.geom_rgba[table_top_id], (0.92, 0.92, 0.92, 1.0))


def test_dice_is_textured_free_body_on_table(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dice_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    dice_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dice_visual_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dice_visual"
    )
    dice_collision_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dice_collision"
    )

    assert dice_body_id >= 0
    assert dice_joint_id >= 0
    assert dice_visual_id >= 0
    assert dice_collision_id >= 0
    assert model.jnt_type[dice_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_allclose(model.body_pos[dice_body_id], (0.75, 0.0, 0.8248))
    np.testing.assert_allclose(
        model.geom_size[dice_collision_id], (0.0248, 0.0248, 0.0248)
    )
    assert model.geom_matid[dice_visual_id] >= 0


def test_robot_only_conversion_omits_table(tmp_path: Path) -> None:
    output_path = tmp_path / "a2d_robot_only.xml"
    convert_a2d_urdf_to_mjcf(
        DEFAULT_A2D_URDF,
        output_path,
        include_table=False,
    )
    model = mujoco.MjModel.from_xml_path(str(output_path))

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice") == -1


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

    assert trajectory.effector_positions is not None
    effector_state = interpolate_effector_state(trajectory, midpoint)
    assert effector_state is not None
    effector, effector_velocity = effector_state
    np.testing.assert_allclose(
        effector,
        (trajectory.effector_positions[10] + trajectory.effector_positions[11])
        / 2.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        effector_velocity,
        (
            trajectory.effector_positions[11]
            - trajectory.effector_positions[10]
        )
        / (trajectory.times_s[11] - trajectory.times_s[10]),
        atol=1e-12,
    )


def test_grasp_event_and_dice_pick_place(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    assert infer_grasp_frames(trajectory) == (1, 54, 75)

    plan = build_dice_replay_plan(model, trajectory, bindings)
    assert plan is not None
    assert plan.side == "right"
    np.testing.assert_allclose(
        plan.initial_position,
        (0.6988463543, 0.1164891874, 0.9892485517),
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        plan.initial_quaternion, (1.0, 0.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        plan.landing_position,
        (0.7163038805, 0.1357286645, 0.8248),
        atol=1e-9,
    )

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    apply_kinematic_pose(
        model,
        data,
        trajectory,
        bindings,
        trajectory.duration_s,
        show_target=False,
        dice_plan=plan,
    )
    dice_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dice_qpos_address = int(model.jnt_qposadr[dice_joint_id])
    np.testing.assert_allclose(
        data.qpos[dice_qpos_address : dice_qpos_address + 3],
        plan.landing_position,
        atol=1e-12,
    )


def test_body_lift_moves_inferred_dice_grasp_position(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dataset = Path("datasets/fixed_spine3_to_g1_0723_add_effector_after")
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )
    bindings = bind_joints(model, trajectory.joint_names)
    original = build_dice_replay_plan(model, trajectory, bindings)
    assert original is not None
    lift_delta = 0.8248 - original.initial_position[2]
    adjusted_pose = tuple(
        (
            name,
            position + lift_delta if name == "joint_lift_body" else position,
        )
        for name, position in A2D_UPPER_BODY_POSE
    )
    adjusted = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=adjusted_pose,
    )
    assert adjusted is not None

    np.testing.assert_allclose(adjusted.initial_position[2], 0.8248, atol=1e-12)
    np.testing.assert_array_equal(
        adjusted.initial_quaternion, (1.0, 0.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        adjusted.initial_position[:2], original.initial_position[:2], atol=1e-12
    )
    np.testing.assert_allclose(
        adjusted.release_position[2],
        original.release_position[2] + lift_delta,
        atol=1e-12,
    )


def test_torso_adjust_replays_effectors_and_keeps_dice_on_table(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    custom_pose = {
        "joint_head_yaw": np.deg2rad(5.0),
        "joint_head_pitch": np.deg2rad(20.0),
        "joint_body_pitch": 0.25,
        "joint_lift_body": 0.20,
    }
    controller = PoseController(model, custom_pose)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    apply_kinematic_pose(
        model,
        data,
        trajectory,
        bindings,
        trajectory.duration_s,
        show_target=False,
        dice_plan=None,
        upper_body_pose=controller.pose(),
    )
    set_static_dice_pose(model, data)
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(
        data.qpos[bindings.qpos_addresses], trajectory.joint_positions[-1]
    )
    for joint_name, expected in custom_pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(expected)
    dice_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dice_qpos_address = int(model.jnt_qposadr[dice_joint_id])
    np.testing.assert_allclose(
        data.qpos[dice_qpos_address : dice_qpos_address + 7],
        (0.75, 0.0, 0.8248, 1.0, 0.0, 0.0, 0.0),
    )


def test_torso_adjust_can_hold_right_gripper_fully_closed() -> None:
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    assert trajectory.effector_positions is not None
    overridden = override_effector_commands(trajectory, right=0.0)

    np.testing.assert_array_equal(
        overridden.effector_positions[:, 0], trajectory.effector_positions[:, 0]
    )
    np.testing.assert_array_equal(
        overridden.effector_positions[:, 1], np.zeros(trajectory.frames)
    )
    assert not np.shares_memory(
        overridden.effector_positions, trajectory.effector_positions
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
    assert trajectory.effector_positions is not None
    for side_index, side in enumerate(("left", "right")):
        wide_positions = gripper_joint_positions(
            trajectory.effector_positions[-1, side_index]
        )
        for finger, sign in (("wide", 1.0), ("narrow", -1.0)):
            for link, expected in zip(
                GRIPPER_LINK_ORDER, sign * wide_positions, strict=True
            ):
                joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{finger}{link}_joint",
                )
                assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(
                    expected
                )
    for joint_name, expected in A2D_UPPER_BODY_POSE:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(expected)
    assert data.time == pytest.approx(trajectory.duration_s)


def test_zero_pose_sets_all_arm_joints_to_zero(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    data, qpos_addresses = prepare_zero_pose(model)

    assert qpos_addresses.shape == (14,)
    np.testing.assert_array_equal(data.qpos[qpos_addresses], np.zeros(14))
    for joint_name, expected in A2D_UPPER_BODY_POSE:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert data.qpos[model.jnt_qposadr[joint_id]] == pytest.approx(expected)
    np.testing.assert_array_equal(data.qvel, np.zeros(model.nv))
    assert data.time == 0.0
