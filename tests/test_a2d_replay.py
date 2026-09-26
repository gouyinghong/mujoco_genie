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
    DICE_HALF_EXTENT_M,
    DICE_TABLE_CENTER_Z,
    GRIPPER_LINK_ORDER,
    apply_kinematic_pose,
    apply_texture_gamma,
    bind_joints,
    build_dice_replay_plan,
    disable_dice,
    gripper_joint_positions,
    infer_grasp_frames,
    interpolate_effector_state,
    interpolate_joint_state,
    load_trajectory,
    set_cardboard_box_pose,
    trajectory_frame_at_time,
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
    assert np.count_nonzero((model.geom_group == 0) & (model.geom_contype != 0)) == 29

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


def test_textured_cardboard_box_asset_can_be_added_separately(
    tmp_path: Path,
) -> None:
    default_path = tmp_path / "default.xml"
    box_path = tmp_path / "with_box.xml"
    convert_a2d_urdf_to_mjcf(DEFAULT_A2D_URDF, default_path)
    convert_a2d_urdf_to_mjcf(
        DEFAULT_A2D_URDF,
        box_path,
        include_cardboard_box=True,
    )
    default_model = mujoco.MjModel.from_xml_path(str(default_path))
    box_model = mujoco.MjModel.from_xml_path(str(box_path))

    assert (
        mujoco.mj_name2id(
            default_model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box"
        )
        == -1
    )
    box_body_id = mujoco.mj_name2id(
        box_model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box"
    )
    visual_id = mujoco.mj_name2id(
        box_model, mujoco.mjtObj.mjOBJ_GEOM, "cardboard_box_visual"
    )
    base_collision_id = mujoco.mj_name2id(
        box_model, mujoco.mjtObj.mjOBJ_GEOM, "cardboard_box_collision_base"
    )
    mesh_id = mujoco.mj_name2id(
        box_model, mujoco.mjtObj.mjOBJ_MESH, "cardboard_box_mesh"
    )
    material_id = mujoco.mj_name2id(
        box_model, mujoco.mjtObj.mjOBJ_MATERIAL, "cardboard_box_material"
    )
    texture_id = mujoco.mj_name2id(
        box_model, mujoco.mjtObj.mjOBJ_TEXTURE, "cardboard_box_texture"
    )

    assert (
        min(
            box_body_id,
            visual_id,
            base_collision_id,
            mesh_id,
            material_id,
            texture_id,
        )
        >= 0
    )
    assert box_model.ngeom == default_model.ngeom + 6
    np.testing.assert_allclose(
        box_model.body_pos[box_body_id], (0.65582, 0.03264, 0.8)
    )
    np.testing.assert_allclose(
        box_model.body_quat[box_body_id],
        (0.906307787, 0.0, 0.0, 0.422618262),
        atol=1e-9,
    )
    assert box_model.geom_matid[visual_id] == material_id
    assert box_model.geom_dataid[visual_id] == mesh_id
    assert texture_id in box_model.mat_texid[material_id]
    assert box_model.mat_emission[material_id] == pytest.approx(0.2)
    assert box_model.mat_specular[material_id] == pytest.approx(0.05)
    assert box_model.mat_shininess[material_id] == pytest.approx(0.02)
    np.testing.assert_allclose(
        box_model.geom_pos[base_collision_id], (0.0, 0.0, 0.001)
    )
    np.testing.assert_allclose(
        box_model.geom_size[base_collision_id], (0.12, 0.08, 0.001)
    )
    assert box_model.geom_contype[visual_id] == 0
    box_geom_ids = np.flatnonzero(box_model.geom_bodyid == box_body_id)
    collision_geom_ids = box_geom_ids[box_model.geom_contype[box_geom_ids] != 0]
    assert collision_geom_ids.size == 5

    texture_address = int(box_model.tex_adr[texture_id])
    texture_size = int(
        box_model.tex_width[texture_id]
        * box_model.tex_height[texture_id]
        * box_model.tex_nchannel[texture_id]
    )
    original_texture = box_model.tex_data[
        texture_address : texture_address + texture_size
    ].copy()
    assert apply_texture_gamma(box_model, "cardboard_box_texture", 0.65)
    corrected_texture = box_model.tex_data[
        texture_address : texture_address + texture_size
    ]
    assert float(np.mean(corrected_texture)) > float(np.mean(original_texture))

    assert set_cardboard_box_pose(
        box_model, x=0.64615, y=0.04115, yaw_deg=0.0
    )
    np.testing.assert_allclose(
        box_model.body_pos[box_body_id], (0.64615, 0.04115, 0.8)
    )
    np.testing.assert_allclose(
        box_model.body_quat[box_body_id], (1.0, 0.0, 0.0, 0.0)
    )


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
    np.testing.assert_allclose(
        model.body_pos[dice_body_id], (0.75, 0.0, DICE_TABLE_CENTER_Z)
    )
    assert model.geom_type[dice_collision_id] == mujoco.mjtGeom.mjGEOM_MESH
    np.testing.assert_allclose(
        model.geom_size[dice_collision_id], (DICE_HALF_EXTENT_M,) * 3
    )
    assert model.geom_matid[dice_visual_id] >= 0


def test_dice_can_be_disabled_without_removing_table(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    model_path, _ = converted_a2d_model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    dice_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    dice_geom_ids = np.flatnonzero(model.geom_bodyid == dice_body_id)
    dice_material_ids = np.unique(model.geom_matid[dice_geom_ids])
    dice_material_ids = dice_material_ids[dice_material_ids >= 0]

    assert disable_dice(model)
    assert table_body_id >= 0
    assert dice_geom_ids.size == 2
    np.testing.assert_array_equal(model.geom_rgba[dice_geom_ids, 3], 0.0)
    np.testing.assert_array_equal(model.mat_rgba[dice_material_ids, 3], 0.0)
    np.testing.assert_array_equal(model.geom_contype[dice_geom_ids], 0)
    np.testing.assert_array_equal(model.geom_conaffinity[dice_geom_ids], 0)


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


def test_replay_time_reports_trajectory_and_source_frames() -> None:
    dataset = Path("pico_to_g1_pipeline/outputs/fixed_spine3_to_g1_0723_complete")
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )

    assert trajectory_frame_at_time(trajectory, 0.0) == 0
    assert trajectory_frame_at_time(trajectory, trajectory.times_s[37]) == 37
    midpoint = (trajectory.times_s[37] + trajectory.times_s[38]) / 2.0
    assert trajectory_frame_at_time(trajectory, midpoint) == 37
    assert trajectory_frame_at_time(trajectory, trajectory.duration_s) == 109
    assert trajectory.episode_frame_indices[37] == 37
    assert trajectory.source_frame_indices[37] == 189


def test_grasp_event_and_dice_pick_place(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    trajectory = load_trajectory(DEFAULT_EPISODE, DEFAULT_SUMMARY)
    bindings = bind_joints(model, trajectory.joint_names)
    assert infer_grasp_frames(trajectory) == (1, 31, 73)

    plan = build_dice_replay_plan(model, trajectory, bindings)
    assert plan is not None
    assert plan.side == "right"
    np.testing.assert_allclose(
        plan.initial_position,
        (0.7829626892, -0.0961581924, 0.8910232761),
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        plan.initial_quaternion, (1.0, 0.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(
        plan.landing_position,
        (0.6472271958, 0.0682514306, DICE_TABLE_CENTER_Z),
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


def test_dice_on_table_with_safe_body_lift_has_continuous_grasp(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dataset = Path("pico_to_g1_pipeline/outputs/fixed_spine3_to_g1_0723_complete")
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )
    bindings = bind_joints(model, trajectory.joint_names)
    upper_body_pose = tuple(
        (name, 0.215 if name == "joint_lift_body" else position)
        for name, position in A2D_UPPER_BODY_POSE
    )
    plan = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=upper_body_pose,
        dice_on_table=True,
    )
    assert plan is not None
    assert plan.grasp_start_frame == 31
    assert plan.grasp_frame == 41
    np.testing.assert_allclose(
        plan.initial_position[2], DICE_TABLE_CENTER_Z, atol=1e-12
    )
    np.testing.assert_array_equal(
        plan.initial_quaternion, (1.0, 0.0, 0.0, 0.0)
    )

    data = mujoco.MjData(model)
    dice_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    dice_qpos_address = int(model.jnt_qposadr[dice_joint_id])
    for time_s in (0.0, plan.grasp_start_s, plan.grasp_s):
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            time_s,
            show_target=False,
            dice_plan=plan,
            upper_body_pose=upper_body_pose,
        )
        np.testing.assert_allclose(
            data.qpos[dice_qpos_address : dice_qpos_address + 3],
            plan.initial_position,
            atol=1e-12,
        )


def test_dice_can_align_faces_with_gripper_closing_axis(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dataset = Path("pico_to_g1_pipeline/outputs/fixed_spine3_to_g1_0723_complete")
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )
    bindings = bind_joints(model, trajectory.joint_names)
    upper_body_pose = tuple(
        (name, 0.215 if name == "joint_lift_body" else position)
        for name, position in A2D_UPPER_BODY_POSE
    )
    plan = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=upper_body_pose,
        dice_on_table=True,
        align_dice_to_gripper=True,
    )
    assert plan is not None

    assert np.degrees(plan.initial_yaw_rad) == pytest.approx(
        -20.5933787114, abs=1e-6
    )

    np.testing.assert_allclose(
        plan.initial_position[2], DICE_TABLE_CENTER_Z, atol=1e-12
    )
    np.testing.assert_allclose(
        plan.initial_quaternion,
        (
            np.cos(plan.initial_yaw_rad / 2.0),
            0.0,
            0.0,
            np.sin(plan.initial_yaw_rad / 2.0),
        ),
        atol=1e-12,
    )


def test_dice_xy_can_use_frame_37_fingertip_center(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dataset = Path("pico_to_g1_pipeline/outputs/fixed_spine3_to_g1_0723_complete")
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )
    bindings = bind_joints(model, trajectory.joint_names)
    upper_body_pose = tuple(
        (name, 0.215 if name == "joint_lift_body" else position)
        for name, position in A2D_UPPER_BODY_POSE
    )
    plan = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=upper_body_pose,
        dice_on_table=True,
        align_dice_to_gripper=True,
        dice_center_frame=37,
    )
    assert plan is not None

    assert plan.position_frame == 37
    np.testing.assert_allclose(
        plan.initial_position,
        (0.7650962004, -0.1158210821, DICE_TABLE_CENTER_Z),
        atol=1e-6,
    )
    assert np.degrees(plan.initial_yaw_rad) == pytest.approx(
        -20.5933787114, abs=1e-6
    )


def test_dice_xy_offset_moves_table_pose_without_changing_height(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dataset = Path("pico_to_g1_pipeline/outputs/fixed_spine3_to_g1_0723_complete")
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )
    bindings = bind_joints(model, trajectory.joint_names)
    base = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        dice_on_table=True,
        align_dice_to_gripper=True,
        dice_center_frame=37,
    )
    shifted = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        dice_on_table=True,
        align_dice_to_gripper=True,
        dice_center_frame=37,
        dice_xy_offset_m=(0.01, 0.02),
    )
    assert base is not None and shifted is not None

    np.testing.assert_allclose(
        shifted.initial_position - base.initial_position,
        (0.01, 0.02, 0.0),
        atol=1e-12,
    )


def test_new_return_trajectory_avoids_simplified_fingertip_wall_contacts(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "with_box.xml"
    convert_a2d_urdf_to_mjcf(
        DEFAULT_A2D_URDF,
        model_path,
        include_cardboard_box=True,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    dataset = Path(
        "pico_to_g1_pipeline/outputs/fixed_spine3_to_g1_0723_complete"
    )
    trajectory = load_trajectory(
        dataset / "episode_000000.npz", dataset / "retarget_summary.json"
    )
    bindings = bind_joints(model, trajectory.joint_names)
    upper_body_pose = tuple(
        (name, 0.215 if name == "joint_lift_body" else position)
        for name, position in A2D_UPPER_BODY_POSE
    )
    plan = build_dice_replay_plan(
        model,
        trajectory,
        bindings,
        upper_body_pose=upper_body_pose,
        dice_on_table=True,
        align_dice_to_gripper=True,
        dice_center_frame=37,
    )
    assert plan is not None

    np.testing.assert_allclose(
        plan.landing_position,
        (0.6476980610, 0.0434957074, 0.832),
        atol=3e-3,
    )
    wall_ids = {
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"cardboard_box_collision_{wall}",
        )
        for wall in (
            "wall_x_negative",
            "wall_x_positive",
            "wall_y_negative",
            "wall_y_positive",
        )
    }
    data = mujoco.MjData(model)
    wall_contact_frames = []
    for frame, time_s in enumerate(trajectory.times_s):
        apply_kinematic_pose(
            model,
            data,
            trajectory,
            bindings,
            float(time_s),
            show_target=False,
            dice_plan=plan,
            upper_body_pose=upper_body_pose,
        )
        if any(
            int(contact.geom[0]) in wall_ids or int(contact.geom[1]) in wall_ids
            for contact in data.contact
        ):
            wall_contact_frames.append(frame)
    assert wall_contact_frames == []


def test_body_lift_moves_inferred_dice_grasp_position(
    converted_a2d_model: tuple[Path, mujoco.MjModel],
) -> None:
    _, model = converted_a2d_model
    dataset = Path("datasets/fixed_spine3_to_g1_0723_add_effector_after")
    if not (dataset / "retarget_summary.json").is_file():
        pytest.skip("optional fixed_spine3_to_g1_0723_add_effector_after dataset missing")
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
