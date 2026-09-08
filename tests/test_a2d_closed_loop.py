from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.a2d_closed_loop import ClosedLoopGripper, load_physics_model, loop_error_m, lower_grasp_trajectory
from scripts.prepare_a2d_grasp_hold import insert_grasp_hold, prepare
from scripts.replay_a2d import gripper_joint_positions


MODEL = "assets/A2D_Omnipicker/A2D_with_box.xml"


def test_closed_linkage_opens_and_returns_without_disconnecting() -> None:
    model = load_physics_model(MODEL)
    gripper = ClosedLoopGripper(model, close_bias=0)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    # Isolate the hand mechanism from scene/arm collisions for this mechanism test.
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    fixed_q = np.setdiff1d(np.arange(model.nq), gripper.qpos_addresses)
    fixed_v = np.setdiff1d(np.arange(model.nv), gripper.dof_addresses)
    assert model.nu == 2
    assert model.neq == 10
    for openness in (0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0):
        for _ in range(300):
            data.qpos[fixed_q] = model.qpos0[fixed_q]
            data.qvel[fixed_v] = 0
            gripper.command(data, np.full(2, openness))
            mujoco.mj_step(model, data)
            assert np.max(np.abs(data.actuator_force)) <= 1 + 1e-12
        mujoco.mj_forward(model, data)
        assert loop_error_m(model, data) < 1e-4
        for side in ("left", "right"):
            wide = np.array([data.joint(f"{side}_wide{i}_joint").qpos[0] for i in (1, 3, 4, 2)])
            narrow = np.array([data.joint(f"{side}_narrow{i}_joint").qpos[0] for i in (1, 3, 4, 2)])
            np.testing.assert_allclose(wide, -narrow, atol=0.005)
            np.testing.assert_allclose(wide, gripper_joint_positions(openness), atol=0.06)
    assert sum(w.number for w in data.warning) == 0


def test_control_changes_only_drive_targets_and_reset_initializes_fingers() -> None:
    model = load_physics_model(MODEL)
    gripper = ClosedLoopGripper(model)
    data = mujoco.MjData(model)
    gripper.command(data, np.ones(2), initialize=True)
    before = data.qpos.copy()
    data.qvel[:] = 0.12
    gripper.command(data, np.full(2, 0.35))
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(data.qvel, 0.12)
    assert np.all(data.ctrl < 0.35 * np.pi / 4)
    gripper.command(data, np.full(2, 0.35), minimum=0.3)
    np.testing.assert_allclose(data.ctrl, 0.3 * np.pi / 4)
    mujoco.mj_resetData(model, data)
    gripper.command(data, np.ones(2), initialize=True)
    np.testing.assert_array_equal(data.qpos, before)


def test_hold_preserves_arm_samples_and_delays_all_later_timestamps() -> None:
    n = 12
    arrays = {
        "local_timestamps_ns": 100_000_000_000 + np.arange(n) * 100_000_000,
        "action_joint_position": np.arange(n * 14).reshape(n, 14).astype(float),
        "action_effector": np.column_stack([np.ones(n), [1, 1, 1, .8, .6, .35, .35, .35, .5, .7, 1, 1]]),
        "source_frame_index": np.arange(n) + 20,
        "scalar": np.array(3),
    }
    before = {k: v.copy() for k, v in arrays.items()}
    result, indices = insert_grasp_hold(arrays, close_start=2, close_end=5,
                                       hold_frame=4, side=1, duration_s=2)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    np.testing.assert_array_equal(result["action_joint_position"], arrays["action_joint_position"][indices])
    np.testing.assert_array_equal(result["action_joint_position"][4:25], np.repeat(arrays["action_joint_position"][[4]], 21, axis=0))
    np.testing.assert_array_equal(result["local_timestamps_ns"][25:], arrays["local_timestamps_ns"][5:] + 2_000_000_000)
    np.testing.assert_allclose(result["action_effector"][4:25, 1], np.linspace(1, .35, 21))
    np.testing.assert_array_equal(result["action_effector"][25:, 1], arrays["action_effector"][5:, 1])
    assert np.all(np.diff(result["local_timestamps_ns"]) > 0)


def test_prepared_episode_is_separate_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest_path = Path("datasets/replay_layouts.json")
    manifest = json.loads(manifest_path.read_text())
    episode = "episode_000000.npz"
    record = next(r for r in manifest["episodes"] if r["episode"] == episode)
    source = Path(manifest["dataset_dir"]) / episode
    source_cache = manifest_path.parent / record["cache"]
    inputs = [source, source_cache, manifest_path]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    output_dir = tmp_path / "derived"
    output = prepare(manifest_path, episode, output_dir)
    derived = json.loads(output.read_text())
    with np.load(source_cache) as orig, np.load(output_dir / "corrected_joints.npz") as copy:
        with np.load(output_dir / episode) as ep:
            np.testing.assert_array_equal(copy["joint_positions"], orig["joint_positions"][ep["parent_episode_frame_index"]])
            assert ep["synthetic_hold_frame"].sum() == 60
    assert derived["processing"]["close_duration_s"] == 2
    assert derived["processing"]["source_sha256"] == before[0]
    with pytest.raises(FileExistsError):
        prepare(manifest_path, episode, output_dir)
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]


@pytest.mark.parametrize("kwargs", [{"gripper_kp": 0}, {"gripper_kv": -1},
                                   {"gripper_max_torque": float("nan")},
                                   {"gripper_sliding_friction": -1},
                                   {"gripper_sliding_friction": float("nan")},
                                   {"gripper_control": "kinematic", "gripper_sliding_friction": 3}])
def test_bad_physics_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        load_physics_model(MODEL, **kwargs)


def test_kinematic_model_preserves_original_configuration() -> None:
    model = load_physics_model(MODEL, gripper_control="kinematic")
    original = mujoco.MjModel.from_xml_path(MODEL)
    assert model.nu == original.nu
    assert model.neq == original.neq
    assert model.opt.integrator == original.opt.integrator


@pytest.mark.parametrize("lower_m,friction,constrained", [(0.015, None, False), (0.0, 3.0, False), (0.0, 3.0, True)])
def test_episode_holds_dice_with_both_fingers_and_releases_into_box(
    tmp_path: Path, lower_m: float, friction: float | None, constrained: bool,
) -> None:
    from scripts.a2d_batch import fixed_upper_body_pose, load_corrected_trajectory
    from scripts.replay_a2d import bind_joints, build_dice_replay_plan, set_cardboard_box_pose
    from scripts.replay_a2d_physics import configure_dice_dynamics
    from scripts.search_a2d_physics_layout import candidate_metrics

    manifest_path = prepare(Path("datasets/replay_layouts.json"), "episode_000000.npz",
                            tmp_path / "hold41", hold_frame=41)
    manifest = json.loads(manifest_path.read_text())
    record = manifest["episodes"][0]
    model = load_physics_model(
        manifest["model"], gripper_sliding_friction=friction,
        arm_contact_mode="constrained" if constrained else "legacy",
        physics_timestep=0.0005 if constrained else None,
        contact_impratio=100 if constrained else 10,
    )
    configure_dice_dynamics(model, linear_damping=0.02, angular_damping=0.0005)
    gripper = ClosedLoopGripper(model, close_bias=0 if constrained else .15)
    torso = manifest["fixed_torso"]
    pose = fixed_upper_body_pose(torso["body_lift_m"], torso["body_pitch_rad"])
    trajectory = load_corrected_trajectory(
        manifest_path.parent / record["episode"], Path(manifest["summary"]),
        manifest_path.parent / record["cache"],
    )
    original_joints = trajectory.joint_positions.copy()
    bindings = bind_joints(model, trajectory.joint_names)
    lowered = lower_grasp_trajectory(model, trajectory, bindings, pose, lower_m)
    np.testing.assert_array_equal(trajectory.joint_positions, original_joints)
    if lower_m == 0:
        np.testing.assert_array_equal(lowered.joint_positions, original_joints)
    box = record["box"]
    set_cardboard_box_pose(model, x=box["x"], y=box["y"], yaw_deg=box["yaw_deg"])
    plan = build_dice_replay_plan(model, lowered, bindings, upper_body_pose=pose,
                                  dice_on_table=True, align_dice_to_gripper=True,
                                  dice_center_frame=41)
    metrics = candidate_metrics(
        model, lowered, bindings, pose, plan,
        np.array(record["dice"]["initial_position"]), record["dice"]["initial_yaw_deg"],
        settle_time_s=0.4, min_gripper_openness=0, gripper=gripper, post_rollout_s=1,
    )
    assert metrics["pick_and_place_success"]
    # Both historical configurations retain the die but visibly slide/rotate.
    # Do not certify them as a low-slip grasp merely because it lands in the box.
    if constrained:
        assert metrics["max_dice_translation_in_gripper_m"] < 0.002
        assert metrics["max_dice_rotation_in_gripper_deg"] < 1
        assert metrics["low_slip_pick_and_place_success"]
    else:
        assert metrics["max_dice_drop_relative_to_gripper_m"] > 0.02
        assert metrics["max_dice_rotation_in_gripper_deg"] > 10
        assert not metrics["low_slip_grasp_success"]
        assert not metrics["low_slip_pick_and_place_success"]
    assert metrics["retention_bilateral_contact_fraction"] == 1
    assert metrics["max_preopening_contact_loss_s"] == 0
    assert metrics["carry_bilateral_contact_fraction"] >= 0.95
    assert metrics["carry_support_contact_fraction"] == 0
    assert metrics["carry_height_max_m"] >= 0.93
    assert metrics["max_robot_support_penetration_m"] == 0
    assert metrics["max_loop_error_m"] < 0.0001
    assert metrics["max_drive_torque_nm"] <= 1.0 + 1e-12
    assert metrics["physics_warnings"] == 0


def test_prescribed_arm_constraints_leave_fingers_free_and_cancel_velocity_damping() -> None:
    from scripts.a2d_closed_loop import update_prescribed_arm_constraints
    from scripts.replay_a2d import A2D_ARM_JOINT_NAMES, A2D_UPPER_BODY_POSE

    model = load_physics_model(MODEL, arm_contact_mode='constrained', physics_timestep=.0005)
    original = load_physics_model(MODEL)
    np.testing.assert_array_equal(model.dof_armature, original.dof_armature)
    np.testing.assert_array_equal(model.geom_type, original.geom_type)
    np.testing.assert_array_equal(model.geom_size, original.geom_size)
    eq = np.array([i for i in range(model.neq) if model.equality(i).name.startswith('prescribed_')])
    names = {model.joint(int(j)).name for j in model.eq_obj1id[eq]}
    assert names == set(A2D_ARM_JOINT_NAMES) | {n for n, _ in A2D_UPPER_BODY_POSE}
    assert len(eq) == 18
    q = model.jnt_qposadr[model.eq_obj1id[eq]]
    v = model.jnt_dofadr[model.eq_obj1id[eq]]
    data = mujoco.MjData(model)
    data.qpos[q] = model.qpos0[q] + .01
    data.qvel[v] = .05
    before_q, before_v = data.qpos.copy(), data.qvel.copy()
    update_prescribed_arm_constraints(model, data)
    np.testing.assert_array_equal(data.qpos, before_q)
    np.testing.assert_array_equal(data.qvel, before_v)
    mujoco.mj_forward(model, data)
    rows = np.isin(data.efc_id, eq) & (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    assert rows.sum() == 18
    np.testing.assert_allclose(data.efc_aref[rows], 0, atol=1e-8)


def test_fast_release_only_changes_opening_targets_and_resets_per_hand() -> None:
    model = load_physics_model(MODEL)
    data = mujoco.MjData(model)
    gripper = ClosedLoopGripper(model, close_bias=0, release_mode='fast')
    gripper.command(data, np.ones(2), initialize=True)
    gripper.command(data, np.array([.35, .35]))
    np.testing.assert_allclose(data.ctrl[gripper.actuator_ids], .35 * np.pi / 4)
    q, v = data.qpos.copy(), data.qvel.copy()
    gripper.command(data, np.array([.35, .351]))
    np.testing.assert_allclose(data.ctrl[gripper.actuator_ids], np.array([.35, 1]) * np.pi / 4)
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, v)
    gripper.command(data, np.array([.35, .351]))  # Opening plateau stays open.
    assert data.ctrl[gripper.actuator_ids[1]] == np.pi / 4
    gripper.command(data, np.array([.35, .34]))  # New grasp can close again.
    np.testing.assert_allclose(data.ctrl[gripper.actuator_ids], np.array([.35, .34]) * np.pi / 4)
    gripper.command(data, np.array([.35, .35]), initialize=True)
    np.testing.assert_allclose(data.ctrl[gripper.actuator_ids], .35 * np.pi / 4)
    np.testing.assert_array_equal(model.actuator_forcerange[gripper.actuator_ids], [[-1, 1], [-1, 1]])
    with pytest.raises(ValueError): ClosedLoopGripper(model, release_mode='invalid')


def test_fast_release_shortens_loaded_contact_without_changing_carry(tmp_path: Path) -> None:
    from scripts.a2d_batch import fixed_upper_body_pose, load_corrected_trajectory
    from scripts.replay_a2d import bind_joints, build_dice_replay_plan, set_cardboard_box_pose
    from scripts.replay_a2d_physics import configure_dice_dynamics
    from scripts.search_a2d_physics_layout import candidate_metrics

    path = prepare(Path('datasets/replay_layouts.json'), 'episode_000006.npz', tmp_path / 'hold6')
    manifest = json.loads(path.read_text())
    record = manifest['episodes'][0]
    model = load_physics_model(manifest['model'], arm_contact_mode='constrained',
                               physics_timestep=.0005, contact_impratio=100, gripper_sliding_friction=3)
    configure_dice_dynamics(model, linear_damping=.02, angular_damping=.0005)
    torso = manifest['fixed_torso']
    pose = fixed_upper_body_pose(torso['body_lift_m'], torso['body_pitch_rad'])
    tr = load_corrected_trajectory(path.parent / record['episode'], Path(manifest['summary']),
                                   path.parent / record['cache'])
    bindings = bind_joints(model, tr.joint_names)
    box = record['box']
    set_cardboard_box_pose(model, x=box['x'], y=box['y'], yaw_deg=box['yaw_deg'])
    plan = build_dice_replay_plan(model, tr, bindings, upper_body_pose=pose,
                                  dice_on_table=True, align_dice_to_gripper=True,
                                  dice_center_frame=record['dice_center_frame'])
    results = []
    for mode in ['recorded', 'fast']:
        results.append(candidate_metrics(
            model, tr, bindings, pose, plan, np.array(record['dice']['initial_position']),
            record['dice']['initial_yaw_deg'], settle_time_s=.4, min_gripper_openness=0,
            gripper=ClosedLoopGripper(model, close_bias=0, release_mode=mode), post_rollout_s=1,
        ))
    before, after = results
    assert before['last_finger_contact_delay_from_opening_s'] > .1
    assert after['last_finger_contact_delay_from_opening_s'] < .02
    assert not before['landed_in_box'] and after['landed_in_box']
    assert after['retention_bilateral_contact_fraction'] == 1
    assert after['max_dice_translation_in_gripper_m'] == pytest.approx(before['max_dice_translation_in_gripper_m'])
    assert after['max_drive_torque_nm'] <= 1 + 1e-12
    assert after['physics_warnings'] == 0
