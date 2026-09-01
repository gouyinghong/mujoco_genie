from pathlib import Path

import mujoco
import pytest

from scripts.a2d_batch import (
    DEFAULT_BODY_LIFT_M,
    DEFAULT_BODY_PITCH_RAD,
    _right_gripper_center_at_frame,
    fixed_upper_body_pose,
    robot_table_metrics,
)
from scripts.optimize_a2d_fixed_torso import (
    load_episode_inputs,
    solve_body_lift_for_reference_height,
)
from scripts.replay_a2d import bind_joints, load_trajectory


DATASET = Path(
    "datasets/fixed_spine3_to_g1_0723_add_effector_gripper_6cm_return"
)
MODEL_PATH = Path("assets/A2D_Omnipicker/A2D_with_box.xml")


def test_body_lift_solver_preserves_reference_grasp_height() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    reference = load_episode_inputs(model, DATASET, max_episodes=1)[0]
    baseline_pose = fixed_upper_body_pose(
        DEFAULT_BODY_LIFT_M,
        DEFAULT_BODY_PITCH_RAD,
    )
    target_z = float(
        _right_gripper_center_at_frame(
            model,
            reference.trajectory,
            reference.bindings,
            baseline_pose,
            reference.center_frame,
        )[2]
    )

    lift, actual_z = solve_body_lift_for_reference_height(
        model,
        reference,
        DEFAULT_BODY_PITCH_RAD,
        target_z + 0.015,
        tolerance_m=2e-5,
    )

    assert lift == pytest.approx(0.27948, abs=2e-5)
    assert actual_z == pytest.approx(target_z + 0.015, abs=2e-5)


def test_table_metrics_detect_episode_15_penetration() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    trajectory = load_trajectory(
        DATASET / "episode_000015.npz",
        DATASET / "retarget_summary.json",
    )
    bindings = bind_joints(model, trajectory.joint_names)
    metrics = robot_table_metrics(
        model,
        trajectory,
        bindings,
        fixed_upper_body_pose(DEFAULT_BODY_LIFT_M, DEFAULT_BODY_PITCH_RAD),
    )

    assert metrics["contact_frames"] == 4
    assert metrics["first_contact_frame"] == 28
    assert metrics["deepest_contact_frame"] == 30
    assert metrics["deepest_body"] == "right_narrow4_Link"
    assert metrics["max_penetration_m"] == pytest.approx(0.0205755873, abs=1e-9)
