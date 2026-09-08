from __future__ import annotations

import json
from pathlib import Path

from scripts.replay_a2d_physics import (
    PHYSICS_LAYOUT_SCHEMA,
    configure_dice_dynamics,
    load_physics_layout,
)
from scripts.search_a2d_physics_layout import save_result


def test_physics_layout_is_stored_separately_by_dataset_and_episode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physics_replay_layouts.json"
    dataset = tmp_path / "dataset_a"
    record = {
        "status": "ok",
        "dice_position": [0.75, -0.1, 0.83],
        "dice_yaw_deg": -15.0,
    }

    save_result(path, dataset, "episode_000000.npz", record)

    assert load_physics_layout(path, dataset, "episode_000000.npz") == record
    assert load_physics_layout(path, dataset, "episode_000001.npz") is None
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == PHYSICS_LAYOUT_SCHEMA
    assert document["datasets"] == {
        "dataset_a": {"episode_000000.npz": record}
    }


def test_saving_one_physics_pose_preserves_existing_episodes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "physics_replay_layouts.json"
    dataset = tmp_path / "dataset_a"
    first = {
        "status": "best_effort",
        "dice_position": [0.7, 0.0, 0.83],
        "dice_yaw_deg": 0.0,
    }
    second = {
        "status": "ok",
        "dice_position": [0.8, 0.1, 0.83],
        "dice_yaw_deg": 10.0,
    }

    save_result(path, dataset, "episode_000000.npz", first)
    save_result(path, dataset, "episode_000001.npz", second)

    assert load_physics_layout(path, dataset, "episode_000000.npz") == first
    assert load_physics_layout(path, dataset, "episode_000001.npz") == second


def test_configure_dice_dynamics_overrides_mass_friction_and_damping() -> None:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(
        "assets/A2D_Omnipicker/A2D_with_box.xml"
    )
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dice")
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "dice_free_joint"
    )
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dice_collision"
    )
    original_mass = float(model.body_mass[body_id])
    original_inertia = model.body_inertia[body_id].copy()

    configure_dice_dynamics(
        model,
        mass_kg=0.05,
        sliding_friction=1.5,
        linear_damping=0.02,
        angular_damping=0.0005,
    )

    assert model.body_mass[body_id] == 0.05
    np.testing.assert_allclose(
        model.body_inertia[body_id], original_inertia * (0.05 / original_mass)
    )
    assert model.geom_friction[geom_id, 0] == 1.5
    dof_address = int(model.jnt_dofadr[joint_id])
    np.testing.assert_allclose(
        model.dof_damping[dof_address : dof_address + 6],
        (0.02, 0.02, 0.02, 0.0005, 0.0005, 0.0005),
    )


def test_low_slip_search_prefers_placement_then_worst_normalized_motion() -> None:
    from scripts.search_a2d_physics_layout import candidate_rank

    base = dict(success=True, score=100, low_slip_pick_and_place_success=False,
                pick_and_place_success=True, low_slip_translation_tolerance_m=.002,
                low_slip_rotation_tolerance_deg=5,
                max_dice_translation_in_gripper_m=.02,
                max_dice_rotation_in_gripper_deg=10)
    rotating = dict(base, max_dice_translation_in_gripper_m=.001,
                    max_dice_rotation_in_gripper_deg=90)
    abandoned = dict(base, pick_and_place_success=False,
                     max_dice_translation_in_gripper_m=0,
                     max_dice_rotation_in_gripper_deg=0)
    steady = dict(base, low_slip_pick_and_place_success=True,
                  max_dice_translation_in_gripper_m=.001,
                  max_dice_rotation_in_gripper_deg=2)
    assert candidate_rank(steady, 'low-slip') > candidate_rank(base, 'low-slip')
    assert candidate_rank(base, 'low-slip') > candidate_rank(rotating, 'low-slip')
    assert candidate_rank(base, 'low-slip') > candidate_rank(abandoned, 'low-slip')
    assert candidate_rank(base, 'retention') == (True, 100)
