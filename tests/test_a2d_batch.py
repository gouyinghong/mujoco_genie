from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.a2d_batch import (
    DEFAULT_BODY_LIFT_M,
    DEFAULT_BODY_PITCH_RAD,
    LAYOUT_SCHEMA,
    choose_dice_center_frame,
    load_layout_overrides,
    placement_lift_profile,
    prepare_dataset_layouts,
)
from scripts.convert_a2d_to_mjcf import DEFAULT_A2D_URDF, convert_a2d_urdf_to_mjcf


DATASET = Path(
    "datasets/fixed_spine3_to_g1_0723_add_effector_gripper_6cm_return"
)


def test_layout_overrides_are_optional_and_loaded_by_episode(
    tmp_path: Path,
) -> None:
    assert load_layout_overrides(tmp_path) == {}
    override_path = tmp_path / "replay_layout_overrides.json"
    override_path.write_text(
        json.dumps(
            {"episode_000002.npz": {"dice_xy_offset_m": [0.0, 0.02]}}
        ),
        encoding="utf-8",
    )

    assert load_layout_overrides(tmp_path) == {
        "episode_000002.npz": {"dice_xy_offset_m": [0.0, 0.02]}
    }


def test_layout_overrides_can_be_shared_outside_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "datasets" / "example_dataset"
    dataset_dir.mkdir(parents=True)
    shared_path = dataset_dir.parent / "replay_layout_overrides.json"
    shared_path.write_text(
        json.dumps(
            {
                "example_dataset": {
                    "episode_000003.npz": {
                        "dice_xy_offset_m": [0.0, 0.015]
                    }
                },
                "different_dataset": {
                    "episode_000003.npz": {
                        "dice_xy_offset_m": [0.0, -0.02]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_layout_overrides(dataset_dir) == {
        "episode_000003.npz": {"dice_xy_offset_m": [0.0, 0.015]}
    }


def test_center_frame_uses_episode_relative_closing_phase() -> None:
    assert (
        choose_dice_center_frame(
            {
                "close_start_frame": 31,
                "close_end_frame": 41,
                "open_start_frame": 72,
                "open_end_frame": 82,
            },
            110,
        )
        == 37
    )
    assert (
        choose_dice_center_frame(
            {
                "close_start_frame": 11,
                "close_end_frame": 21,
                "open_start_frame": 50,
                "open_end_frame": 60,
            },
            90,
        )
        == 17
    )


def test_placement_lift_starts_after_grasp_and_returns_after_release() -> None:
    boundaries = {
        "close_start_frame": 31,
        "close_end_frame": 41,
        "open_start_frame": 72,
        "open_end_frame": 82,
    }
    profile = placement_lift_profile(110, boundaries, 0.03)

    np.testing.assert_array_equal(profile[:42], 0.0)
    assert profile[67] == 0.03
    assert profile[82] == 0.03
    assert profile[92] == 0.0
    np.testing.assert_array_equal(profile[93:], 0.0)


def test_reference_episode_batch_layout_matches_tuned_replay(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "with_box.xml"
    convert_a2d_urdf_to_mjcf(
        DEFAULT_A2D_URDF,
        model_path,
        include_cardboard_box=True,
    )
    output_path = tmp_path / "replay_layouts.json"
    document = prepare_dataset_layouts(
        model_path,
        DATASET,
        output_path,
        body_lift_m=DEFAULT_BODY_LIFT_M,
        body_pitch_rad=DEFAULT_BODY_PITCH_RAD,
        max_episodes=1,
    )

    assert document["schema"] == LAYOUT_SCHEMA
    assert document["counts"] == {"total": 1, "ok": 1, "failed": 0}
    record = document["episodes"][0]
    assert record["episode"] == "episode_000000.npz"
    assert record["dice_center_frame"] == 37
    assert record["placement_lift_m"] == 0.0
    assert record["metrics"]["box_wall_contacts"] == 0
    np.testing.assert_allclose(
        (record["box"]["x"], record["box"]["y"], record["box"]["yaw_deg"]),
        (0.6461452726, 0.0411534499, 0.0),
        atol=1e-9,
    )
    cache_path = tmp_path / record["cache"]
    assert cache_path.is_file()
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache["joint_positions"].shape == (110, 14)
    with output_path.open("r", encoding="utf-8") as stream:
        assert json.load(stream)["counts"]["ok"] == 1


def test_incremental_preparation_merges_only_requested_episode(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "with_box.xml"
    convert_a2d_urdf_to_mjcf(
        DEFAULT_A2D_URDF,
        model_path,
        include_cardboard_box=True,
    )
    output_path = tmp_path / "replay_layouts.json"
    initial = prepare_dataset_layouts(
        model_path,
        DATASET,
        output_path,
        body_lift_m=DEFAULT_BODY_LIFT_M,
        body_pitch_rad=DEFAULT_BODY_PITCH_RAD,
        max_episodes=2,
    )
    first_record = initial["episodes"][0]
    first_cache = tmp_path / first_record["cache"]
    first_cache_mtime_ns = first_cache.stat().st_mtime_ns

    merged = prepare_dataset_layouts(
        model_path,
        DATASET,
        output_path,
        body_lift_m=DEFAULT_BODY_LIFT_M,
        body_pitch_rad=DEFAULT_BODY_PITCH_RAD,
        episode_names=("episode_000001.npz",),
    )

    assert [record["episode"] for record in merged["episodes"][:2]] == [
        "episode_000000.npz",
        "episode_000001.npz",
    ]
    assert merged["episodes"][0] == first_record
    assert first_cache.stat().st_mtime_ns == first_cache_mtime_ns
