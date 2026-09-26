from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.a2d_batch import (
    DEFAULT_BODY_LIFT_M,
    DEFAULT_BODY_PITCH_RAD,
    DEFAULT_RETARGET_DATASET,
    GRIPPER_ADJUSTMENT_SCHEMA,
    LAYOUT_SCHEMA,
    choose_dice_center_frame,
    load_gripper_boundaries,
    load_layout_overrides,
    placement_lift_profile,
    prepare_dataset_layouts,
)
from scripts.convert_a2d_to_mjcf import DEFAULT_A2D_URDF, convert_a2d_urdf_to_mjcf


DATASET = DEFAULT_RETARGET_DATASET


def test_gripper_boundaries_are_loaded_from_new_episode_report(tmp_path: Path) -> None:
    episode = tmp_path / "episode_000000.npz"
    episode.touch()
    expected = {
        "close_start_frame": 10,
        "close_end_frame": 20,
        "open_start_frame": 30,
        "open_end_frame": 40,
    }
    report = {
        "frames": 50,
        "gripper_adjustment": {
            "schema": GRIPPER_ADJUSTMENT_SCHEMA,
            "boundaries": expected,
        },
    }
    episode.with_name("episode_000000_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    episode.with_name("episode_000000_gripper_adjustment.json").write_text(
        json.dumps({"boundaries": {}}), encoding="utf-8"
    )

    assert load_gripper_boundaries(episode) == expected


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
        (0.647699, 0.043614, 0.0),
        atol=3e-3,
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
        body_lift_m=0.27948,
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
        episode_names=("episode_000001.npz",),
    )

    assert [record["episode"] for record in merged["episodes"][:2]] == [
        "episode_000000.npz",
        "episode_000001.npz",
    ]
    assert merged["episodes"][0] == first_record
    assert first_cache.stat().st_mtime_ns == first_cache_mtime_ns

    assert merged["fixed_torso"] == initial["fixed_torso"]
    before_rejected_update = output_path.read_bytes()
    with pytest.raises(ValueError, match="different torso pose"):
        prepare_dataset_layouts(
            model_path, DATASET, output_path,
            body_lift_m=DEFAULT_BODY_LIFT_M,
            episode_names=("episode_000001.npz",),
        )
    assert output_path.read_bytes() == before_rejected_update
