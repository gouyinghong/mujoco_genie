import numpy as np
import pytest

from scripts.a2d_head_camera import HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH
from scripts.collect_a2d_lerobot import (
    ACTION_FEATURE,
    GRIPPER_NAMES,
    IMAGE_FEATURE,
    ROI_HEIGHT,
    ROI_WIDTH,
    ROI_X,
    ROI_Y,
    STATE_FEATURE,
    compose_robot_state,
    crop_head_camera_roi,
    fixed_rate_sample_times,
    has_pending_frames,
    lerobot_features,
)


def test_fixed_rate_sample_times_stay_inside_source_duration() -> None:
    times = fixed_rate_sample_times(0.105, 30)
    np.testing.assert_allclose(times, (0.0, 1 / 30, 2 / 30, 3 / 30))
    assert times[-1] <= 0.105
    np.testing.assert_allclose(np.diff(times), 1 / 30)


def test_compose_robot_state_contains_14_arm_and_two_gripper_values() -> None:
    arm = np.arange(14, dtype=np.float64)
    gripper = np.array((0.25, 0.75))
    state = compose_robot_state(arm, gripper)

    assert state.shape == (16,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[:14], arm)
    np.testing.assert_allclose(state[14:], gripper)


def test_compose_robot_state_rejects_invalid_gripper_value() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        compose_robot_state(np.zeros(14), np.array((0.0, 1.1)))


def test_crop_head_camera_roi_uses_requested_bounds() -> None:
    image = np.arange(
        HEAD_CAMERA_HEIGHT * HEAD_CAMERA_WIDTH * 3, dtype=np.uint32
    ).reshape(HEAD_CAMERA_HEIGHT, HEAD_CAMERA_WIDTH, 3)

    cropped = crop_head_camera_roi(image)

    assert cropped.shape == (ROI_HEIGHT, ROI_WIDTH, 3)
    assert cropped.flags.c_contiguous
    np.testing.assert_array_equal(cropped[0, 0], image[ROI_Y, ROI_X])
    np.testing.assert_array_equal(
        cropped[-1, -1], image[ROI_Y + ROI_HEIGHT - 1, ROI_X + ROI_WIDTH - 1]
    )


def test_crop_head_camera_roi_rejects_wrong_input_shape() -> None:
    with pytest.raises(ValueError, match="Expected full head-camera"):
        crop_head_camera_roi(np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8))


def test_lerobot_feature_schema() -> None:
    joint_names = tuple(f"joint_{index}" for index in range(14))
    features = lerobot_features(joint_names)

    assert features[IMAGE_FEATURE] == {
        "dtype": "video",
        "shape": (ROI_HEIGHT, ROI_WIDTH, 3),
        "names": ["height", "width", "channels"],
    }
    assert features[STATE_FEATURE]["shape"] == (16,)
    assert features[STATE_FEATURE]["names"] == [*joint_names, *GRIPPER_NAMES]
    assert features[ACTION_FEATURE] == features[STATE_FEATURE]


def test_pending_frame_compatibility_for_lerobot_04() -> None:
    class Dataset:
        episode_buffer = {"size": 3}

    assert has_pending_frames(Dataset())
