from __future__ import annotations

import numpy as np

from scripts.replay_a2d_physics_smooth_release import insert_smooth_release


def test_smooth_release_replaces_recorded_ramp_and_freezes_other_state() -> None:
    times = np.arange(6, dtype=np.int64) * 100_000_000
    effectors = np.array(
        [[1.0, 0.2], [1.0, 0.2], [1.0, 0.2],
         [1.0, 0.5], [1.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    joints = np.arange(12, dtype=float).reshape(6, 2)
    arrays = {
        "local_timestamps_ns": times,
        "episode_frame_index": np.arange(6, dtype=np.int64),
        "action_effector": effectors.copy(),
        "hand_status": effectors.copy(),
        "action_joint_position": joints.copy(),
    }

    result, indices = insert_smooth_release(
        arrays,
        open_start=2,
        open_end=4,
        side=1,
        duration_s=0.4,
    )

    np.testing.assert_array_equal(indices, (0, 1, 2, 2, 2, 2, 2, 5))
    assert len(result["local_timestamps_ns"]) == 8
    assert (
        result["local_timestamps_ns"][6]
        - result["local_timestamps_ns"][2]
        == 400_000_000
    )
    np.testing.assert_allclose(
        result["action_effector"][2:7, 1],
        np.linspace(0.2, 1.0, 5),
    )
    np.testing.assert_allclose(
        result["action_joint_position"][2:7],
        np.repeat(joints[2][None, :], 5, axis=0),
    )
    np.testing.assert_array_equal(
        result["hand_status"], result["action_effector"]
    )
    np.testing.assert_array_equal(arrays["action_effector"], effectors)
    assert result["synthetic_release_frame"].sum() == 4
