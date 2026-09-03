import mujoco
import numpy as np
import pytest

from scripts.eval_mujoco_gr00t_genie1 import (
    ACTION_DIM,
    clamp_policy_action,
    compose_policy_state,
    iter_action_chunk,
    make_policy_observation,
    normalize_action_array,
    policy_arm_positions,
    policy_gripper_positions,
)
from scripts.collect_a2d_lerobot import ROI_HEIGHT, ROI_WIDTH
from scripts.replay_a2d import bind_joints


def test_flat_action_chunk_preserves_dataset_order() -> None:
    values = np.arange(2 * ACTION_DIM, dtype=np.float32).reshape(1, 2, ACTION_DIM)
    result = list(iter_action_chunk({"action": values}))
    np.testing.assert_array_equal(result[0], values[0, 0])
    np.testing.assert_array_equal(result[1], values[0, 1])


def test_structured_action_chunk_uses_left_right_then_grippers() -> None:
    action = {
        "left_arm": np.full((1, 2, 7), 1, dtype=np.float32),
        "right_arm": np.full((1, 2, 7), 2, dtype=np.float32),
        "left_gripper": np.full((1, 2, 1), 3, dtype=np.float32),
        "right_gripper": np.full((1, 2, 1), 4, dtype=np.float32),
    }
    result = list(iter_action_chunk(action))
    np.testing.assert_array_equal(result[0], [*([1] * 7), 3, *([2] * 7), 4])


def test_policy_state_uses_genie1_interleaved_order() -> None:
    arms = np.arange(14, dtype=np.float32)
    grippers = np.array((0.25, 0.75), dtype=np.float32)
    state = compose_policy_state(arms, grippers)
    np.testing.assert_array_equal(
        state, [*range(7), 0.25, *range(7, 14), 0.75]
    )
    np.testing.assert_array_equal(policy_arm_positions(state), arms)
    np.testing.assert_array_equal(policy_gripper_positions(state), grippers)


def test_policy_observation_matches_training_modalities() -> None:
    image = np.zeros((ROI_HEIGHT, ROI_WIDTH, 3), dtype=np.uint8)
    state = np.arange(ACTION_DIM, dtype=np.float32)
    observation = make_policy_observation(image, state, "task")
    assert observation["video"]["ego_view"].shape == (1, 1, 480, 848, 3)
    np.testing.assert_array_equal(observation["state"]["left_arm"].reshape(-1), state[:7])
    assert list(observation["state"]) == [
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
    ]
    np.testing.assert_array_equal(observation["state"]["left_gripper"].reshape(-1), state[7:8])
    np.testing.assert_array_equal(observation["state"]["right_arm"].reshape(-1), state[8:15])
    np.testing.assert_array_equal(observation["state"]["right_gripper"].reshape(-1), state[15:16])
    assert observation["language"]["task_description"] == [["task"]]


def test_normalize_action_rejects_rank_four() -> None:
    with pytest.raises(ValueError, match="rank"):
        normalize_action_array(np.zeros((1, 1, 1, 16)))


def test_action_clamping_uses_arm_limits_and_normalized_grippers() -> None:
    model = mujoco.MjModel.from_xml_path("assets/A2D_Omnipicker/A2D_with_box.xml")
    names = tuple([f"idx{21+i}_arm_l_joint{i+1}" for i in range(7)] + [f"idx{61+i}_arm_r_joint{i+1}" for i in range(7)])
    bindings = bind_joints(model, names)
    action = np.zeros(ACTION_DIM)
    action[0] = 100.0
    action[7] = -0.5
    action[15] = 1.5
    result, clipped = clamp_policy_action(model, bindings, action)
    assert clipped
    assert result[0] == pytest.approx(model.jnt_range[bindings.joint_ids[0], 1])
    np.testing.assert_array_equal(result[[7, 15]], (0.0, 1.0))
