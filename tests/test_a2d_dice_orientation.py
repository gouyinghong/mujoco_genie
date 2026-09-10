import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.a2d_dice_orientation import FACE_AXES, set_up_face, dice_quaternion, face_offset, physics_dice_layout
from scripts.a2d_augmentation import perturb_record


@pytest.mark.parametrize('face', FACE_AXES)
def test_face_up_and_heading_transfer(face):
    dice = {'initial_position': [.7, 0, .83], 'initial_yaw_deg': 23.}
    set_up_face(dice, face)
    q = dice_quaternion(dice)
    normal = np.zeros(3)
    normal['xyz'.index(face[1])] = 1 if face[0] == '+' else -1
    np.testing.assert_allclose(Rotation.from_quat(q[[1, 2, 3, 0]]).apply(normal), [0, 0, 1], atol=1e-14)
    record = {'dice': dice, 'box': {'x': .6, 'y': .2, 'yaw_deg': 0}}
    before = copy.deepcopy(record)
    changed, dd, _ = perturb_record(record, [.01, 0], 5, [0, 0], 0)
    assert record == before
    np.testing.assert_allclose(face_offset(changed['dice']), face_offset(dice), atol=1e-14)
    assert physics_dice_layout(dice)['dice_quaternion_wxyz'] == q.tolist()
    # Discrete face change affects die orientation, never the arm's spatial transfer.
    other, other_dd, _ = perturb_record(record, [.01, 0], 5, [0, 0], 0, dice_up_face=face)
    np.testing.assert_allclose(dd, other_dd)
    source_grip = np.eye(4)
    source_grip[:3, 3] = [0, .02, -.06]
    source_grip = source_grip @ face_offset(dice)
    ideal_actual = source_grip @ np.linalg.inv(face_offset(dice)) @ face_offset(other['dice'])
    correction = source_grip @ np.linalg.inv(face_offset(dice)) @ face_offset(other['dice']) @ np.linalg.inv(ideal_actual)
    np.testing.assert_allclose(correction, np.eye(4), atol=1e-14)


def test_legacy_yaw_and_invalid_quaternion():
    np.testing.assert_allclose(dice_quaternion({'initial_yaw_deg': 0}), [1, 0, 0, 0])
    with pytest.raises(ValueError):
        dice_quaternion({'initial_quaternion_wxyz': [0, 0, 0, 0]})
