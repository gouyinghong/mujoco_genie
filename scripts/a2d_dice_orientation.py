"""Persist full die orientation while retaining legacy yaw-only records."""
import numpy as np
from scipy.spatial.transform import Rotation

# Local face normals, not pip numbers (which depend on the mesh texture).
FACE_AXES = ('+z', '-z', '+x', '-x', '+y', '-y')
FACE_EULER = ((0, 0, 0), (180, 0, 0), (0, -90, 0),
              (0, 90, 0), (90, 0, 0), (-90, 0, 0))


def dice_quaternion(dice):
    if 'initial_quaternion_wxyz' in dice:
        q = np.asarray(dice['initial_quaternion_wxyz'], dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
            raise ValueError('Invalid die quaternion')
        return q / np.linalg.norm(q)
    yaw = np.deg2rad(dice['initial_yaw_deg']) / 2
    return np.array([np.cos(yaw), 0., 0., np.sin(yaw)])


def face_offset(dice):
    """Die orientation relative to its horizontal heading, as a 4x4 transform."""
    q = dice_quaternion(dice)
    full = Rotation.from_quat(q[[1, 2, 3, 0]])
    heading = Rotation.from_euler('z', dice['initial_yaw_deg'], degrees=True)
    result = np.eye(4)
    result[:3, :3] = (heading.inv() * full).as_matrix()
    return result


def set_up_face(dice, face):
    offset = Rotation.from_euler('xyz', FACE_EULER[FACE_AXES.index(face)], degrees=True)
    full = Rotation.from_euler('z', dice['initial_yaw_deg'], degrees=True) * offset
    dice['initial_quaternion_wxyz'] = full.as_quat()[[3, 0, 1, 2]].tolist()
    dice['up_face_axis'] = face


def physics_dice_layout(dice):
    return {'dice_position': dice['initial_position'],
            'dice_yaw_deg': dice['initial_yaw_deg'],
            'dice_quaternion_wxyz': dice_quaternion(dice).tolist()}
