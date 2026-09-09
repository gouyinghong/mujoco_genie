"""Object-centric pose transfer and continuous IK for A2D physics augmentation.

Uses the MimicGen method conceptually; no dependency on robosuite/MimicGen.
All transforms are world-frame homogeneous poses, rotations in radians.
"""
from dataclasses import replace
import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.replay_a2d import (apply_kinematic_pose, EEF_BODY_NAMES,
                                _pose_in_parent_frame, validate_joint_limits)
from scripts.collect_a2d_physics_lerobot import snapshots


def transform(position, yaw=0.):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    result[:3, 3] = position
    return result


def body_pose(data, body_id):
    result = np.eye(4)
    result[:3, :3] = data.xmat[body_id].reshape(3, 3)
    result[:3, 3] = data.xpos[body_id]
    return result


def blend(a, b, weight):
    result = np.eye(4)
    result[:3, 3] = (1 - weight) * a[:3, 3] + weight * b[:3, 3]
    delta = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
    result[:3, :3] = Rotation.from_rotvec(weight * delta).as_matrix() @ a[:3, :3]
    return result


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * x * (10 + x * (-15 + 6 * x))


def boundaries(tr, record):
    b = record['boundaries']
    close_start = float(tr.times_s[b['close_start_frame']])
    grasp = float(tr.times_s[b['close_end_frame']])
    # Openness begins rising between open_start-1 and open_start, as in validation.
    opening = float(tr.times_s[max(0, b['open_start_frame'] - 1)])
    anchor = grasp + .15
    transit_end = opening - .20
    if close_start <= 0 or transit_end <= anchor + .1:
        raise ValueError('Insufficient approach/carry duration for smooth object-centric transfer')
    return {'close_start_s': close_start, 'grasp_s': grasp,
            'grip_anchor_s': anchor, 'transfer_end_s': transit_end, 'opening_s': opening,
            'retract_s': min(float(tr.times_s[b['open_end_frame']]) + .2, tr.duration_s),
            'end_s': tr.duration_s}


def source_poses(model, tr, bindings, pose):
    data = mujoco.MjData(model)
    result = []
    for t in tr.times_s:
        apply_kinematic_pose(model, data, tr, bindings, float(t), show_target=False, upper_body_pose=pose)
        result.append(body_pose(data, model.body('right_base_link').id))
    return np.array(result)


def transferred_targets(poses, times, phases, dice_delta, box_delta, grip_correction=None):
    """Approach relative to die; carry transitions to box; release/retract relative to box."""
    correction = np.eye(4) if grip_correction is None else grip_correction
    out = []
    for src, t in zip(poses, times, strict=True):
        approach = smoothstep(t / phases['close_start_s'])
        carry = smoothstep((t - phases['grip_anchor_s']) /
                           (phases['transfer_end_s'] - phases['grip_anchor_s']))
        grasp_target = blend(src, dice_delta @ src, approach)
        place_target = box_delta @ src @ correction
        target = blend(grasp_target, place_target, carry)
        # Once released and clear of the box, smoothly return to the original rest pose.
        retract = smoothstep((t - phases['retract_s']) / max(.001, phases['end_s'] - phases['retract_s']))
        out.append(blend(target, src, retract))
    return np.array(out)


def solve_trajectory(model, tr, bindings, pose, targets):
    """6D DLS IK, warm-started by prior correction to retain the source elbow branch."""
    data = mujoco.MjData(model)
    body_id = model.body('right_base_link').id
    ids = bindings.joint_ids[7:]
    qadr, vadr = bindings.qpos_addresses[7:], bindings.dof_addresses[7:]
    lo, hi = model.jnt_range[ids, 0], model.jnt_range[ids, 1]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    qs = tr.joint_positions.copy()
    previous = np.zeros(7)
    errors = []
    eef = []
    for i, t in enumerate(tr.times_s):
        apply_kinematic_pose(model, data, tr, bindings, float(t), show_target=False, upper_body_pose=pose)
        original = tr.joint_positions[i, 7:]
        q = np.clip(original + previous, lo, hi)
        target = targets[i]
        for _ in range(100):
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            actual = body_pose(data, body_id)
            ep = target[:3, 3] - actual[:3, 3]
            er = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            if np.linalg.norm(ep) < .0002 and np.linalg.norm(er) < .002:
                break
            mujoco.mj_jacBody(model, data, jp, jr, body_id)
            jac = np.vstack([jp[:, vadr], .15 * jr[:, vadr]])
            error = np.r_[ep, .15 * er]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), error)
            dq *= min(1., .08 / max(np.linalg.norm(dq), 1e-12))
            q = np.clip(q + dq, lo, hi)
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        actual = body_pose(data, body_id)
        errors.append([np.linalg.norm(target[:3, 3] - actual[:3, 3]),
                       np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec())])
        qs[i, 7:] = q
        previous = q - original
        eef.append(np.concatenate([_pose_in_parent_frame(model, data, name) for name in EEF_BODY_NAMES]))
    errors = np.array(errors)
    velocity = np.diff(qs, axis=0) / np.diff(tr.times_s)[:, None]
    source_velocity = np.diff(tr.joint_positions, axis=0) / np.diff(tr.times_s)[:, None]
    metrics = {'max_ik_position_error_m': float(errors[:, 0].max()),
               'max_ik_rotation_error_deg': float(np.rad2deg(errors[:, 1].max())),
               'max_joint_step_rad': float(np.abs(np.diff(qs, axis=0)).max()),
               'max_joint_speed_rad_s': float(np.abs(velocity).max())}
    if (metrics['max_ik_position_error_m'] > .001 or metrics['max_ik_rotation_error_deg'] > 1
        or metrics['max_joint_step_rad'] > max(.15, 1.5 * np.abs(np.diff(tr.joint_positions, axis=0)).max())):
        raise ValueError(f'IK/continuity gate failed: {metrics}')
    corrected = replace(tr, joint_positions=qs, target_eef_wxyz=np.array(eef), achieved_eef_wxyz=np.array(eef))
    validate_joint_limits(model, corrected, bindings)
    return corrected, metrics


def retime_trajectory(tr, record, max_speed=2.):
    """Slow movement where needed; preserve stationary gripper closure duration exactly."""
    dt = np.diff(tr.times_s)
    required = np.max(np.abs(np.diff(tr.joint_positions, axis=0)), axis=1) / max_speed
    durations = np.maximum(dt, required)
    b = record['boundaries']
    closing = slice(b['close_start_frame'], b['close_end_frame'])
    if np.any(required[closing] > dt[closing] + 1e-8):
        raise ValueError('Arm moves too quickly during stationary closing phase')
    durations[closing] = dt[closing]
    times = np.r_[tr.times_s[0], tr.times_s[0] + np.cumsum(durations)]
    return replace(tr, times_s=times)


def grip_relation(model, tr, bindings, pose, record, gripper, anchor):
    """Measure real die-in-hand transform after closing; do not attach/reset the die."""
    base_id, dice_id = model.body('right_base_link').id, model.body('dice').id
    for data, _, _, t, _ in snapshots(model, tr, bindings, pose, record, gripper, 100, 0):
        if t + 1e-9 >= anchor:
            return np.linalg.inv(body_pose(data, base_id)) @ body_pose(data, dice_id)
    raise ValueError('Grip anchor beyond trajectory')


def perturb_record(record, dice_xy, dice_yaw_deg, box_xy, box_yaw_deg):
    r = copy.deepcopy(record)
    old_dice = transform(record['dice']['initial_position'], np.deg2rad(record['dice']['initial_yaw_deg']))
    new_dice = transform(old_dice[:3, 3] + np.r_[dice_xy, 0],
                         np.deg2rad(record['dice']['initial_yaw_deg'] + dice_yaw_deg))
    b = record['box']
    old_box = transform([b['x'], b['y'], 0], np.deg2rad(b['yaw_deg']))
    new_box = transform(old_box[:3, 3] + np.r_[box_xy, 0], np.deg2rad(b['yaw_deg'] + box_yaw_deg))
    r['dice']['initial_position'] = new_dice[:3, 3].tolist()
    r['dice']['initial_yaw_deg'] += float(dice_yaw_deg)
    r['box'].update(x=float(new_box[0, 3]), y=float(new_box[1, 3]), yaw_deg=b['yaw_deg'] + float(box_yaw_deg))
    return r, new_dice @ np.linalg.inv(old_dice), new_box @ np.linalg.inv(old_box)


def quality(metrics, max_slip_m=.01, max_rotation_deg=10.):
    reasons = []
    # Permit brief unilateral contact gaps but require actual bilateral carrying.
    gates = {'not_landed': metrics['landed_in_box'],
             'insufficient_lift': metrics['carry_height_max_m'] >= .91,
             'poor_carry_contact': metrics['carry_bilateral_contact_fraction'] >= .95,
             'poor_retention': metrics['retention_bilateral_contact_fraction'] >= .98,
             'contact_loss': metrics['max_preopening_contact_loss_s'] <= .02,
             'slip': metrics['max_dice_translation_in_gripper_m'] <= max_slip_m,
             'rotation': metrics['max_dice_rotation_in_gripper_deg'] <= max_rotation_deg,
             'supported_carry': metrics['carry_support_contact_fraction'] <= .05,
             'collision': metrics['max_robot_support_penetration_m'] <= .001,
             'solver_warning': metrics['physics_warnings'] == 0,
             'linkage_error': metrics['max_loop_error_m'] < .001,
             'excessive_speed': metrics['max_dice_speed_m_s'] <= 2.}
    for reason, passed in gates.items():
        if not passed:
            reasons.append(reason)
    return reasons
