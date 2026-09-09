import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.a2d_augmentation import (transform, blend, transferred_targets, perturb_record,
    boundaries, source_poses, solve_trajectory, quality)
from scripts.collect_a2d_physics_lerobot import legacy, make_scene, select_records
from scripts.generate_a2d_physics_augment import scene_group, parse_args, save_candidate, export_manifest
from scripts.prepare_a2d_grasp_hold import prepare


def test_object_centric_transform_and_stage_boundaries():
    times = np.array([0, 1, 2, 3, 4, 5, 6.])
    phases = dict(close_start_s=1., grip_anchor_s=2., transfer_end_s=3., retract_s=5., end_s=6.)
    original = np.repeat(transform([.7, 0, .9])[None], len(times), axis=0)
    dd = transform([.01, -.02, 0], .1)
    bd = transform([-.03, .02, 0], -.2)
    targets = transferred_targets(original, times, phases, dd, bd)
    np.testing.assert_allclose(targets[0], original[0])
    np.testing.assert_allclose(targets[1], dd @ original[1])
    np.testing.assert_allclose(targets[2], dd @ original[2])
    np.testing.assert_allclose(targets[3], bd @ original[3])
    np.testing.assert_allclose(targets[-1], original[-1], atol=1e-15)
    for x in targets:
        np.testing.assert_allclose(x[:3, :3].T @ x[:3, :3], np.eye(3), atol=1e-15)


def test_grip_correction_preserves_desired_die_pose():
    source_rel = transform([0, .02, -.06], .03)
    actual_rel = transform([.001, .018, -.065], .06)
    base, bd = transform([.7, .1, 1.], .2), transform([.02, -.01, 0], -.1)
    corrected_base = bd @ base @ source_rel @ np.linalg.inv(actual_rel)
    np.testing.assert_allclose(corrected_base @ actual_rel, bd @ base @ source_rel, atol=1e-15)


def test_generation_exclusion_uses_source_not_new_episode_number():
    doc = {'processing': {'type': 'object_centric_augmentation'}, 'episodes': [
        {'episode': 'episode_000003.npz', 'source_episode': 'episode_000000.npz', 'status': 'ok'},
        {'episode': 'episode_000000.npz', 'source_episode': 'episode_000003.npz', 'status': 'ok'}]}
    assert [r['episode'] for r in select_records(doc)] == ['episode_000003.npz']


def test_zero_transfer_roundtrip_and_replay_export(tmp_path):
    manifest = Path('datasets/replay_layouts.json').resolve()
    prepared = prepare(manifest, 'episode_000000.npz', tmp_path/'source')
    scene = make_scene(prepared, 'fast')
    model, tr, bindings, pose, plan, record, _ = scene
    poses = source_poses(model, tr, bindings, pose)
    targets = transferred_targets(poses, tr.times_s, boundaries(tr, record), np.eye(4), np.eye(4))
    corrected, metrics = solve_trajectory(model, tr, bindings, pose, targets)
    np.testing.assert_allclose(corrected.joint_positions, tr.joint_positions, atol=1e-12)
    assert metrics['max_ik_position_error_m'] < 1e-12
    r, dd, bd = perturb_record(record, [.01, -.01], 5, [-.01, .02], -3)
    assert record['dice']['initial_position'] != r['dice']['initial_position']
    assert np.linalg.det(dd[:3, :3]) == pytest.approx(1)
    r['episode'] = 'episode_000003.npz'
    r['source_episode'] = record['episode']
    meta = {'close_duration_s': 2, 'source_episode': record['episode'], 'split': 'train'}
    candidate = tmp_path/'candidate'
    save_candidate(candidate, prepared, r, corrected, meta)
    loaded = make_scene(candidate/'manifest.json', 'fast')
    np.testing.assert_allclose(loaded[1].joint_positions, corrected.joint_positions)
    assert loaded[5]['dice'] == r['dice']
    with pytest.raises(FileExistsError):
        save_candidate(candidate, prepared, r, corrected, meta)
    out = tmp_path
    accepted = [{'candidate_manifest': str(candidate/'manifest.json'), 'split': 'train'}]
    export_manifest(out, legacy.load_manifest(manifest), accepted, 2)
    doc = legacy.load_manifest(out/'manifest.json')
    assert len(select_records(doc)) == 1
    assert (Path(doc['dataset_dir'])/'episode_000003.npz').is_file()
    assert scene_group(r) == scene_group(copy.deepcopy(r))
    assert json.loads((out/'test_manifest.json').read_text())['counts']['total'] == 0


@pytest.mark.parametrize('flag,value', [('--attempts','0'),('--dice-xy-range-m','nan'),('--max-slip-m','0')])
def test_invalid_generation_parameters(flag, value):
    with pytest.raises(SystemExit):
        parse_args([flag,value])


def test_retiming_preserves_closure_and_limits_speed():
    from dataclasses import replace
    from scripts.a2d_augmentation import retime_trajectory
    from scripts.replay_a2d import Trajectory
    times = np.array([0., .1, .2, 1.2, 2.2, 2.3])
    q = np.zeros((6, 14)); q[:, 7] = [0, .5, .6, .6, .6, 1.1]
    tr = Trajectory(times, q, tuple(f'q{i}' for i in range(14)),
                    np.zeros((6,14)), np.zeros((6,14)), np.zeros((6,2)), np.arange(6), np.arange(6))
    r = {'boundaries': {'close_start_frame': 2, 'close_end_frame': 4}}
    new = retime_trajectory(tr, r)
    assert new.times_s[4] - new.times_s[2] == pytest.approx(2.)
    assert np.max(np.abs(np.diff(new.joint_positions,axis=0))/np.diff(new.times_s)[:,None]) <= 2 + 1e-12
    assert new.duration_s > tr.duration_s
    np.testing.assert_array_equal(new.joint_positions, tr.joint_positions)


def test_quality_rejects_accidental_landing_without_grasp():
    d = dict(landed_in_box=True, carry_height_max_m=1., carry_bilateral_contact_fraction=0.,
             retention_bilateral_contact_fraction=0., max_preopening_contact_loss_s=.8,
             max_dice_translation_in_gripper_m=.3, max_dice_rotation_in_gripper_deg=20.,
             carry_support_contact_fraction=0., max_robot_support_penetration_m=0.,
             physics_warnings=0, max_loop_error_m=0., max_dice_speed_m_s=1.)
    assert {'poor_carry_contact','poor_retention','contact_loss','slip'} <= set(quality(d))
