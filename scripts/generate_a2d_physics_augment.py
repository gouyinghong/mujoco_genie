#!/usr/bin/env python3
"""Generate physics-validated object-centric A2D demonstrations in a new directory."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.a2d_augmentation import (boundaries, source_poses, transferred_targets, solve_trajectory,
    grip_relation, perturb_record, quality, retime_trajectory)
from scripts.a2d_batch import robot_table_metrics, TABLE_X_BOUNDS_M, TABLE_Y_BOUNDS_M
from scripts.collect_a2d_physics_lerobot import (legacy, make_scene, select_records, write_json)
from scripts.prepare_a2d_grasp_hold import prepare
from scripts.replay_a2d import set_cardboard_box_pose
from scripts.replay_a2d_physics import PHYSICS_LAYOUT_SCHEMA
from scripts.search_a2d_physics_layout import candidate_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('datasets/replay_layouts.json'))
    p.add_argument('--output-dir', type=Path, default=Path('datasets') / ('a2d_augmented_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--episodes', nargs='+', help='Source episode filenames; excluded originals remain excluded')
    p.add_argument('--max-sources', type=int, default=5)
    p.add_argument('--attempts', type=int, default=200)
    p.add_argument('--target-successes', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--close-duration-s', type=float, default=2)
    p.add_argument('--dice-xy-range-m', type=float, default=.02)
    p.add_argument('--box-xy-range-m', type=float, default=.03)
    p.add_argument('--dice-yaw-range-deg', type=float, default=10)
    p.add_argument('--box-yaw-range-deg', type=float, default=5)
    p.add_argument('--max-slip-m', type=float, default=.01)
    p.add_argument('--max-rotation-deg', type=float, default=10)
    p.add_argument('--collect', action='store_true', help='Render accepted episodes to LeRobot after generation')
    p.add_argument('--visual-randomization', action='store_true', help='Store per-episode lighting/table-color/box-gamma variations for collection')
    args = p.parse_args(argv)
    for k in ('max_sources', 'attempts', 'target_successes'):
        if getattr(args, k) <= 0:
            p.error(f'{k} must be positive')
    for k in ('close_duration_s', 'dice_xy_range_m', 'box_xy_range_m', 'dice_yaw_range_deg',
              'box_yaw_range_deg', 'max_slip_m', 'max_rotation_deg'):
        v = getattr(args, k)
        if not np.isfinite(v) or v < 0 or (k in ('close_duration_s', 'max_slip_m', 'max_rotation_deg') and v == 0):
            p.error(f'Invalid {k}')
    return args


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(scene, tr=None, record=None):
    model, source_tr, bindings, pose, plan, src_record, gripper = scene
    r = src_record if record is None else record
    return candidate_metrics(model, source_tr if tr is None else tr, bindings, pose, plan,
        np.array(r['dice']['initial_position']), r['dice']['initial_yaw_deg'],
        settle_time_s=.4, min_gripper_openness=0, gripper=gripper, post_rollout_s=1)


def scene_group(record):
    # Partition absolute scene cells, not individual episodes/frames.
    d, b = record['dice'], record['box']
    values = [*d['initial_position'][:2], b['x'], b['y']]
    cells = [int(np.floor(v / .01)) for v in values]
    cells += [int(np.floor(d['initial_yaw_deg'] / 5)), int(np.floor(b['yaw_deg'] / 5))]
    key = ','.join(map(str, cells))
    split = 'test' if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0 else 'train'
    return key, split


def in_workspace(r):
    p, b = r['dice']['initial_position'], r['box']
    # Conservative bounding circle for box; die stays flat on table, yaw only.
    return (TABLE_X_BOUNDS_M[0] + .043 < p[0] < TABLE_X_BOUNDS_M[1] - .043
        and TABLE_Y_BOUNDS_M[0] + .043 < p[1] < TABLE_Y_BOUNDS_M[1] - .043
        and TABLE_X_BOUNDS_M[0] + .15 < b['x'] < TABLE_X_BOUNDS_M[1] - .15
        and TABLE_Y_BOUNDS_M[0] + .15 < b['y'] < TABLE_Y_BOUNDS_M[1] - .15)


def save_candidate(path, source_manifest, record, tr, metadata):
    """Save complete replayable candidate, including failures, without modifying sources."""
    path.mkdir(parents=True, exist_ok=False)
    doc = legacy.load_manifest(source_manifest)
    source_path = Path(doc['dataset_dir']) / doc['episodes'][0]['episode']
    with np.load(source_path, allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    arrays['action_joint_position'] = tr.joint_positions
    arrays['local_timestamps_ns'] = np.rint(tr.times_s * 1e9).astype(np.int64)
    arrays['target_eef_wxyz'] = tr.target_eef_wxyz
    arrays['achieved_eef_wxyz'] = tr.achieved_eef_wxyz
    name = record['episode']
    np.savez_compressed(path / name, **arrays)
    np.savez_compressed(path / 'corrected_joints.npz', joint_positions=tr.joint_positions)
    shutil.copy2(doc['summary'], path / 'retarget_summary.json')
    r = copy.deepcopy(record)
    r.update(cache='corrected_joints.npz', augmentation=metadata)
    r.pop('metrics', None)
    new = {k: copy.deepcopy(doc[k]) for k in ('schema', 'model', 'fixed_torso')}
    new.update(dataset_dir=str(path), summary=str(path / 'retarget_summary.json'),
               episodes=[r], counts={'total': 1, 'ok': 1, 'failed': 0},
               processing={'type': 'object_centric_augmentation', 'close_duration_s': metadata['close_duration_s'],
                           'gripper_release_mode': 'fast'}, physics_layout=str(path / 'physics_layout.json'))
    layout = {'schema': PHYSICS_LAYOUT_SCHEMA, 'datasets': {path.name: {name: {
        'dice_position': r['dice']['initial_position'], 'dice_yaw_deg': r['dice']['initial_yaw_deg']}}}}
    write_json(path / 'physics_layout.json', layout)
    write_json(path / 'manifest.json', new)
    return new


def export_manifest(out, source_doc, accepted, close_duration):
    # Export accepted records only. Failed candidates remain outside this dataset directory.
    data = out / 'data'
    data.mkdir(exist_ok=True)
    summary = data / 'retarget_summary.json'
    if not summary.exists():
        shutil.copy2(source_doc['summary'], summary)
    records = []
    layout = {}
    for item in accepted:
        p = Path(item['candidate_manifest'])
        doc = legacy.load_manifest(p)
        r = copy.deepcopy(doc['episodes'][0])
        name = r['episode']
        dest = data / name
        if not dest.exists():
            shutil.copy2(Path(doc['dataset_dir']) / name, dest)
        r['cache'] = str((p.parent / r['cache']).relative_to(out))
        r['prepared_manifest'] = str(p)
        r['augmentation']['split'] = item['split']
        records.append(r)
        layout[name] = {'dice_position': r['dice']['initial_position'], 'dice_yaw_deg': r['dice']['initial_yaw_deg']}
    new = {k: copy.deepcopy(source_doc[k]) for k in ('schema', 'model', 'fixed_torso')}
    new.update(dataset_dir=str(data), summary=str(summary), episodes=records,
               counts={'total': len(records), 'ok': len(records), 'failed': 0},
               physics_layout=str(out / 'physics_layout.json'),
               processing={'type': 'object_centric_augmentation', 'close_duration_s': close_duration,
                           'gripper_release_mode': 'fast'})
    write_json(out / 'physics_layout.json', {'schema': PHYSICS_LAYOUT_SCHEMA, 'datasets': {'data': layout}})
    write_json(out / 'manifest.json', new)
    for split in ('train', 'test'):
        part = copy.deepcopy(new)
        part['episodes'] = [r for r in records if r['augmentation']['split'] == split]
        n = len(part['episodes'])
        part['counts'] = {'total': n, 'ok': n, 'failed': 0}
        write_json(out / f'{split}_manifest.json', part)


def main(argv=None):
    args = parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    doc = legacy.load_manifest(args.manifest)
    if doc.get('processing'):
        raise ValueError('Use the original replay manifest as augmentation source')
    records = select_records(doc, args.episodes)
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_manifest_sha256': fingerprint(args.manifest), 'sources': [], 'candidates': [], 'accepted': [],
              'model_sha256': fingerprint(doc['model']), 'summary_sha256': fingerprint(doc['summary']),
              'quality_gate': {'max_slip_m': args.max_slip_m, 'max_rotation_deg': args.max_rotation_deg,
                               'max_contact_loss_s': .02, 'minimum_retention': .98},
              'split_method': 'absolute 1cm scene cells / 5deg yaw bins hashed 80:20; last source held out when >=2 sources'}
    sources = []
    try:
        for record in records:
            name = record['episode']
            print(f'Validating source {name}', flush=True)
            prepared = prepare(args.manifest, name, out / 'sources' / Path(name).stem, duration_s=args.close_duration_s)
            scene = make_scene(prepared, 'fast')
            if scene[4].side != 'right':
                raise ValueError('This generator currently supports right-hand pick/place only')
            metrics = evaluate(scene)
            entry = {'episode': name, 'metrics': metrics, 'eligible': bool(metrics['low_slip_pick_and_place_success']),
                     'source_cache_sha256': fingerprint(args.manifest.parent / record['cache']) if record.get('cache') else None,
                     'source_npz_sha256': fingerprint(Path(doc['dataset_dir']) / name)}
            report['sources'].append(entry)
            write_json(out / 'generation_report.json', report)
            if entry['eligible']:
                m, tr, b, pose, plan, r, grip = scene
                phases = boundaries(tr, r)
                poses = source_poses(m, tr, b, pose)
                rel = grip_relation(m, tr, b, pose, r, grip, phases['grip_anchor_s'])
                sources.append((prepared, scene, phases, poses, rel))
                if len(sources) >= args.max_sources:
                    break
        if not sources:
            raise ValueError('No strictly stable source demonstrations found')
        rng = np.random.default_rng(args.seed)
        for attempt in range(args.attempts):
            if len(report['accepted']) >= args.target_successes:
                break
            si = attempt % len(sources)
            prepared, scene, phases, poses, source_rel = sources[si]
            m, tr, bindings, pose, plan, original, grip = scene
            perturb = {'dice_xy': rng.uniform(-args.dice_xy_range_m, args.dice_xy_range_m, 2).tolist(),
                       'dice_yaw_deg': float(rng.uniform(-args.dice_yaw_range_deg, args.dice_yaw_range_deg)),
                       'box_xy': rng.uniform(-args.box_xy_range_m, args.box_xy_range_m, 2).tolist(),
                       'box_yaw_deg': float(rng.uniform(-args.box_yaw_range_deg, args.box_yaw_range_deg))}
            r, dd, bd = perturb_record(original, **perturb)
            # Source and scene groups are both disjoint between training and test.
            desired_split = 'test' if len(sources) > 1 and si == len(sources) - 1 else 'train'
            for _ in range(1000):
                group, split = scene_group(r)
                if len(sources) == 1 or split == desired_split:
                    break
                perturb['dice_xy'] = rng.uniform(-args.dice_xy_range_m, args.dice_xy_range_m, 2).tolist()
                perturb['box_xy'] = rng.uniform(-args.box_xy_range_m, args.box_xy_range_m, 2).tolist()
                r, dd, bd = perturb_record(original, **perturb)
            else:
                raise ValueError('Cannot sample requested split; increase spatial ranges')
            metadata = {'source_episode': original['episode'], 'source_manifest': str(prepared),
                        'close_duration_s': args.close_duration_s, 'attempt': attempt, 'seed': args.seed,
                        'perturbation': perturb, 'phases': phases, 'scene_group': group, 'split': split}
            if args.visual_randomization:
                metadata['appearance'] = {
                    'light_scale': float(rng.uniform(.8, 1.2)),
                    'table_rgb': rng.uniform(.65, .95, 3).tolist(),
                    'box_texture_gamma': float(rng.uniform(.5, .8)),
                }
            r['episode'] = f'episode_{attempt:06d}.npz'
            r['source_episode'] = original['episode']
            result = {'attempt': attempt, 'source_episode': original['episode'], 'split': split, 'scene_group': group,
                      'perturbation': perturb, 'accepted': False}
            report['candidates'].append(result)
            print(f'[{attempt + 1}/{args.attempts}] source={original["episode"]} split={split}', flush=True)
            try:
                if not in_workspace(r):
                    raise ValueError('Sample outside conservative tabletop bounds')
                b = r['box']
                set_cardboard_box_pose(m, x=b['x'], y=b['y'], yaw_deg=b['yaw_deg'])
                targets = transferred_targets(poses, tr.times_s, phases, dd, bd)
                corrected, ik = solve_trajectory(m, tr, bindings, pose, targets)
                corrected = retime_trajectory(corrected, r)
                actual_rel = grip_relation(m, corrected, bindings, pose, r, grip, boundaries(corrected, r)['grip_anchor_s'])
                correction = source_rel @ np.linalg.inv(actual_rel)
                if np.linalg.norm(correction[:3, 3]) > .025:
                    raise ValueError('Grip offset too large for safe placement correction')
                targets = transferred_targets(poses, tr.times_s, phases, dd, bd, correction)
                corrected, ik = solve_trajectory(m, tr, bindings, pose, targets)
                corrected = retime_trajectory(corrected, r)
                metadata['generated_phases'] = boundaries(corrected, r)
                metadata['added_movement_time_s'] = corrected.duration_s - tr.duration_s
                table = robot_table_metrics(m, corrected, bindings, pose)
                metadata.update(ik=ik, grip_correction=correction.tolist(), table_check=table)
                result['ik'] = ik
                candidate_path = out / 'candidates' / f'{attempt:06d}'
                save_candidate(candidate_path, prepared, r, corrected, metadata)
                result['candidate_manifest'] = str(candidate_path / 'manifest.json')
                if table['contact_frames']:
                    raise ValueError('Robot/table collision in IK path')
                # Validate the serialized candidate through the same loader used by collection.
                metrics = evaluate(make_scene(candidate_path / 'manifest.json', 'fast'))
                result['metrics'] = metrics
                result['rejection_reasons'] = quality(metrics, args.max_slip_m, args.max_rotation_deg)
                result['accepted'] = not result['rejection_reasons']
                if result['accepted']:
                    report['accepted'].append(copy.deepcopy(result))
                print(f'  accepted={result["accepted"]}; slip={metrics["max_dice_translation_in_gripper_m"]*1000:.2f}mm; landed={metrics["landed_in_box"]}', flush=True)
            except ValueError as exc:
                result['rejection_reasons'] = [str(exc)]
                print(f'  rejected: {exc}', flush=True)
            write_json(out / 'generation_report.json', report)
            export_manifest(out, doc, report['accepted'], args.close_duration_s)
        report['status'] = 'complete'
        report['accepted_count'] = len(report['accepted'])
        report['attempted_count'] = len(report['candidates'])
        report['target_reached'] = len(report['accepted']) >= args.target_successes
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        report['error'] = str(exc)
        raise
    finally:
        write_json(out / 'generation_report.json', report)
    print(f'Accepted {len(report["accepted"])}/{len(report["candidates"])} -> {out / "manifest.json"}', flush=True)
    if args.collect and report['accepted']:
        from scripts.collect_a2d_physics_lerobot import main as collect
        for split in ('train', 'test'):
            if any(r['split'] == split for r in report['accepted']):
                collect(['--manifest', str(out / f'{split}_manifest.json'),
                         '--output-dir', str(out / 'collection' / split)])


if __name__ == '__main__':
    main()
