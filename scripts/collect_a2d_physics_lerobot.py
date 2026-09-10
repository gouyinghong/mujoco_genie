#!/usr/bin/env python3
"""Collect dynamic A2D replay into a NEW local LeRobot dataset (no uploads)."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_a2d_lerobot as legacy
from scripts.a2d_dice_orientation import dice_quaternion
from scripts.a2d_batch import fixed_upper_body_pose, load_corrected_trajectory
from scripts.a2d_closed_loop import ClosedLoopGripper, load_physics_model
from scripts.prepare_a2d_grasp_hold import prepare
from scripts.replay_a2d import bind_joints, build_dice_replay_plan, interpolate_joint_state
from scripts.replay_a2d import set_cardboard_box_pose, set_target_visibility, apply_texture_gamma
from scripts.replay_a2d_physics import configure_dice_dynamics, set_initial_dice_pose, set_robot_target
from scripts.search_a2d_physics_layout import candidate_metrics

EXCLUDED = tuple(f"episode_{i:06d}.npz" for i in (3, 13, 15, 23))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=legacy.DEFAULT_MANIFEST)
    p.add_argument('--output-dir', type=Path, default=Path('collected_datasets') / ('a2d_physics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--episodes', nargs='+')
    p.add_argument('--max-episodes', type=int)
    p.add_argument('--close-duration-s', type=float, default=2)
    p.add_argument('--gripper-release-mode', choices=('fast', 'recorded'), default='fast')
    p.add_argument('--fps', type=int, default=int(legacy.HEAD_CAMERA_FPS))
    p.add_argument('--post-rollout-s', type=float, default=1)
    p.add_argument('--repo-id', default='local/a2d_physics_pick_place')
    p.add_argument('--task', default=legacy.DEFAULT_TASK)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--no-distortion', action='store_true')
    p.add_argument('--image-writer-threads', type=int, default=4)
    args = p.parse_args(argv)
    if not np.isfinite(args.close_duration_s) or args.close_duration_s <= 0:
        p.error('close-duration-s must be finite and positive')
    if not np.isfinite(args.post_rollout_s) or args.post_rollout_s < 0:
        p.error('post-rollout-s must be finite and nonnegative')
    if not 0 < args.fps <= 2000 or args.image_writer_threads < 0:
        p.error('fps must be in [1, 2000]; image-writer-threads must be nonnegative')
    if args.max_episodes is not None and args.max_episodes <= 0:
        p.error('max-episodes must be positive')
    return args


def select_records(manifest, episodes=None, limit=None):
    available = {r['episode']: r for r in manifest['episodes']}
    if episodes:
        missing = set(episodes) - available.keys()
        if missing:
            raise ValueError(f'Unknown episodes: {sorted(missing)}')
    augmented = manifest.get('processing', {}).get('type') == 'object_centric_augmentation'
    records = [r for r in manifest['episodes'] if (r.get('source_episode', r['episode']) if augmented else r['episode']) not in EXCLUDED
               and r.get('status') == 'ok' and (not episodes or r['episode'] in episodes)]
    records = records[:limit]
    if not records:
        raise ValueError('No eligible episodes; 3, 13, 15, 23 are always excluded')
    return records


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def make_scene(manifest_path, release_mode):
    doc = legacy.load_manifest(manifest_path)
    r = doc['episodes'][0]
    model = load_physics_model(doc['model'], gripper_control='closed-loop',
                               gripper_kp=30, gripper_kv=.2, gripper_max_torque=1,
                               gripper_sliding_friction=3, arm_contact_mode='constrained',
                               physics_timestep=.0005, contact_impratio=100)
    configure_dice_dynamics(model, linear_damping=.02, angular_damping=.0005)
    torso = doc['fixed_torso']
    pose = fixed_upper_body_pose(torso['body_lift_m'], torso['body_pitch_rad'])
    tr = load_corrected_trajectory(Path(doc['dataset_dir']) / r['episode'],
                                   Path(doc['summary']), manifest_path.parent / r['cache'] if r.get('cache') else None)
    bindings = bind_joints(model, tr.joint_names)
    legacy.validate_joint_limits(model, tr, bindings)
    box = r['box']
    set_cardboard_box_pose(model, x=box['x'], y=box['y'], yaw_deg=box['yaw_deg'])
    plan = build_dice_replay_plan(model, tr, bindings, upper_body_pose=pose,
                                dice_on_table=True, align_dice_to_gripper=True,
                                dice_center_frame=r['dice_center_frame'],
                                dice_xy_offset_m=tuple(r.get('dice_xy_offset_m', (0, 0))))
    if plan is None:
        raise ValueError('Cannot build dice plan')
    return model, tr, bindings, pose, plan, r, ClosedLoopGripper(model, 0, release_mode)


def snapshots(model, tr, bindings, pose, record, gripper, fps, post_rollout_s):
    """Yield states at nearest fixed physics steps; never reposition the die after reset."""
    data = mujoco.MjData(model)
    visual = mujoco.MjData(model)
    def target(t, moving=True, initialize=False):
        set_robot_target(model, data, tr, bindings, t, pose, 0,
                         moving=moving, gripper=gripper, initialize_gripper=initialize)
    target(0, False, True)
    set_initial_dice_pose(model, data, np.array(record['dice']['initial_position']),
                          dice_quaternion(record['dice']))
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    for _ in range(round(.4 / dt)):
        target(0, False)
        mujoco.mj_step(model, data)
    step = 0
    addresses = [(model.joint(f'{side}_wide1_joint').qposadr[0],
                  model.joint(f'{side}_narrow1_joint').qposadr[0]) for side in ('left', 'right')]
    for requested in legacy.fixed_rate_sample_times(tr.duration_s + post_rollout_s, fps):
        wanted = int(round(requested / dt))
        while step < wanted:
            target(min(step * dt, tr.duration_s), step * dt < tr.duration_s)
            mujoco.mj_step(model, data)
            step += 1
        # Forward only a copy for rendering, preserving the live solver's warm start.
        mujoco.mj_copyData(visual, model, data)
        mujoco.mj_forward(model, visual)
        actual = np.array([(data.qpos[a] - data.qpos[b]) * .5 / (np.pi / 4) for a, b in addresses])
        state = legacy.compose_robot_state(data.qpos[bindings.qpos_addresses], np.clip(actual, 0, 1))
        commanded_arm, _ = interpolate_joint_state(tr, min(max(0, step - 1) * dt, tr.duration_s))
        command = legacy.compose_robot_state(commanded_arm, np.clip(data.ctrl[gripper.actuator_ids] / (np.pi / 4), 0, 1))
        yield visual, state, command, step * dt, actual


def main(argv=None):
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = legacy.load_manifest(manifest_path)
    augmented = manifest.get('processing', {}).get('type') == 'object_centric_augmentation'
    if augmented:
        args.close_duration_s = manifest['processing']['close_duration_s']
        if args.gripper_release_mode != manifest['processing']['gripper_release_mode']:
            raise ValueError('Augmented data must use its validated gripper release mode')
    if manifest.get('processing', {}).get('type') == 'stationary_gripper_closure':
        raise ValueError('Use the original multi-episode manifest; this script prepares closure copies itself')
    records = select_records(manifest, args.episodes, args.max_episodes)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Choose a NEW output directory: {output}')
    LeRobotDataset = legacy.import_lerobot_dataset()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'excluded': list(EXCLUDED), 'episodes': [],
              'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'physics': {'timestep_s': .0005, 'arm_contact_mode': 'constrained', 'impratio': 100,
                          'gripper_kp': 30, 'gripper_kv': .2, 'max_torque_nm': 1, 'friction': 3,
                          'close_bias': 0, 'grasp_lower_m': 0, 'settle_time_s': .4,
                          'dice_linear_damping': .02, 'dice_angular_damping': .0005},
              'action_semantics': 'Next sampled effective position target: arm radians + normalized gripper actuator targets; observation is measured state, not command. Timestamps quantized to nearest 0.5 ms.'}
    report_path = output / 'collection_report.json'
    write_json(report_path, report)
    dataset = None
    renderer = None
    try:
        for record in records:
            name = record['episode']
            print(f'[{len(report["episodes"]) + 1}/{len(records)}] {name}', flush=True)
            prepared = (Path(record['prepared_manifest']) if augmented else
                        prepare(manifest_path, name, output / 'prepared' / Path(name).stem, duration_s=args.close_duration_s))
            model, tr, bindings, pose, plan, r, gripper = make_scene(prepared, args.gripper_release_mode)
            metrics = candidate_metrics(model, tr, bindings, pose, plan,
                                        np.array(r['dice']['initial_position']), r['dice']['initial_yaw_deg'],
                                        settle_time_s=.4, min_gripper_openness=0, gripper=gripper,
                                        post_rollout_s=args.post_rollout_s, initial_quaternion=dice_quaternion(r['dice']))
            if dataset is None:
                joint_names = tr.joint_names
                dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps,
                            features=legacy.lerobot_features(joint_names), root=output / 'lerobot',
                            robot_type='a2d_omnipicker_mujoco_physics', use_videos=True,
                            image_writer_processes=0, image_writer_threads=args.image_writer_threads, vcodec='h264')
            elif tr.joint_names != joint_names:
                raise ValueError(f'Joint order changed in {name}')
            model.vis.global_.offwidth = legacy.HEAD_CAMERA_WIDTH
            model.vis.global_.offheight = legacy.HEAD_CAMERA_HEIGHT
            legacy.hide_closed_head_shell(model)
            appearance = record.get('augmentation', {}).get('appearance', {})
            apply_texture_gamma(model, 'cardboard_box_texture', appearance.get('box_texture_gamma', .65))
            if appearance:
                model.light_diffuse[:] *= appearance['light_scale']
                model.geom_rgba[model.geom('table_top').id, :3] = appearance['table_rgb']
            set_target_visibility(model, False)
            option = mujoco.MjvOption()
            option.geomgroup[0] = 0
            option.geomgroup[legacy.CAMERA_OCCLUDER_GROUP] = 0
            renderer = mujoco.Renderer(model, height=legacy.HEAD_CAMERA_HEIGHT, width=legacy.HEAD_CAMERA_WIDTH)
            maps = None if args.no_distortion else legacy.distortion_maps()
            previous = None
            states, actions, times, apertures, dice_poses = [], [], [], [], []
            dice_adr = model.joint('dice_free_joint').qposadr[0]
            for visual, state, command, time_s, actual in snapshots(model, tr, bindings, pose, r, gripper, args.fps, args.post_rollout_s):
                if previous is not None:
                    image, previous_state = previous
                    dataset.add_frame({legacy.IMAGE_FEATURE: image, legacy.STATE_FEATURE: previous_state,
                                       legacy.ACTION_FEATURE: command, 'task': args.task})
                    actions.append(command.copy())
                renderer.update_scene(visual, camera=legacy.HEAD_CAMERA_NAME, scene_option=option)
                image = renderer.render().copy()
                if maps is not None:
                    image = cv2.remap(image, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                image = legacy.crop_head_camera_roi(image)
                previous = image, state
                states.append(state.copy()); times.append(time_s); apertures.append(actual.copy())
                dice_poses.append(visual.qpos[dice_adr:dice_adr + 7].copy())
                if args.preview:
                    cv2.imshow('Physics collection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                        raise KeyboardInterrupt
            renderer.close(); renderer = None
            if not actions:
                raise ValueError('Episode needs at least two samples')
            dataset.save_episode(parallel_encoding=False)
            index = len(report['episodes'])
            np.savez_compressed(output / f'episode_{index:06d}_physics.npz',
                                sample_times_s=times, measured_state=states, action=actions,
                                raw_gripper_openness=apertures, dice_qpos=dice_poses)
            report['episodes'].append({'lerobot_episode_index': index, 'source_episode': name,
                                      'augmentation': record.get('augmentation'),
                                      'prepared_manifest': str(prepared), 'frames': len(actions),
                                      'validation_metrics': metrics})
            write_json(report_path, report)
            print(f'  saved {len(actions)} frames; landed={metrics["landed_in_box"]}', flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        report['error'] = str(exc)
        if dataset is not None and legacy.has_pending_frames(dataset):
            dataset.clear_episode_buffer()
        raise
    finally:
        try:
            if dataset is not None:
                dataset.finalize()
        except BaseException as exc:
            report['status'] = 'error'
            report['finalization_error'] = str(exc)
            raise
        finally:
            if renderer is not None:
                renderer.close()
            if args.preview:
                cv2.destroyAllWindows()
            write_json(report_path, report)
    print(f'Done: {len(report["episodes"])} episodes -> {output / "lerobot"}', flush=True)


if __name__ == '__main__':
    main()
