"""Policy mapping and real physics integration (run with MUJOCO_GL=egl)."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import eval_a2d_physics_gr00t as ev
from scripts.collect_a2d_physics_lerobot import snapshots


def test_action_and_observation_mapping():
    row = np.arange(16, dtype=float)
    np.testing.assert_array_equal(ev.action_chunk({'action': row})[0], row)
    interleaved = row[[*range(7), 14, *range(7, 14), 15]]
    np.testing.assert_array_equal(ev.action_chunk({'action': interleaved}, 'interleaved')[0], row)
    structured = {'left_arm': row[:7], 'right_arm': row[7:14],
                  'left_gripper': row[14:15], 'right_gripper': row[15:]}
    np.testing.assert_array_equal(ev.action_chunk(structured)[0], row)
    row[14:] = [.2, .8]
    obs = ev.observation(np.zeros((480, 848, 3), dtype=np.uint8), row, 'test')
    np.testing.assert_array_equal(obs['state']['right_arm'].ravel(), row[7:14])
    np.testing.assert_allclose(obs['state']['left_gripper'].ravel(), [.2])
    for invalid in (np.full(16, np.nan), np.zeros((1, 0, 16)), np.zeros(15)):
        with pytest.raises(ValueError):
            ev.action_chunk({'action': invalid})


@pytest.fixture
def scene():
    if not ev.DEFAULT_DATASET.exists():
        pytest.skip('Local v2 scene assets are required')
    _, info, report = ev.load_scenes(ev.DEFAULT_DATASET)
    return report['episodes'][0], info, report


def test_initial_state_matches_collection_and_dynamic_clock(scene):
    entry, info, report = scene
    env = ev.PhysicsEnv(*scene)
    m, tr, bindings, pose, _, record, gripper = ev.make_scene(Path(entry['prepared_manifest']), 'fast')
    sample = next(snapshots(m, tr, bindings, pose, record, gripper, info['fps'], 0))
    np.testing.assert_allclose(env.state(), sample[1], atol=1e-7)
    original = env.target.copy()
    # A held policy must not advance the stored demonstration.
    for _ in range(30):
        env.advance(original, 2.)
    assert env.ticks == 2000
    np.testing.assert_array_equal(env.target, original)
    assert env.max_torque <= 1.000001
    assert not env.metrics()['placement_condition']
    assert env.metrics()['physics_warnings'] == 0
    # An actuator command must not teleport linked finger positions.
    before = env.data.qpos.copy()
    env._target(original[:14], np.zeros(14), np.zeros(2))
    for a, b in env.aperture_addresses:
        assert env.data.qpos[a] == before[a]
        assert env.data.qpos[b] == before[b]


def test_local_policy_rollout_artifacts(scene, tmp_path):
    class HoldPolicy:
        def reset(self, options):
            assert options['language'] == 'test'

        def get_action(self, obs):
            assert obs['video']['ego_view'].shape == (1, 1, 480, 848, 3)
            return {k: np.repeat(v, 2, axis=1) for k, v in obs['state'].items()}, {}

    args = ev.parse_args(['--headless', '--max-steps', '3', '--save-video'])
    args.language = 'test'
    result = ev.run_episode(args, *scene, tmp_path, HoldPolicy())
    assert result['status'] == 'timeout', result
    assert result['inference_calls'] == 2
    data = np.load(tmp_path/'rollout.npz')
    assert data['state'].shape == (3, 16)
    np.testing.assert_allclose(data['state_time_s'], [0, .0335, .0665])
    np.testing.assert_allclose(data['next_time_s'], [.0335, .0665, .1])
    np.testing.assert_array_equal(data['next_state'][:-1], data['state'][1:])
    assert (tmp_path/'rollout.mp4').stat().st_size > 0
    assert json.loads((tmp_path/'result.json').read_text())['success'] is False


def test_summary_success_rate_and_empty_dry_runs(tmp_path):
    path = tmp_path/'summary.json'
    summary = {'episodes': ([{'status': 'success', 'success': True}] * 25
                            + [{'status': 'timeout', 'success': False}] * 43)}
    ev.save_summary(path, summary)
    saved = json.loads(path.read_text())
    assert saved['evaluated_episodes'] == 68
    assert saved['successful_episodes'] == 25
    assert saved['unsuccessful_episodes'] == 43
    assert saved['success_rate'] == pytest.approx(25/68)
    assert saved['success_rate_percent'] == 36.76
    assert saved['status_counts'] == {'success': 25, 'timeout': 43}
    for episodes in ([], [{'status': 'dry_run', 'success': False}]):
        ev.save_summary(path, {'episodes': episodes})
        saved = json.loads(path.read_text())
        assert saved['evaluated_episodes'] == 0
        assert saved['success_rate'] is None
        assert saved['success_rate_percent'] is None
    ev.save_summary(path, {'episodes': [{'status': 'error', 'success': False}]})
    assert json.loads(path.read_text())['success_rate'] == 0
