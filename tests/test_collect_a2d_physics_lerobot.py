from pathlib import Path

import numpy as np
import pytest

from scripts.collect_a2d_physics_lerobot import (
    EXCLUDED, legacy, make_scene, parse_args, select_records, snapshots,
)
from scripts.prepare_a2d_grasp_hold import prepare


def test_exclusions_apply_even_to_explicit_selection():
    doc = {'episodes': [{'episode': f'episode_{i:06d}.npz', 'status': 'ok'} for i in range(31)]}
    assert len(select_records(doc)) == 27
    assert not set(EXCLUDED) & {r['episode'] for r in select_records(doc)}
    assert [r['episode'] for r in select_records(doc, ['episode_000003.npz', 'episode_000008.npz'])] == ['episode_000008.npz']
    with pytest.raises(ValueError, match='No eligible'):
        select_records(doc, list(EXCLUDED))
    with pytest.raises(ValueError, match='Unknown'):
        select_records(doc, ['missing.npz'])


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_reject_invalid_closure(value):
    with pytest.raises(SystemExit):
        parse_args(['--close-duration-s', value])


def test_dynamic_snapshots_have_actual_state_and_fast_targets(tmp_path):
    manifest = Path('datasets/replay_layouts.json')
    prepared = prepare(manifest, 'episode_000000.npz', tmp_path / 'prepared', duration_s=1.5)
    model, tr, bindings, pose, plan, record, gripper = make_scene(prepared, 'fast')
    collected = []
    dice_adr = model.joint('dice_free_joint').qposadr[0]
    for visual, state, command, time, raw in snapshots(model, tr, bindings, pose, record, gripper, 30, 1):
        collected.append((state.copy(), command.copy(), time, visual.qpos[dice_adr:dice_adr + 7].copy()))
    states = np.array([s[0] for s in collected])
    commands = np.array([s[1] for s in collected])
    times = np.array([s[2] for s in collected])
    dice = np.array([s[3] for s in collected])
    assert states.shape[1] == 16
    assert np.isfinite(states).all()
    assert np.max(np.abs(times - np.arange(len(times)) / 30)) <= .00025 + 1e-10
    # During grip the measured aperture is held open by the die while the target closes.
    assert np.max(np.abs(states[:, 15] - commands[:, 15])) > .05
    # Fast opening uses the effective full-open target, not the slowly rising source command.
    after_release = (times > plan.release_s) & (times < plan.release_s + .2)
    assert after_release.any()
    np.testing.assert_allclose(commands[after_release, 15], 1)
    assert np.ptp(dice[:, 2]) > .08
    # Repeating the same physics with another image sampling rate must not affect the die.
    other = []
    for visual, *_ in snapshots(model, tr, bindings, pose, record, gripper, 10, 1):
        other.append(visual.qpos[dice_adr:dice_adr + 7].copy())
    np.testing.assert_allclose(dice[::3], other, atol=1e-10)
