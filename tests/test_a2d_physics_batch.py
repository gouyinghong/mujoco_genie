from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import replay_a2d_physics_batch as batch
from scripts.replay_a2d_physics import LAYOUT_SCHEMA


def manifest_fixture(tmp_path: Path) -> Path:
    data = tmp_path / 'source'
    data.mkdir()
    doc = {'schema': LAYOUT_SCHEMA, 'dataset_dir': str(data),
           'model': str(tmp_path / 'model.xml'), 'summary': str(tmp_path / 'source_summary.json'),
           'episodes': [{'episode': f'episode_{i:06d}.npz', 'status': 'failed' if i == 3 else 'ok'} for i in range(4)]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(doc))
    return path


def fake_metrics(passed: bool) -> dict:
    return {'low_slip_pick_and_place_success': passed, 'pick_and_place_success': passed,
            'max_dice_translation_in_gripper_m': .001 if passed else .01,
            'max_dice_rotation_in_gripper_deg': 1.0}


def test_batch_isolates_errors_records_failures_and_resumes_without_replaying_completed(tmp_path, monkeypatch):
    manifest = manifest_fixture(tmp_path)
    output = tmp_path / 'batch'
    argv = ['--manifest', str(manifest), '--output-dir', str(output), '--headless', '--keep-timing']
    calls = []

    def child(command, **kwargs):
        episode = command[command.index('--episode') + 1]
        calls.append(episode)
        assert '--headless' in command
        assert command[command.index('--arm-contact-mode') + 1] == 'constrained'
        assert command[command.index('--gripper-close-bias') + 1] == '0'
        if episode == 'episode_000002.npz' and calls.count(episode) == 1:
            raise subprocess.TimeoutExpired(command, 1)
        metrics = Path(command[command.index('--metrics-output') + 1])
        metrics.write_text(json.dumps(fake_metrics(episode != 'episode_000001.npz')))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(batch.subprocess, 'run', child)
    assert batch.run_batch(batch.parse_args(argv)) == 1
    report = json.loads((output / 'summary.json').read_text())
    assert [r['status'] for r in report['episodes']] == ['passed', 'failed', 'error', 'skipped']
    assert (output / 'summary.csv').exists()
    assert batch.run_batch(batch.parse_args(argv + ['--resume'])) == 1
    assert calls == ['episode_000000.npz', 'episode_000001.npz', 'episode_000002.npz', 'episode_000002.npz']
    assert json.loads((output / 'summary.json').read_text())['counts']['passed'] == 2
    with pytest.raises(FileExistsError):
        batch.run_batch(batch.parse_args(argv))
    with pytest.raises(ValueError, match='parameters or input files changed'):
        batch.run_batch(batch.parse_args(argv + ['--resume', '--gripper-close-bias', '.15']))
    (tmp_path / 'source/episode_000000.npz').write_bytes(b'changed input')
    with pytest.raises(ValueError, match='parameters or input files changed'):
        batch.run_batch(batch.parse_args(argv + ['--resume']))


def test_closing_batch_viewer_stops_queue_and_preserves_pending_work(tmp_path, monkeypatch):
    manifest = manifest_fixture(tmp_path)
    output = tmp_path / 'batch'
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert '--start-immediately' in command and '--exit-when-finished' in command
        return subprocess.CompletedProcess(command, 130)

    monkeypatch.setattr(batch.subprocess, 'run', child)
    args = batch.parse_args(['--manifest', str(manifest), '--output-dir', str(output), '--keep-timing'])
    assert batch.run_batch(args) == 130
    report = json.loads((output / 'summary.json').read_text())
    assert len(calls) == 1
    assert report['episodes'][0]['status'] == 'interrupted'
    assert report['episodes'][1]['status'] == 'pending'


def test_selection_is_explicit_and_uses_manifest_order(tmp_path):
    manifest = json.loads(manifest_fixture(tmp_path).read_text())
    records = batch.select_records(manifest, ['episode_000002.npz', 'episode_000000.npz'], 0, None)
    assert [r['episode'] for r in records] == ['episode_000000.npz', 'episode_000002.npz']
    assert batch.select_records(manifest, None, 1, 1)[0]['episode'] == 'episode_000001.npz'
    with pytest.raises(ValueError, match='absent'):
        batch.select_records(manifest, ['episode_999999.npz'], 0, None)
    with pytest.raises(ValueError):
        batch.select_records(manifest, None, -1, None)
    with pytest.raises(SystemExit):
        batch.parse_args(['--resume'])
    with pytest.raises(SystemExit):
        batch.parse_args(['--close-duration-s', 'nan'])


@pytest.mark.parametrize('finish', [True, False])
def test_single_viewer_batch_exit_and_metrics(tmp_path, monkeypatch, finish):
    import numpy as np
    from types import SimpleNamespace
    from mujoco import viewer
    from scripts import replay_a2d_physics as replay

    class FakeViewer:
        cam = SimpleNamespace(lookat=np.zeros(3))
        opt = SimpleNamespace(geomgroup=np.zeros(6), flags=np.zeros(128))

        def __enter__(self): return self
        def __exit__(self, *args): pass
        def is_running(self): return finish
        def sync(self): pass

    monkeypatch.setattr(viewer, 'launch_passive', lambda *args, **kwargs: FakeViewer())
    clock = iter(range(0, 1000000, 10))
    monkeypatch.setattr(replay.time, 'monotonic', lambda: next(clock))
    # GUI lifecycle uses a real physics model; visual rendering itself is stubbed.
    metrics = tmp_path / 'metrics.json'
    monkeypatch.setattr(replay.sys, 'argv', [
        'replay_a2d_physics.py', '--manifest',
        'datasets/a2d_closed_loop_episode_000000_hold41_close2s/manifest.json',
        '--episode', 'episode_000000.npz', '--start-immediately', '--exit-when-finished',
        '--speed', '100', '--metrics-output', str(metrics),
        '--arm-contact-mode', 'constrained', '--physics-timestep', '.0005',
        '--contact-impratio', '100', '--gripper-close-bias', '0',
        '--gripper-sliding-friction', '3', '--dice-linear-damping', '.02',
        '--dice-angular-damping', '.0005',
    ])
    if finish:
        replay.main()
        assert json.loads(metrics.read_text())['low_slip_pick_and_place_success']
        before = metrics.read_bytes()
        with pytest.raises(FileExistsError): replay.main()
        assert metrics.read_bytes() == before
    else:
        with pytest.raises(SystemExit) as error: replay.main()
        assert error.value.code == 130
        assert not metrics.exists()
