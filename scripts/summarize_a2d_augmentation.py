#!/usr/bin/env python3
"""Summarize augmentation quality and audit train/test provenance separation."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def summarize(path):
    d = json.loads(path.read_text())
    rows = []
    for c in d['candidates']:
        m = c.get('metrics', {})
        rows.append({'attempt': c['attempt'], 'source_episode': c['source_episode'],
                     'split': c['split'], 'accepted': c['accepted'],
                     'dice_dx_m': c['perturbation']['dice_xy'][0], 'dice_dy_m': c['perturbation']['dice_xy'][1],
                     'dice_yaw_delta_deg': c['perturbation']['dice_yaw_deg'],
                     'dice_up_face': c['perturbation'].get('dice_up_face', 'unchanged'),
                     'box_dx_m': c['perturbation']['box_xy'][0], 'box_dy_m': c['perturbation']['box_xy'][1],
                     'box_yaw_delta_deg': c['perturbation']['box_yaw_deg'],
                     'slip_mm': m.get('max_dice_translation_in_gripper_m', np.nan) * 1000,
                     'rotation_deg': m.get('max_dice_rotation_in_gripper_deg'),
                     'landed_in_box': m.get('landed_in_box'),
                     'reasons': '; '.join(c.get('rejection_reasons', []))})
    accepted = [c for c in d['candidates'] if c['accepted']]
    assert len(accepted) == len(d['accepted'])
    groups = {s: {c['scene_group'] for c in accepted if c['split'] == s} for s in ('train', 'test')}
    sources = {s: {c['source_episode'] for c in accepted if c['split'] == s} for s in ('train', 'test')}
    assert not groups['train'] & groups['test'], 'Scene-group leakage'
    source_separated = not sources['train'] & sources['test']
    if sum(s['eligible'] for s in d['sources']) >= 2:
        assert source_separated, 'Source-demo leakage'
    excluded = {f'episode_{i:06d}.npz' for i in (3,13,15,23)}
    assert not excluded & (sources['train'] | sources['test']), 'Excluded source used'
    reasons = Counter()
    for c in d['candidates']:
        for reason in c.get('rejection_reasons', []):
            reasons[reason.split(':',1)[0]] += 1
    result = {'generation_status': d['status'], 'attempts': len(rows), 'accepted': len(accepted),
              'acceptance_rate': len(accepted)/len(rows) if rows else 0,
              'split_counts': dict(Counter(c['split'] for c in accepted)),
              'source_counts': dict(Counter(c['source_episode'] for c in accepted)),
              'attempted_face_counts': dict(Counter(c['perturbation'].get('dice_up_face', 'unchanged') for c in d['candidates'])),
              'accepted_face_counts': dict(Counter(c['perturbation'].get('dice_up_face', 'unchanged') for c in accepted)),
              'rejection_reasons': dict(reasons), 'scene_groups_disjoint': True,
              'source_groups_disjoint': source_separated, 'excluded_sources_absent': True,
              'accepted_perturbation_ranges': {}}
    for key in ('dice_dx_m','dice_dy_m','dice_yaw_delta_deg','box_dx_m','box_dy_m','box_yaw_delta_deg'):
        vals = [r[key] for r in rows if r['accepted']]
        if vals:
            result['accepted_perturbation_ranges'][key] = [min(vals),max(vals)]
    return rows, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--output-prefix', type=Path, help='NEW path prefix for CSV and JSON')
    args = p.parse_args()
    rows, result = summarize(args.report)
    if args.output_prefix:
        csv_path = Path(str(args.output_prefix) + '.csv')
        json_path = Path(str(args.output_prefix) + '.json')
        if csv_path.exists() or json_path.exists():
            raise FileExistsError('Summary outputs already exist; choose another prefix')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('x', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['attempt'])
            w.writeheader(); w.writerows(rows)
        with json_path.open('x') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
