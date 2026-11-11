"""Replay saved parameters and check every Stage1 -> Stage2 evidence contract."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def verify(data, output):
    source = Path(__file__).resolve().parents[1] / 'stage1_final/stage1_evidence_v2.py'
    spec = importlib.util.spec_from_file_location('evidence_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    for path in sorted(output.glob('machine-*.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        name = result['machine']
        train = np.loadtxt(data / 'train' / (name + '.txt'), delimiter=',')
        test = np.loadtxt(data / 'test' / (name + '.txt'), delimiter=',')
        assert result['config']['code_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert result['config']['train_sha256'] == hashlib.sha256(train.tobytes()).hexdigest()
        model = module.EvidenceModel.__new__(module.EvidenceModel)
        with np.load(output / 'models' / (name + '.npz')) as params:
            for key in ('median', 'scale', 'cal_dynamic', 'cal_state'):
                setattr(model, key, params[key].copy())
            model.denominator = {n: params['denominator_' + n].copy() for n in module.FEATURES}
        replay = model.score(test)
        flat = {k: v for k, v in replay.items() if k != 'components'}
        flat.update({'component_' + k: v for k, v in replay['components'].items()})
        with np.load(output / 'scores' / (name + '.npz')) as saved:
            assert set(saved.files) == set(flat)
            for key, values in flat.items():
                np.testing.assert_array_equal(saved[key], values, err_msg=name + ':' + key)
        cores = {}
        rows_checked = 0
        for tile in result['intervals']:
            a, b = tile['start'], tile['end']
            left, right = tile['channel_ranking_start'], tile['channel_ranking_end']
            assert b - a == 224 and a <= left < right <= b
            key = (tile['band'], tile['parent_start'], tile['parent_end'])
            cores.setdefault(key, []).append((left, right))
            chosen = [r['channel'] for r in tile['channels']]
            assert len(chosen) == len(set(chosen)) <= result['config']['max_channels']
            assert set(chosen).isdisjoint(tile['omitted_channels'])
            assert set(chosen) | set(tile['omitted_channels']) == set(range(result['n_channels']))
            for row in tile['root_cause_candidates']:
                rows_checked += 1
                assert left <= row['score_peak'] < right
                s, e = row['evidence_interval']
                fs, fe = row['focus_interval']
                assert a <= s < e <= b and a <= fs < fe <= b and fe - fs == 60
                assert row['evidence_score'] > 0
                assert row['score_peak'] == row[row['source'] + '_peak']
                if row['dominant_component'] == 'variance60':
                    assert [fs, fe] == [s, e]
                if 'dino_aux_peak' in row:
                    ds, de = row['dino_aux_scope']
                    assert s <= ds <= row['dino_aux_peak'] < de <= e
        for (_, parent_start, parent_end), bounds in cores.items():
            bounds.sort()
            assert bounds[0][0] == parent_start and bounds[-1][1] == parent_end
            assert all(x[1] == y[0] for x, y in zip(bounds, bounds[1:]))
        high = np.zeros(len(test), dtype=bool)
        review = high.copy()
        for event in result['detected_events']:
            high[event['start']:event['end']] = True
        for event in result['review_events']:
            review[event['start']:event['end']] = True
        assert not np.any(high & review)
        results.append(dict(machine=name, replay='exact', tiles=len(result['intervals']),
                            evidence_rows=rows_checked, contracts='passed'))
    if not results:
        raise ValueError('No machine JSON artifacts found')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    results = verify(args.data, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
