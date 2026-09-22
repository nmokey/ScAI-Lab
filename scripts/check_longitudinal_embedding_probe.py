"""Independently reproduce saved linear-probe scores and pair-count metrics.

NumPy is used only to read archives; prediction and metric arithmetic use the
standard library, without importing sklearn or the probe's scoring code.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'docs/audit_2026-09-14/stratified/embedding_probe'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = json.loads((ROOT / 'results.json').read_text())
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    split = json.loads(Path(protocol['split_path']).read_text())
    assert report['status'] == 'complete'
    assert digest(ROOT / 'PROTOCOL.json') == report['protocol_sha256']
    assert digest(protocol['records_path']) == protocol['records_sha256']
    assert digest(protocol['split_path']) == protocol['split_sha256']
    for key in ('baseline_files_sha256', 'forecast_manifests_sha256'):
        for path, expected in report[key].items():
            assert digest(path) == expected
    output = dict(status='complete', arms={}, max_score_error=0., verified_fits=0)
    for arm, result in report['arms'].items():
        total_credit = total_pairs = total_correct = tp = tn = 0
        held = []
        for fold, expected_fold in zip(result['folds'], split['folds']):
            path = ROOT / f'{arm}_{fold["fold"]}.npz'
            assert digest(path) == fold['artifact_sha256']
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k].tolist() for k in archive.files}
            assert data['subjects'] == expected_fold['train_subjects'] + expected_fold['test_subjects']
            for j, (mean, scale) in enumerate(zip(data['mean'], data['scale'])):
                values = [x[j] for x in data['X'][:24]]
                expected_mean = math.fsum(values) / 24
                expected_scale = math.sqrt(math.fsum((x - expected_mean)**2 for x in values) / 24) or 1.
                assert math.isclose(mean, expected_mean, rel_tol=0, abs_tol=1e-10)
                assert math.isclose(scale, expected_scale, rel_tol=1e-10, abs_tol=1e-10)
            for x, row, label in zip(data['X'], fold['rows'], data['y']):
                score = math.fsum((v - m) / s * c for v, m, s, c in zip(
                    x, data['mean'], data['scale'], data['coefficient'])) + data['intercept']
                error = abs(score - row['score']); assert error < 1e-10
                output['max_score_error'] = max(output['max_score_error'], error)
                assert label == row['label']
            rows = fold['rows'][24:]; held += [r['pid'] for r in rows]
            positive = [r['score'] for r in rows if r['label'] == 1]
            negative = [r['score'] for r in rows if r['label'] == 0]
            credit = sum((a > b) + .5 * (a == b) for a in positive for b in negative)
            pairs = len(positive) * len(negative)
            correct = sum((r['score'] > 0) == r['label'] for r in rows)
            assert credit / pairs == fold['test']['auc']
            assert correct == fold['test']['correct']
            total_credit += credit; total_pairs += pairs; total_correct += correct
            tp += sum(r['label'] == 1 and r['score'] > 0 for r in rows)
            tn += sum(r['label'] == 0 and r['score'] <= 0 for r in rows)
            output['verified_fits'] += 1
        assert sorted(held) == split['subjects'] and total_pairs == 62
        metrics = dict(auc=total_credit / total_pairs, accuracy=total_correct / 32,
                       correct=total_correct, balanced_accuracy=.5 * (tp / 14 + tn / 18))
        assert all(value == result['summary'][key] for key, value in metrics.items())
        output['arms'][arm] = metrics
    (ROOT / 'independent_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
