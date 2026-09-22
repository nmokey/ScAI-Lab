"""Independent saved-feature, linear-score, and forecast-metric verification.

No experiment scorer, sklearn, or training module is imported. NumPy reads
arrays; scalar predictions, class pairs, cosine, and aggregation use Python.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'docs/audit_2026-09-14/residual_rollout_experiment'
OLD = Path('/data1/Processed_NIfTI_Test/embeddings/longitudinal_stratified_20260914')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, rel_tol=0, abs_tol=tolerance), (a, b)


def mean(values):
    values = list(values)
    assert values
    return math.fsum(values) / len(values)


def cosine(a, b):
    numerator = math.fsum(float(x) * float(y) for x, y in zip(a, b))
    return numerator / math.sqrt(math.fsum(float(x)**2 for x in a) * math.fsum(float(y)**2 for y in b))


def main():
    protocol = json.loads((ROOT / 'PROTOCOL.json').read_text())
    report = json.loads((ROOT / 'results.json').read_text())
    assert report['status'] == 'complete' and report['protocol_sha256'] == object_digest(protocol)
    for name, expected in protocol['source_sha256'].items():
        assert digest(REPO / 'scripts' / name) == digest(ROOT / 'source_snapshot' / name) == expected
    assert digest(protocol['embeddings_path']) == protocol['embeddings_sha256']
    assert digest(protocol['records_path']) == protocol['records_sha256']
    records = json.loads(Path(protocol['records_path']).read_text())
    labels = {}
    for row in records:
        value = row['answer_vqa_numeric']['genotype']
        assert row['pid'] not in labels or labels[row['pid']] == value
        labels[row['pid']] = value
    split = protocol['population']; new = Path(protocol['forecast_output'])
    archive = np.load(protocol['embeddings_path'], allow_pickle=True)
    lookup = {(str(s), str(w)): e for s, w, e in zip(archive['subject_ids'], archive['weeks'], archive['embeddings'])}
    output = dict(status='complete', verified_classifiers=0, verified_forecast_tokens=0,
                  max_score_error=0., genotype={}, forecast_subject_mean={})
    for fold in split['folds']:
        key = fold['key']
        assert digest(OLD / key / 'manifest.json') == protocol['original_manifests_sha256'][key]
        assert digest(new / key / 'manifest.json') == report['manifests_sha256'][key]
        manifest = json.loads((new / key / 'manifest.json').read_text())
        assert manifest['experiment_protocol_sha256'] == object_digest(protocol)
        assert manifest['fold'] == fold
        for name, expected in manifest['files'].items():
            assert digest(new / key / name) == expected
            value = np.load(new / key / name, allow_pickle=False)
            sid = name.rsplit('_ts', 1)[0]; fit = manifest['fits'][manifest['queries'][sid]]
            assert sid not in fit['fit_subjects']
            assert set(fit['fit_subjects']) <= set(fold['train_subjects'])
            assert not fit['genotype_conditioning'] and not fit['genotype_supervision']
            assert value.shape == (768,) and np.isfinite(value).all()
            close(math.sqrt(math.fsum(float(v)**2 for v in value)), fit['norm_reference'], tolerance=1e-5)
            output['verified_forecast_tokens'] += 1
        for fit_key, fit in manifest['fits'].items():
            assert fit['reload_exact'] and fit['updates'] == protocol['settings']['epochs']
            assert digest(new / key / f'{fit_key}.pt') == fit['checkpoint_sha256']
    assert output['verified_forecast_tokens'] == 384
    for arm, result in report['arms'].items():
        credit = pairs = correct = tp = tn = 0
        held = []
        kind, slots = protocol['classifier_arms'][arm]
        for entry, fold in zip(result['folds'], split['folds']):
            path = ROOT / f'{arm}_{fold["key"]}.npz'
            assert digest(path) == entry['artifact_sha256']
            with np.load(path, allow_pickle=False) as saved:
                data = {k: saved[k].tolist() for k in saved.files}
            ids = fold['train_subjects'] + fold['test_subjects']
            assert data['subjects'] == ids and data['y'] == [labels[s] for s in ids]
            for sid, actual_features in zip(ids, data['X']):
                tokens = [lookup[(sid, 'Week 12')]] + [np.load((OLD if kind == 'old' else new) /
                    fold['key'] / f'{sid}_ts{k}.npy', allow_pickle=False) for k in (1, 2, 3)]
                expected = [float(v) for slot in slots for v in tokens[slot]]
                assert expected == actual_features
            for j, (center, scale) in enumerate(zip(data['mean'], data['scale'])):
                values = [x[j] for x in data['X'][:24]]
                expected_center = mean(values)
                expected_scale = math.sqrt(mean((x - expected_center)**2 for x in values)) or 1.
                close(center, expected_center); close(scale, expected_scale)
            for features, row, sid in zip(data['X'], entry['rows'], ids):
                score = math.fsum((v - center) / scale * weight for v, center, scale, weight in zip(
                    features, data['mean'], data['scale'], data['coefficient'])) + data['intercept']
                error = abs(score - row['score']); assert error < 1e-10
                output['max_score_error'] = max(error, output['max_score_error'])
                assert row['pid'] == sid and row['label'] == labels[sid]
            for membership, rows in [('train', entry['rows'][:24]), ('test', entry['rows'][24:])]:
                assert all(r['membership'] == membership for r in rows)
                positives = [r['score'] for r in rows if r['label'] == 1]
                negatives = [r['score'] for r in rows if r['label'] == 0]
                wins = sum((a > b) + .5 * (a == b) for a in positives for b in negatives)
                count = len(positives) * len(negatives)
                ncorrect = sum((r['score'] > 0) == r['label'] for r in rows)
                close(wins / count, entry[membership]['auc'])
                assert ncorrect == entry[membership]['correct']
                if membership == 'test':
                    credit += wins; pairs += count; correct += ncorrect
                    held += [r['pid'] for r in rows]
                    tp += sum(r['label'] == 1 and r['score'] > 0 for r in rows)
                    tn += sum(r['label'] == 0 and r['score'] <= 0 for r in rows)
            output['verified_classifiers'] += 1
        assert sorted(held) == split['subjects'] and pairs == 62
        summary = dict(auc=credit / pairs, accuracy=correct / 32, correct=correct,
                       balanced_accuracy=.5 * (tp / 14 + tn / 18))
        for key, value in summary.items():
            close(value, result['summary'][key])
        output['genotype'][arm] = summary
    observed = {(s, w) for s in split['subjects'] for w in ('Week 15', 'Week 18', 'Week 20') if (s, w) in lookup}
    assert {(r['pid'], r['week']) for r in report['forecast_rows']} == observed
    assert len(observed) == len(report['forecast_rows'])
    for row in report['forecast_rows']:
        sid, week = row['pid'], row['week']
        fold = next(f for f in split['folds'] if sid in f['test_subjects'])
        assert fold['key'] == row['fold']
        slot = {'Week 15': 1, 'Week 18': 2, 'Week 20': 3}[week]
        actual = lookup[(sid, week)]
        vectors = [lookup[(s, week)] for s in fold['train_subjects'] if (s, week) in lookup]
        centroid = [mean(float(v[j]) for v in vectors) for j in range(768)]
        expected = dict(current_cosine=cosine(np.load(OLD / fold['key'] / f'{sid}_ts{slot}.npy'), actual),
                        residual_cosine=cosine(np.load(new / fold['key'] / f'{sid}_ts{slot}.npy'), actual),
                        persistence_cosine=cosine(lookup[(sid, 'Week 12')], actual),
                        centroid_cosine=cosine(centroid, actual))
        for key, value in expected.items():
            # The experiment's centroid used an FP32 mean; the independent
            # scalar calculation uses double precision over the same mice.
            close(value, row[key], tolerance=2e-7 if key == 'centroid_cosine' else 1e-12)
    keys = ['current_cosine', 'residual_cosine', 'persistence_cosine', 'centroid_cosine']
    for key in keys:
        value = mean(mean(r[key] for r in report['forecast_rows'] if r['pid'] == sid) for sid in split['subjects'])
        close(value, report['forecast_subject_mean'][key]); output['forecast_subject_mean'][key] = value
    for week, summary in report['forecast_by_horizon'].items():
        rows = [r for r in report['forecast_rows'] if r['week'] == week]
        assert len(rows) == summary['n']
        for key in keys:
            close(mean(r[key] for r in rows), summary[key])
    output['observed_future_scans'] = len(observed)
    auc = {a: v['auc'] for a, v in output['genotype'].items()}; cos = output['forecast_subject_mean']
    advance = (auc['residual_future'] >= auc['baseline'] and auc['residual_combined'] >= auc['baseline']
               and all(cos['residual_cosine'] > cos[k] for k in keys if k != 'residual_cosine'))
    assert report['advance_criterion_passed'] == advance
    output['advance_criterion_passed'] = advance
    (ROOT / 'independent_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
