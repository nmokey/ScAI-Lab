"""Freeze, fit, and evaluate one residual full-rollout development experiment."""
import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import sklearn
import torch
from safetensors.numpy import load_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from residual_rollout_forecast import SETTINGS, build_fold, verify_fold, fit_digest
from build_nested_forecasts import embedding_lookup, WEEK_ORDER
from stratified_vlm_protocol import validate_split, verify_forecast
from probe_stratified_longitudinal_genotype import metrics
from utils.research_io import atomic_json, file_sha256, object_sha256
from utils.target_contract import unique_subject_targets

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / 'docs/audit_2026-09-14/residual_rollout_experiment'
PARENT = REPO / 'docs/audit_2026-09-14/stratified'
RECORDS = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')
EMBEDDINGS = Path('/data1/Processed_NIfTI_Test/embeddings/raddino/raddino_embeddings.npz')
OLD = Path('/data1/Processed_NIfTI_Test/embeddings/longitudinal_stratified_20260914')
NEW = Path('/data1/Processed_NIfTI_Test/embeddings/longitudinal_residual_rollout_20260915')
SOURCES = ['residual_rollout_forecast.py', 'run_residual_rollout_experiment.py',
           'probe_stratified_longitudinal_genotype.py', 'build_nested_forecasts.py',
           'stratified_vlm_protocol.py', 'train_longitudinal.py']
ARMS = {'baseline': ('old', [0]), 'current_future': ('old', [1, 2, 3]),
        'current_combined': ('old', [0, 1, 2, 3]), 'residual_future': ('new', [1, 2, 3]),
        'residual_combined': ('new', [0, 1, 2, 3])}


def prepare():
    records = json.loads(RECORDS.read_text()); split = json.loads((PARENT / 'PROTOCOL.json').read_text())
    validate_split(split, records)
    data = np.load(EMBEDDINGS, allow_pickle=True)
    lookup = embedding_lookup(data['embeddings'], data['subject_ids'], data['weeks'])
    manifests = {}
    for fold in split['folds']:
        manifests[fold['key']] = verify_forecast(OLD / fold['key'], fold, split)
        # Check old fit provenance against the current encoder archive too.
        for fit in manifests[fold['key']]['fits'].values():
            assert fit['training_digest'] == fit_digest(lookup, set(fit['fit_subjects']))
    for row in records:
        baseline = load_file(row['embedding_path_ts0'])['embeddings'][0]
        np.testing.assert_array_equal(baseline, lookup[(row['pid'], 'Week 12')])
    protocol = dict(
        method='baseline_anchored_residual_full_rollout', settings=SETTINGS,
        population=split, embeddings_path=str(EMBEDDINGS), embeddings_sha256=file_sha256(EMBEDDINGS),
        records_path=str(RECORDS), records_sha256=file_sha256(RECORDS),
        source_sha256={name: file_sha256(REPO / 'scripts' / name) for name in SOURCES},
        original_manifests_sha256={fold['key']: file_sha256(OLD / fold['key'] / 'manifest.json') for fold in split['folds']},
        original_probe_sha256=file_sha256(PARENT / 'embedding_probe/results.json'),
        forecast_output=str(NEW), torch_version=str(torch.__version__), sklearn_version=sklearn.__version__,
        architecture='Same two hidden layers of width 512 with LayerNorm/GELU; final residual layer initialized to zero',
        rollout='unit_prediction[t] = normalize(unit_baseline + MLP(unit_prediction[t-1], cohort, step)); first input is unit_baseline',
        training='Full baseline rollout; no detached transitions; all available future visits; no teacher forcing',
        objective='Mean per-subject masked cosine loss + 0.01 * mean squared L2 norm of residual over subjects and all three steps',
        export_scale='Same mean observed consecutive-target norm as the original forecaster, computed within each fit',
        optimizer='Adam lr=0.001, default betas, no weight decay; one full-subject batch per epoch (19/20/24 mice); 300 updates',
        checkpoint='Final fixed update only, no held-out or training-based checkpoint selection',
        supervision='Observed embeddings only; neither genotype inputs nor genotype supervision; no PET labels',
        classifier='Same training-only StandardScaler + L2 LogisticRegression(C=1, max_iter=2000, tol=1e-4, lbfgs, random_state=0)',
        classifier_arms=ARMS, genotype_metric='62 within-fold KO-WT pairs, pair-weighted AUROC; zero-logit accuracy',
        forecast_metric='Primary: average future-target cosine within each mouse, then over 32 mice; report every horizon separately',
        comparisons=['Existing forecaster', 'Baseline persistence', 'Outer-training observed week centroid'],
        advance_criterion='Residual future-only and combined AUROCs at least baseline; residual mean subject cosine exceeds current, persistence, and training-week centroid',
        scope='One pre-fixed model variant and forecast seed schedule; embedding evaluation only; no parameter search',
        interpretation='Development experiment after repeated inspection of these folds; no untouched confirmatory test or independent-acquisition claim')
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol)), 'Frozen experiment protocol changed'
    else:
        atomic_json(path, protocol)
        snapshot = REPORT / 'source_snapshot'; snapshot.mkdir()
        for name in SOURCES:
            (snapshot / name).write_bytes((REPO / 'scripts' / name).read_bytes())
    return records, split, lookup, protocol


def train(split, lookup, protocol, device):
    NEW.mkdir(parents=True, exist_ok=True)
    for fold in split['folds']:
        directory = NEW / fold['key']
        if directory.exists():
            manifest = verify_fold(directory, lookup, split, fold, SETTINGS)
            assert manifest['experiment_protocol_sha256'] == object_sha256(protocol)
            continue
        with tempfile.TemporaryDirectory(dir=NEW, prefix=f'.{fold["key"]}-') as tmp:
            arrays, manifest = build_fold(lookup, split, fold, SETTINGS, device, tmp)
            for name, value in arrays.items():
                np.save(Path(tmp) / name, value)
            manifest['files'] = {name: file_sha256(Path(tmp) / name) for name in arrays}
            manifest['experiment_protocol_sha256'] = object_sha256(protocol)
            atomic_json(Path(tmp) / 'manifest.json', manifest)
            verify_fold(tmp, lookup, split, fold, SETTINGS)
            os.rename(tmp, directory)
        print(f'Completed {fold["key"]}: 6 fits, 96 forecast tokens', flush=True)
    atomic_json(NEW / 'index.json', dict(protocol_sha256=object_sha256(protocol),
                folds={f['key']: file_sha256(NEW / f['key'] / 'manifest.json') for f in split['folds']}))


def cosine(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate(records, split, lookup, protocol):
    targets = unique_subject_targets(records)
    old_probe = json.loads((PARENT / 'embedding_probe/results.json').read_text())
    report = dict(status='running', protocol_sha256=object_sha256(protocol),
                  arms={a: dict(folds=[]) for a in ARMS}, forecast_rows=[], manifests_sha256={})
    warnings.simplefilter('error', ConvergenceWarning)
    for fold in split['folds']:
        old_manifest = verify_forecast(OLD / fold['key'], fold, split)
        new_manifest = verify_fold(NEW / fold['key'], lookup, split, fold, SETTINGS)
        assert new_manifest['experiment_protocol_sha256'] == object_sha256(protocol)
        for key in old_manifest['fits']:
            assert old_manifest['fits'][key]['norm_reference'] == new_manifest['fits'][key]['norm_reference']
        report['manifests_sha256'][fold['key']] = file_sha256(NEW / fold['key'] / 'manifest.json')
        ids = fold['train_subjects'] + fold['test_subjects']
        token_sets = {}
        for kind, root in [('old', OLD), ('new', NEW)]:
            token_sets[kind] = np.stack([np.stack([lookup[(s, 'Week 12')]] + [
                np.load(root / fold['key'] / f'{s}_ts{k}.npy', allow_pickle=False) for k in (1, 2, 3)]) for s in ids]).astype(float)
            assert token_sets[kind].shape == (32, 4, 768) and np.isfinite(token_sets[kind]).all()
        y = np.array([targets[s]['genotype'] for s in ids])
        for arm, (kind, slots) in ARMS.items():
            X = token_sets[kind][:, slots, :].reshape(32, -1)
            classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1., penalty='l2', solver='lbfgs',
                                            max_iter=2000, tol=1e-4, random_state=0, class_weight=None))
            classifier.fit(X[:24], y[:24]); scores = classifier.decision_function(X)
            scaler, model = classifier.steps[0][1], classifier.steps[1][1]
            np.testing.assert_allclose(scores, ((X - scaler.mean_) / scaler.scale_) @ model.coef_[0] + model.intercept_[0], rtol=0, atol=1e-10)
            path = REPORT / f'{arm}_{fold["key"]}.npz'
            np.savez(path, X=X, y=y, subjects=np.array(ids), mean=scaler.mean_, scale=scaler.scale_,
                     coefficient=model.coef_[0], intercept=model.intercept_[0])
            result = dict(fold=fold['key'], train=metrics(y[:24], scores[:24]), test=metrics(y[24:], scores[24:]),
                          artifact_sha256=file_sha256(path), iterations=int(model.n_iter_[0]),
                          rows=[dict(pid=s, label=int(label), score=float(score), membership='train' if i < 24 else 'test')
                                for i, (s, label, score) in enumerate(zip(ids, y, scores))])
            if kind == 'old':
                prior_name = {'baseline': 'baseline', 'current_future': 'predicted_only', 'current_combined': 'baseline_and_predicted'}[arm]
                prior = next(f for f in old_probe['arms'][prior_name]['folds'] if f['fold'] == fold['key'])
                assert result['rows'] == prior['rows'] and result['test'] == prior['test'], 'Original probe no longer reproduces exactly'
            report['arms'][arm]['folds'].append(result)
        for slot, week in enumerate(WEEK_ORDER[1:], 1):
            centroid = np.mean([lookup[(s, week)] for s in fold['train_subjects'] if (s, week) in lookup], axis=0)
            for i, sid in enumerate(ids[24:], 24):
                if (sid, week) not in lookup:
                    continue
                actual = lookup[(sid, week)]; baseline = lookup[(sid, 'Week 12')]
                current = token_sets['old'][i, slot]; residual = token_sets['new'][i, slot]
                report['forecast_rows'].append(dict(pid=sid, fold=fold['key'], week=week,
                    current_cosine=cosine(current, actual), residual_cosine=cosine(residual, actual),
                    persistence_cosine=cosine(baseline, actual), centroid_cosine=cosine(centroid, actual),
                    residual_to_baseline_cosine=cosine(residual, baseline), current_to_baseline_cosine=cosine(current, baseline)))
    for arm, result in report['arms'].items():
        tests = [f['test'] for f in result['folds']]
        assert sum(t['pairs'] for t in tests) == 62
        assert sorted(r['pid'] for f in result['folds'] for r in f['rows'] if r['membership'] == 'test') == split['subjects']
        result['summary'] = dict(auc=sum(t['pair_credit'] for t in tests) / 62,
                                accuracy=sum(t['correct'] for t in tests) / 32,
                                correct=sum(t['correct'] for t in tests),
                                balanced_accuracy=.5 * (sum(t['tp'] for t in tests) / 14 + sum(t['tn'] for t in tests) / 18),
                                train_auc_mean=float(np.mean([f['train']['auc'] for f in result['folds']])))
    rows = report['forecast_rows']
    assert len({(r['pid'], r['week']) for r in rows}) == len(rows)
    assert {r['pid'] for r in rows} == set(split['subjects'])
    keys = ['current_cosine', 'residual_cosine', 'persistence_cosine', 'centroid_cosine']
    report['forecast_by_horizon'] = {w: dict(n=sum(r['week'] == w for r in rows), **{
        k: float(np.mean([r[k] for r in rows if r['week'] == w])) for k in keys}) for w in WEEK_ORDER[1:]}
    report['forecast_per_subject'] = {s: {k: float(np.mean([r[k] for r in rows if r['pid'] == s])) for k in keys} for s in split['subjects']}
    report['forecast_subject_mean'] = {k: float(np.mean([v[k] for v in report['forecast_per_subject'].values()])) for k in keys}
    aucs = {a: v['summary']['auc'] for a, v in report['arms'].items()}; means = report['forecast_subject_mean']
    report['advance_criterion_passed'] = bool(aucs['residual_future'] >= aucs['baseline'] and aucs['residual_combined'] >= aucs['baseline']
        and all(means['residual_cosine'] > means[k] for k in keys if k != 'residual_cosine'))
    report['status'] = 'complete'; atomic_json(REPORT / 'results.json', report)
    render(report)
    print(json.dumps(dict(genotype={a: v['summary'] for a, v in report['arms'].items()},
                         forecast=means, advance_criterion_passed=report['advance_criterion_passed']), indent=2), flush=True)


def render(report):
    lines = ['# Residual full-rollout experiment', '',
             'One fixed development experiment on the existing four subject folds. No genotype supervision, '
             'no parameter search, and no VLM fitting. Every training query receives inner-cross-fitted forecasts; '
             'outer test forecasts use only outer training mice. See [the frozen protocol](PROTOCOL.json).', '',
             '## Direct genotype readout', '',
             '| Input | Within-fold AUROC | Correct / 32 | Balanced accuracy | Mean training AUROC | Fold AUROCs |',
             '|---|---:|---:|---:|---:|---|']
    for arm, entry in report['arms'].items():
        s = entry['summary']; folds = ', '.join(f"{f['test']['auc']:.4f}" for f in entry['folds'])
        lines.append(f"| {arm} | {s['auc']:.4f} | {s['correct']}/32 | {s['balanced_accuracy']:.4f} | {s['train_auc_mean']:.4f} | {folds} |")
    lines += ['', 'Identical fixed standardized logistic readout for all arms. All baseline/current predictions reproduce '
              'the preceding direct-embedding probe exactly. Always-WT accuracy is 18/32 (56.25%). '
              'AUROC counts only the 62 within-model KO-WT pairs.', '', '## Forecast fidelity', '',
              'Cosine similarity to the observed future embedding; higher is better. Each horizon uses the same '
              'available held-out scans for all four predictors. Centroids use outer-training mice only.', '',
              '| Horizon | Observations | Current forecast | Residual forecast | Persistence | Training-week centroid |',
              '|---|---:|---:|---:|---:|---:|']
    for week, entry in report['forecast_by_horizon'].items():
        lines.append(f"| {week} | {entry['n']} | {entry['current_cosine']:.4f} | {entry['residual_cosine']:.4f} | {entry['persistence_cosine']:.4f} | {entry['centroid_cosine']:.4f} |")
    m = report['forecast_subject_mean']
    lines += [f"| Mean within mouse, then across mice | 32 mice | {m['current_cosine']:.4f} | {m['residual_cosine']:.4f} | {m['persistence_cosine']:.4f} | {m['centroid_cosine']:.4f} |", '',
              f"**Predeclared advancement criterion passed: {report['advance_criterion_passed']}.** "
              'The criterion requires both residual genotype readouts to meet baseline AUROC and residual subject-mean '
              'forecast cosine to exceed all three forecast comparators.', '',
              'This combines residual anchoring, full-rollout training, unit-direction inputs, and a fixed residual penalty; '
              'it does not isolate which change caused an effect. Genotype is not disease severity. Acquisition and diet remain '
              'confounded, and these folds have been inspected repeatedly. Results are descriptive development evidence; '
              'no independent-acquisition significance or confirmatory improvement is claimed.', '',
              'Evidence: [raw scores, forecast comparisons, and hashes](results.json). '
              'Checkpoints and all 384 forecast tokens are retained at the output path in the protocol.', '']
    (REPORT / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'all'], default='all')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    records, split, lookup, protocol = prepare()
    if args.stage in ('all', 'train'):
        train(split, lookup, protocol, args.device)
    if args.stage in ('all', 'evaluate'):
        evaluate(records, split, lookup, protocol)


if __name__ == '__main__':
    main()
