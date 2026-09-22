"""Fixed, post-audit linear readout of the existing nested VLM input embeddings.

Diagnostic only: no parameter search, no VLM fitting, no new forecast fitting.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import sklearn
from safetensors.numpy import load_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stratified_vlm_protocol import validate_split, verify_forecast
from utils.research_io import atomic_json, file_sha256
from utils.target_contract import unique_subject_targets

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'docs/audit_2026-09-14/stratified/embedding_probe'
SPLIT = OUTPUT.parent / 'PROTOCOL.json'
RECORDS = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')
FORECASTS = Path('/data1/Processed_NIfTI_Test/embeddings/longitudinal_stratified_20260914')
ARMS = {'baseline': [0], 'predicted_only': [1, 2, 3], 'baseline_and_predicted': [0, 1, 2, 3]}


def metrics(labels, scores):
    y = np.asarray(labels); s = np.asarray(scores)
    assert set(y) == {0, 1} and np.isfinite(s).all()
    positives, negatives = s[y == 1], s[y == 0]
    # Direct pair arithmetic must agree with the library scorer.
    credit = sum(float(a > b) + .5 * float(a == b) for a in positives for b in negatives)
    pairs = len(positives) * len(negatives)
    assert abs(credit / pairs - roc_auc_score(y, s)) < 1e-12
    pred = s > 0
    return dict(auc=credit / pairs, pair_credit=credit, pairs=pairs,
                accuracy=float(np.mean(pred == y)), correct=int(np.sum(pred == y)),
                tp=int(np.sum(pred & (y == 1))), tn=int(np.sum(~pred & (y == 0))),
                fp=int(np.sum(pred & (y == 0))), fn=int(np.sum(~pred & (y == 1))),
                bce=float(np.mean(np.logaddexp(0, s) - y * s)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        purpose='Post-audit diagnostic of linearly readable genotype information in corrected VLM inputs',
        split_path=str(SPLIT), split_sha256=file_sha256(SPLIT),
        records_path=str(RECORDS), records_sha256=file_sha256(RECORDS),
        source_sha256=file_sha256(__file__), sklearn_version=sklearn.__version__,
        representations=ARMS, feature_order='Concatenate the indicated 768-dimensional tokens in time order',
        preprocessing='StandardScaler, fitted only on the 24 training mice in each outer fold',
        classifier=dict(type='LogisticRegression', C=1.0, penalty='l2', solver='lbfgs',
                        max_iter=2000, tol=1e-4, random_state=0, class_weight=None),
        genotype_positive='KO=1', threshold=0.0,
        metric='Within-fold KO-WT pair-weighted AUROC; no cross-model ranking',
        selection='Three representations fixed before fitting; all reported; no tuning or model selection',
        forecast_policy='Reuse verified outer-nested, inner-cross-fitted, genotype-unconditioned forecasts',
        limits=['32 mice, subject holdout, acquisition and diet confounding',
                'No independent-group significance test or population confidence interval',
                'Linear readout diagnostic cannot exclude nonlinear information',
                'Training and test forecast tokens come from different fitted forecasters',
                'Adding dimensions changes the effective regularized feature geometry'])
    protocol_path = OUTPUT / 'PROTOCOL.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Frozen probe protocol changed'
    else:
        atomic_json(protocol_path, protocol)  # Freeze settings before accessing probe outcomes.
    records = json.loads(RECORDS.read_text()); split = json.loads(SPLIT.read_text())
    validate_split(split, records)
    targets = unique_subject_targets(records)
    genotypes = {r['pid']: r for r in records if r['content_type'] == 'genotype'}
    assert len(genotypes) == len(split['subjects']) == 32
    baseline = {}; baseline_hashes = {}
    for sid, record in genotypes.items():
        path = Path(record['embedding_path_ts0'])
        value = load_file(path)['embeddings']
        assert value.shape == (1, 768) and np.isfinite(value).all()
        baseline[sid] = value[0].astype(np.float32)
        baseline_hashes[str(path)] = file_sha256(path)
    report = dict(protocol_sha256=file_sha256(protocol_path), status='running',
                  baseline_files_sha256=baseline_hashes, forecast_manifests_sha256={}, arms={})
    for arm in ARMS:
        report['arms'][arm] = dict(dimensions=768 * len(ARMS[arm]), folds=[])
    warnings.simplefilter('error', ConvergenceWarning)
    for fold in split['folds']:
        directory = FORECASTS / fold['key']
        manifest = verify_forecast(directory, fold, split)
        report['forecast_manifests_sha256'][str(directory / 'manifest.json')] = file_sha256(directory / 'manifest.json')
        tokens = {}
        for sid in split['subjects']:
            assert hashlib.sha256(baseline[sid].tobytes()).hexdigest() == manifest['baseline_sha256'][sid]
            future = [np.load(directory / f'{sid}_ts{i}.npy', allow_pickle=False).astype(np.float32) for i in (1, 2, 3)]
            assert all(v.shape == (768,) and np.isfinite(v).all() and np.linalg.norm(v) > 0 for v in future)
            tokens[sid] = np.stack([baseline[sid]] + future).astype(np.float64)
        for arm, slots in ARMS.items():
            ids = fold['train_subjects'] + fold['test_subjects']
            X = np.stack([tokens[s][slots].reshape(-1) for s in ids])
            y = np.array([targets[s]['genotype'] for s in ids])
            classifier = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs', max_iter=2000, tol=1e-4,
                random_state=0, class_weight=None))
            classifier.fit(X[:24], y[:24])
            scaler, model = classifier.steps[0][1], classifier.steps[1][1]
            assert list(model.classes_) == [0, 1] and scaler.n_samples_seen_ == 24
            scores = classifier.decision_function(X)
            manual = ((X - scaler.mean_) / scaler.scale_) @ model.coef_[0] + model.intercept_[0]
            assert np.allclose(scores, manual, rtol=0, atol=1e-10)
            artifact = OUTPUT / f'{arm}_{fold["key"]}.npz'
            np.savez(artifact, X=X, y=y, subjects=np.asarray(ids), mean=scaler.mean_,
                     scale=scaler.scale_, coefficient=model.coef_[0], intercept=model.intercept_[0])
            result = dict(fold=fold['key'], train=metrics(y[:24], scores[:24]),
                          test=metrics(y[24:], scores[24:]), iterations=int(model.n_iter_[0]),
                          training_majority_correct=int(np.sum(y[24:] == int(y[:24].mean() > .5))),
                          artifact_sha256=file_sha256(artifact),
                          rows=[dict(pid=s, label=int(label), score=float(score),
                                     membership='train' if i < 24 else 'test')
                                for i, (s, label, score) in enumerate(zip(ids, y, scores))])
            report['arms'][arm]['folds'].append(result)
            print(arm, fold['key'], json.dumps(result['test']), flush=True)
    for arm, result in report['arms'].items():
        folds = result['folds']; tests = [f['test'] for f in folds]
        held = [r['pid'] for f in folds for r in f['rows'] if r['membership'] == 'test']
        assert sorted(held) == split['subjects']
        pairs = sum(t['pairs'] for t in tests); assert pairs == 62
        result['summary'] = dict(auc=sum(t['pair_credit'] for t in tests) / pairs,
                                 pairs=pairs, accuracy=sum(t['correct'] for t in tests) / 32,
                                 correct=sum(t['correct'] for t in tests),
                                 balanced_accuracy=.5 * (sum(t['tp'] for t in tests) / 14 + sum(t['tn'] for t in tests) / 18),
                                 majority_accuracy=sum(f['training_majority_correct'] for f in folds) / 32,
                                 train_auc_mean=float(np.mean([f['train']['auc'] for f in folds])))
    report['status'] = 'complete'
    atomic_json(OUTPUT / 'results.json', report)
    lines = ['# Direct genotype readout of corrected longitudinal embeddings', '',
             'Post-audit diagnostic: one fixed standardized L2 logistic classifier per representation and fold. '
             'Same 32 mice and four validated mixed-class subject folds; 24 training mice and eight test mice each. '
             'Only existing baseline scans and nested predicted future tokens are inputs. No VLM or new forecaster fitting. '
             'Scaler statistics come only from training mice. KO=1; fixed zero-logit threshold.', '',
             '| Input | Within-fold AUROC | Correct / 32 | Balanced accuracy | Fold AUROCs | Mean training AUROC |',
             '|---|---:|---:|---:|---|---:|']
    for arm, result in report['arms'].items():
        s = result['summary']; aucs = ', '.join(f"{f['test']['auc']:.4f}" for f in result['folds'])
        lines.append(f"| {arm} | {s['auc']:.4f} | {s['correct']}/32 | {s['balanced_accuracy']:.4f} | {aucs} | {s['train_auc_mean']:.4f} |")
    lines += ['', 'The always-WT comparator scores 18/32 (56.25%) accuracy and 0.5 balanced accuracy. '
              'AUROC aggregates 62 within-model KO-WT pairs; scores from different fitted classifiers are never ranked together.', '',
              'These results describe this fixed linear readout, not the maximum recoverable information. '
              'The small cohort and genotype/diet/acquisition confounding prevent an independent-acquisition genotype claim. '
              'Training tokens use inner cross-fitted forecasters, whereas test tokens use the outer-training forecaster; '
              'that representation shift remains part of the current pipeline. No hyperparameters, thresholds, signs, '
              'or representations were selected based on test performance. All three fixed representations are reported.', '',
              'The older strong longitudinal probes used genotype-conditioned forecasts and are invalid evidence. '
              'The separate all-week RAD-DINO encoder association concerns a different population and observed visits; '
              'it is not a performance estimate for these baseline-derived forecasts.', '',
              'Artifacts: [protocol](PROTOCOL.json), [scores and input hashes](results.json). '
              'Each fold also saves its feature matrix and fitted scaler/classifier for independent reproduction.', '']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
