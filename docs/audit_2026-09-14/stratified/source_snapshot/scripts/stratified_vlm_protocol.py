"""Validation amendment: fixed mixed-class subject folds and within-model AUC.

Kept separate from the frozen LOSO implementation so its results remain
reproducible. The existing dataset, training, checkpoint and proxy scorer are
reused without changing their behavior.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'vlm'))
sys.path.insert(0, str(REPO / 'scripts'))
from utils.research_io import file_sha256, object_sha256
from utils.run_contract import validate_records, make_run_spec, completed_fold
from utils.target_contract import unique_subject_targets
from data.validated_metrics import score_predictions, collapse_predictions, regression_metrics

SOURCES = ('stratified_vlm_protocol.py', 'build_stratified_forecasts.py',
           'run_stratified_vlm.py', 'aggregate_stratified_vlm.py')


def make_split(records):
    subjects = validate_records(records)
    targets = unique_subject_targets(records)
    labels = [targets[s]['genotype'] for s in subjects]
    folds = []
    for i, (train, test) in enumerate(StratifiedKFold(4, shuffle=True, random_state=0).split(subjects, labels)):
        folds.append(dict(key=f'fold_{i:02d}', train_subjects=[subjects[j] for j in train],
                          test_subjects=[subjects[j] for j in test]))
    return dict(format_version=1, split_method='StratifiedKFold', n_splits=4, split_seed=0,
                subjects=subjects, folds=folds, seeds=[0, 1, 2], requested_epochs=20,
                genotype_metric='within_fold_positive_negative_pair_weighted_AUROC',
                genotype_threshold=0.0, primary_regression_metric='subject_averaged_future_proxy_MAE',
                amendment='Chosen after inspecting original pooled LOSO results; no model or parameter search',
                population_limit='Subject holdout only; acquisition group and diet remain confounded')


def validate_split(split, records):
    if split != make_split(records):
        raise ValueError('Split differs from the frozen four-fold seed-0 protocol')
    targets = unique_subject_targets(records)
    seen = []
    for fold in split['folds']:
        train, test = set(fold['train_subjects']), set(fold['test_subjects'])
        if train & test or train | test != set(split['subjects']):
            raise ValueError('Invalid train/test membership')
        if {targets[s]['genotype'] for s in test} != {0, 1}:
            raise ValueError('Every test fold must contain both genotypes')
        seen += fold['test_subjects']
    if sorted(seen) != split['subjects']:
        raise ValueError('Every subject must occur in exactly one test fold')


def verify_forecast(directory, fold, split):
    from sklearn.model_selection import KFold
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('format_version') != 2 or manifest.get('fold') != fold or
            manifest.get('split_sha256') != object_sha256(split)):
        raise ValueError('Forecast belongs to a different outer split')
    subjects = split['subjects']; train = fold['train_subjects']
    if sorted(manifest['queries']) != subjects or sorted(manifest['baseline_sha256']) != subjects:
        raise ValueError('Forecast population mismatch')
    settings = manifest['settings']
    expected = {}
    for i, (fit_idx, query_idx) in enumerate(KFold(settings['inner_folds'], shuffle=True,
                                                  random_state=settings['seed']).split(train)):
        expected[f'fit_{i}'] = ([train[j] for j in fit_idx], [train[j] for j in query_idx])
    expected[f"fit_{settings['inner_folds']}"] = (train, fold['test_subjects'])
    if set(expected) != set(manifest['fits']):
        raise ValueError('Unexpected forecast fit count')
    for key, (members, queries) in expected.items():
        fit = manifest['fits'][key]
        if (fit['fit_subjects'] != members or fit.get('genotype_conditioning') is not False or
                fit.get('export_mode') != 'baseline_rollout'):
            raise ValueError('Forecast fit leaks a query or outer-test subject, or uses a different protocol')
        if not np.isfinite(fit['norm_reference']) or fit['norm_reference'] <= 0:
            raise ValueError('Invalid training-only forecast scale')
        for sid in queries:
            if manifest['queries'].get(sid) != key:
                raise ValueError('Wrong cross-fitting assignment')
    expected_files = {f'{s}_ts{k}.npy' for s in subjects for k in (1, 2, 3)}
    if set(manifest['files']) != expected_files:
        raise ValueError('Incomplete forecast token coverage')
    for name, digest in manifest['files'].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f'Forecast token changed: {name}')
    return manifest


def make_spec(params, records, truth_path, split):
    validate_split(split, records)
    if (params['train']['num_train_epochs'] != 20 or params['train']['pool_at'] != 'question_eos' or
            params['train']['load_best_model_at_end'] or params['data']['data_seed'] not in split['seeds']):
        raise ValueError('Training differs from the authorized fixed protocol')
    if params['data']['img_tokens'] not in (1, 4) or params['data'].get('predicted_emb_dir'):
        raise ValueError('Expected baseline-only or nested four-token arm')
    spec = make_run_spec(params, records, truth_path)
    spec['evaluation_protocol'] = split
    spec['evaluation_protocol_sha256'] = object_sha256(split)
    for name in SOURCES:
        spec['source_sha256']['scripts/' + name] = file_sha256(REPO / 'scripts' / name)
    return spec


def score_folds(parts, records, split):
    validate_split(split, records)
    if set(parts) != {f['key'] for f in split['folds']}:
        raise ValueError('Complete four-fold predictions are required')
    all_rows = []; folds = []; subjects = {}; mean_subjects = {}
    total_pairs = 0; auc_sum = 0.; correct = 0.; majority_correct = 0
    for fold in split['folds']:
        train = [r for r in records if r['pid'] in fold['train_subjects']]
        truth = [r for r in records if r['pid'] in fold['test_subjects']]
        rows = parts[fold['key']]
        metrics = score_predictions(rows, truth, train)
        collapsed = collapse_predictions(rows, truth)
        labels = [r['genotype_label'] for r in collapsed.values()]
        pairs = labels.count(0) * labels.count(1)
        if not pairs:
            raise ValueError('Cannot omit a one-class test fold')
        training_labels = [r['genotype'] for r in unique_subject_targets(train).values()]
        majority = int(sum(training_labels) > len(training_labels) / 2)  # Tie -> WT, fixed.
        majority_n = sum(y == majority for y in labels)
        folds.append(dict(key=fold['key'], test_subjects=fold['test_subjects'],
                          genotype_auc=metrics['genotype_auc'], genotype_acc=metrics['genotype_acc'],
                          n=len(labels), positive_negative_pairs=pairs,
                          training_majority_label=majority, majority_accuracy=majority_n / len(labels),
                          subject_mae=metrics['tbr_reg_subject_mae']))
        total_pairs += pairs; auc_sum += pairs * metrics['genotype_auc']
        correct += len(labels) * metrics['genotype_acc']; majority_correct += majority_n
        subjects.update(metrics['per_subject_mae']); mean_subjects.update(metrics['per_subject_train_mean_mae'])
        all_rows += rows
    collapsed = collapse_predictions(all_rows, records)
    horizons = {}
    for slot, week in enumerate((15, 18, 20)):
        entries = [e for e in collapsed.values() if e['tbr_targets'][slot] >= 0]
        y = [e['tbr_targets'][slot] for e in entries]
        horizons[str(week)] = dict(**regression_metrics(y, [e['tbr_predictions'][slot] for e in entries]),
                                  train_mean=regression_metrics(y, [e['train_mean'][slot] for e in entries]))
    n = len(collapsed)
    return dict(evaluation_protocol_sha256=object_sha256(split), n_subjects=n, n_records=len(all_rows),
                genotype_auc_within_fold_pair_weighted=auc_sum / total_pairs,
                genotype_auc_weighting='Number of KO-WT pairs within each test fold; no cross-model comparisons',
                genotype_pairs=total_pairs, genotype_accuracy=correct / n,
                training_majority_accuracy=majority_correct / n, genotype_threshold=0.,
                subject_mae=float(np.mean(list(subjects.values()))),
                train_mean_subject_mae=float(np.mean(list(mean_subjects.values()))),
                per_subject_mae=subjects, per_subject_train_mean_mae=mean_subjects,
                horizons=horizons, folds=folds,
                inference_limit='Descriptive repeated fits on 32 mice; no independent-group inference or full-refit null')


def collect_parts(directory, run_id, records, split):
    parts = {}
    for fold in split['folds']:
        truth = [r for r in records if r['pid'] in fold['test_subjects']]
        part = completed_fold(Path(directory) / fold['key'], run_id, truth)
        if part is not None:
            # Split files are hashed by the new runner and also checked against truth here.
            for name, members in [('train', fold['train_subjects']), ('val', fold['test_subjects'])]:
                actual = json.loads((Path(directory) / fold['key'] / 'splits' / f'{name}.json').read_text())
                if actual != [r for r in records if r['pid'] in members]:
                    raise ValueError('Saved fold records differ from declared split')
            parts[fold['key']] = part
    return parts
