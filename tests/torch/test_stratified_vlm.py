import copy
import json

import numpy as np
import pytest

pytestmark = pytest.mark.torch


def records():
    rows = []
    for i in range(32):
        label = int(i < 14); sid = f"NaF_{'KO' if label else 'WT'}_{i:02d}"
        for kind in ('genotype', 'tbr', 'combined'):
            targets = [float(i+1), float(i+2) if i % 2 else -1., float(i+3), -1.]
            if kind == 'genotype':
                targets = [-1.] * 4
            rows.append(dict(pid=sid, qid=f'{sid}_{kind}', content_type=kind, target_schema_version=1,
                             future_weeks=['Week 15', 'Week 18', 'Week 20'], input_week='Week 12',
                             answer_vqa_numeric=dict(genotype=label, tbr=targets)))
    return rows


def predictions(module, truth, split):
    from utils.target_contract import fold_target_statistics
    parts = {}
    for fold in split['folds']:
        means = fold_target_statistics([r for r in truth if r['pid'] in fold['train_subjects']])['mean']
        parts[fold['key']] = [dict(pid=r['pid'], qid=r['qid'], content_type=r['content_type'],
                                  genotype_label=r['answer_vqa_numeric']['genotype'],
                                  genotype_logit=float(r['answer_vqa_numeric']['genotype']),
                                  tbr_targets=r['answer_vqa_numeric']['tbr'],
                                  tbr_regression=[v+1 if v>=0 else 0 for v in r['answer_vqa_numeric']['tbr']],
                                  train_mean_prediction=means)
                              for r in truth if r['pid'] in fold['test_subjects']]
    return parts


def test_fixed_splits_and_auc_are_invariant_to_cross_model_offsets(scripts):
    module = scripts('stratified_vlm_protocol'); truth = records(); split = module.make_split(truth)
    module.validate_split(split, truth)
    assert [len(f['test_subjects']) for f in split['folds']] == [8] * 4
    parts = predictions(module, truth, split)
    before = module.score_folds(parts, truth, split)
    assert before['genotype_auc_within_fold_pair_weighted'] == 1
    assert before['genotype_pairs'] == 62
    assert before['subject_mae'] == 1
    assert before['training_majority_accuracy'] == 18/32
    # Cross-model offsets must not contaminate within-model rankings.
    for i, rows in enumerate(parts.values()):
        for row in rows:
            row['genotype_logit'] += 1000 * (-1)**i
    after = module.score_folds(parts, truth, split)
    assert after['genotype_auc_within_fold_pair_weighted'] == 1
    assert after['subject_mae'] == before['subject_mae']
    for rows in parts.values():
        for row in rows:
            row['genotype_logit'] = .7
    assert module.score_folds(parts, truth, split)['genotype_auc_within_fold_pair_weighted'] == .5


def test_unequal_pair_counts_use_declared_weighting(scripts):
    module = scripts('stratified_vlm_protocol'); truth = records(); split = module.make_split(truth)
    parts = predictions(module, truth, split)
    for row in parts['fold_00']:
        row['genotype_logit'] *= -1
    result = module.score_folds(parts, truth, split)
    assert result['folds'][0]['genotype_auc'] == 0
    assert result['genotype_auc_within_fold_pair_weighted'] == pytest.approx(46/62)


@pytest.mark.parametrize('damage', ['split', 'missing_fold', 'wrong_mean', 'duplicate', 'wrong_label'])
def test_amended_scoring_rejects_invalid_inputs(scripts, damage):
    module = scripts('stratified_vlm_protocol'); truth = records(); split = module.make_split(truth)
    parts = predictions(module, truth, split)
    if damage == 'split':
        split['folds'][0]['train_subjects'].append(split['folds'][0]['test_subjects'][0])
    elif damage == 'missing_fold':
        parts.pop('fold_00')
    elif damage == 'wrong_mean':
        for row in parts['fold_00']:
            row['train_mean_prediction'] = [0.] * 4
    elif damage == 'duplicate':
        parts['fold_00'].append(parts['fold_00'][0])
    else:
        parts['fold_00'][0]['genotype_label'] = 1 - parts['fold_00'][0]['genotype_label']
    with pytest.raises(ValueError):
        module.score_folds(parts, truth, split)


def test_all_outer_test_futures_are_excluded_and_training_query_is_cross_fitted(scripts):
    module = scripts('build_stratified_forecasts')
    from stratified_vlm_protocol import make_split
    split = make_split(records()); fold = split['folds'][0]
    rng = np.random.default_rng(11)
    lookup = {(s, f'Week {w}'):rng.normal(size=8).astype('float32')
              for s in split['subjects'] for w in (12, 15, 18, 20)}
    settings = dict(epochs=2, hidden=8, inner_folds=5)
    before, manifest = module.build_forecasts(lookup, split, fold, settings)
    changed = copy.deepcopy(lookup)
    for s in fold['test_subjects']:
        for w in (15, 18, 20):
            changed[(s, f'Week {w}')] *= -100
    after, after_manifest = module.build_forecasts(changed, split, fold, settings)
    assert manifest == after_manifest
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    # A training query's own future cannot affect its supplied forecasts.
    query = fold['train_subjects'][0]; changed = copy.deepcopy(lookup)
    changed[(query, 'Week 15')] *= -100
    after, _ = module.build_forecasts(changed, split, fold, settings)
    for slot in (1, 2, 3):
        np.testing.assert_array_equal(before[f'{query}_ts{slot}.npy'], after[f'{query}_ts{slot}.npy'])
    # The same allowed training change must affect outer-test forecasts.
    test = fold['test_subjects'][0]
    assert not np.array_equal(before[f'{test}_ts1.npy'], after[f'{test}_ts1.npy'])


def test_forecast_manifest_rejects_any_outer_test_member_and_stale_tokens(scripts, tmp_path):
    module = scripts('build_stratified_forecasts')
    from stratified_vlm_protocol import make_split, verify_forecast
    from utils.research_io import atomic_json
    split = make_split(records()); fold = split['folds'][0]
    rng = np.random.default_rng(10)
    lookup = {(s, f'Week {w}'):rng.normal(size=8).astype('float32')
              for s in split['subjects'] for w in (12, 15, 18, 20)}
    settings = dict(module.DEFAULT_SETTINGS, epochs=1, hidden=8)
    manifest = module.export_fold(lookup, split, fold, tmp_path, settings, 'cpu')
    assert module.export_fold(lookup, split, fold, tmp_path, settings, 'cpu') == manifest
    root = tmp_path / fold['key']; damaged = copy.deepcopy(manifest)
    damaged['fits']['fit_0']['fit_subjects'].append(fold['test_subjects'][-1])
    atomic_json(root / 'manifest.json', damaged)
    with pytest.raises(ValueError, match='leaks'):
        verify_forecast(root, fold, split)
    atomic_json(root / 'manifest.json', manifest)
    np.save(root / f"{fold['test_subjects'][0]}_ts1.npy", np.zeros(8))
    with pytest.raises(ValueError, match='changed'):
        verify_forecast(root, fold, split)
