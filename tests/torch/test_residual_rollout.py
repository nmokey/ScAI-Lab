import copy

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.torch


def sample_data():
    subjects = [f'NaF_{"KO" if i < 6 else "WT"}_{i:02d}' for i in range(12)]
    rng = np.random.default_rng(33)
    lookup = {(s, f'Week {w}'): rng.normal(size=8).astype('float32')
              for s in subjects for w in (12, 15, 18, 20)}
    fold = dict(key='fold_00', train_subjects=subjects[:10], test_subjects=subjects[10:])
    return lookup, dict(subjects=subjects), fold


def test_initialization_preserves_baseline_and_final_loss_reaches_first_prediction(scripts):
    m = scripts('residual_rollout_forecast')
    torch.manual_seed(7)
    model = m.ResidualRollout(8, 12)
    x = torch.randn(3, 8); cohort = torch.tensor([[1., 0.]] * 3)
    pred, delta = model(x, cohort)
    torch.testing.assert_close(pred, torch.nn.functional.normalize(x, dim=-1)[:, None].expand(-1, 3, -1))
    assert torch.count_nonzero(delta) == 0
    # Nonzero residual weights expose the recurrent gradient path after updates.
    torch.nn.init.normal_(model.net[-1].weight, std=.02)
    states, _ = model.rollout(x, cohort)
    target = torch.randn_like(states[-1])
    loss = (1 - torch.nn.functional.cosine_similarity(states[-1], target, dim=-1)).mean()
    gradient = torch.autograd.grad(loss, states[0])[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_missing_visits_never_change_loss_and_subjects_have_equal_weight(scripts):
    m = scripts('residual_rollout_forecast')
    pred = torch.tensor([[[1., 0.]] * 3, [[1., 0.]] * 3], requires_grad=True)
    targets = torch.tensor([[[1., 0.]] * 3, [[0., 1.]] * 3])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    deltas = torch.zeros_like(pred)
    before = m.rollout_loss(pred, deltas, targets, mask, .01)[0]
    assert before.item() == pytest.approx(.5)
    targets[~mask] = float('nan')
    after = m.rollout_loss(pred, deltas, targets, mask, .01)[0]
    assert before.item() == after.item()
    after.backward(); assert torch.count_nonzero(pred.grad[~mask]) == 0
    mask[0] = False
    with pytest.raises(ValueError, match='at least one'):
        m.rollout_loss(pred, deltas, targets, mask, .01)


def test_outer_future_exclusion_query_crossfit_and_allowed_target_positive_control(scripts):
    m = scripts('residual_rollout_forecast')
    lookup, split, fold = sample_data()
    settings = dict(epochs=3, hidden=8, inner_folds=2)
    before, manifest = m.build_fold(lookup, split, fold, settings)
    changed = copy.deepcopy(lookup)
    for sid in fold['test_subjects']:
        for week in ('Week 15', 'Week 18', 'Week 20'):
            changed[(sid, week)] *= -100
    after, after_manifest = m.build_fold(changed, split, fold, settings)
    assert manifest == after_manifest
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    query = fold['train_subjects'][0]; changed = copy.deepcopy(lookup)
    changed[(query, 'Week 15')] *= -10
    after, _ = m.build_fold(changed, split, fold, settings)
    for slot in (1, 2, 3):
        np.testing.assert_array_equal(before[f'{query}_ts{slot}.npy'], after[f'{query}_ts{slot}.npy'])
    assert not np.array_equal(before[f'{fold["test_subjects"][0]}_ts3.npy'],
                              after[f'{fold["test_subjects"][0]}_ts3.npy'])


def test_genotype_name_is_not_an_input_and_checkpoint_files_are_checked(scripts, tmp_path):
    m = scripts('residual_rollout_forecast')
    lookup, split, fold = sample_data()
    settings = dict(m.SETTINGS, epochs=2, hidden=8, inner_folds=2)
    arrays, manifest = m.build_fold(lookup, split, fold, settings, save_directory=tmp_path)
    def rename(s):
        return s.replace('_KO_', '_TEMP_').replace('_WT_', '_KO_').replace('_TEMP_', '_WT_')
    renamed = {(rename(s), w): v for (s, w), v in lookup.items()}
    other_fold = {**fold, 'train_subjects': [rename(s) for s in fold['train_subjects']],
                  'test_subjects': [rename(s) for s in fold['test_subjects']]}
    other, _ = m.build_fold(renamed, dict(subjects=[rename(s) for s in split['subjects']]), other_fold, settings)
    for name in arrays:
        # Renaming changes the order of the legacy FP32 scale reduction, but
        # must not affect predictions beyond its floating-point roundoff.
        np.testing.assert_allclose(arrays[name], other[rename(name)], rtol=3e-7, atol=1e-7)
    for name, value in arrays.items():
        np.save(tmp_path / name, value)
    manifest['files'] = {name: m.file_sha256(tmp_path / name) for name in arrays}
    from utils.research_io import atomic_json
    atomic_json(tmp_path / 'manifest.json', manifest)
    m.verify_fold(tmp_path, lookup, split, fold, settings)
    np.save(tmp_path / next(iter(arrays)), np.zeros(8))
    with pytest.raises(ValueError, match='file changed'):
        m.verify_fold(tmp_path, lookup, split, fold, settings)
