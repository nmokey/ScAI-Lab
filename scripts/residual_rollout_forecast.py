"""Experimental baseline-anchored residual forecaster; no genotype targets.

Separate from the frozen production forecaster. Fit membership determines every
target and scale statistic; query inference accepts baseline images only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nested_forecasts import build_pairs, fit_digest, WEEK_ORDER
from utils.research_io import file_sha256, object_sha256

SETTINGS = dict(epochs=300, hidden=512, lr=.001, seed=0, inner_folds=5,
                residual_penalty=.01)


class ResidualRollout(nn.Module):
    def __init__(self, dimension=768, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension + 5, hidden), nn.LayerNorm(hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                                 nn.GELU(), nn.Linear(hidden, dimension))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def rollout(self, baseline, cohort):
        base = F.normalize(baseline, dim=-1)
        current = base
        states, deltas = [], []
        for slot in range(3):
            step = baseline.new_zeros((len(baseline), 3)); step[:, slot] = 1
            delta = self.net(torch.cat([current, cohort, step], dim=-1))
            # Each horizon retains the observed baseline anchor. The preceding
            # predicted direction supplies the recurrent input, without detach.
            current = F.normalize(base + delta, dim=-1)
            states.append(current); deltas.append(delta)
        return states, deltas

    def forward(self, baseline, cohort):
        states, deltas = self.rollout(baseline, cohort)
        return torch.stack(states, dim=1), torch.stack(deltas, dim=1)


def rollout_loss(predictions, deltas, targets, mask, penalty):
    if mask.ndim != 2 or not mask.any(dim=1).all():
        raise ValueError('Every training mouse must have at least one observed future')
    safe_targets = torch.where(mask.unsqueeze(-1), targets, torch.zeros_like(targets))
    errors = 1 - F.cosine_similarity(predictions, safe_targets, dim=-1)
    # Equal subject weighting, then equal weight to that subject's observed visits.
    prediction_loss = ((errors * mask).sum(dim=1) / mask.sum(dim=1)).mean()
    residual_loss = deltas.square().sum(dim=-1).mean()
    return prediction_loss + penalty * residual_loss, prediction_loss, residual_loss


def cohort_tensor(subjects, device):
    names = [s.split('_')[0] for s in subjects]
    if not set(names) <= {'NaF', 'FDG'}:
        raise ValueError('Unexpected acquisition cohort')
    return torch.tensor([[1., 0.] if s == 'NaF' else [0., 1.] for s in names], device=device)


def training_arrays(lookup, members, device):
    baseline = np.stack([lookup[(s, 'Week 12')] for s in members])
    dimension = baseline.shape[1]
    target = np.zeros((len(members), 3, dimension), dtype=np.float32)
    mask = np.zeros((len(members), 3), dtype=bool)
    for i, sid in enumerate(members):
        for slot, week in enumerate(WEEK_ORDER[1:]):
            if (sid, week) in lookup:
                target[i, slot] = lookup[(sid, week)]; mask[i, slot] = True
    if not mask.any(axis=1).all():
        raise ValueError('A training subject has no observed future')
    return (torch.tensor(baseline, device=device), cohort_tensor(members, device),
            torch.tensor(target, device=device), torch.tensor(mask, device=device))


def normalization_reference(lookup, members):
    # Match the old forecaster's per-fit export scale exactly, including which
    # observed consecutive-pair targets contribute to the scale statistic.
    keys = sorted(k for k in lookup if k[0] in members)
    _, target, _, _, _ = build_pairs(np.stack([lookup[k] for k in keys]),
                                     np.array([k[0] for k in keys]), np.array([k[1] for k in keys]),
                                     use_conditioning=True, use_genotype=False)
    return float(np.linalg.norm(target, axis=1).mean())


def train_model(lookup, members, settings, seed, device='cpu'):
    torch.manual_seed(seed); np.random.seed(seed)
    baseline, cohort, targets, mask = training_arrays(lookup, members, device)
    model = ResidualRollout(baseline.shape[-1], settings['hidden']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])
    curve = []
    for step in range(settings['epochs'] + 1):
        model.train()
        pred, delta = model(baseline, cohort)
        loss, prediction, residual = rollout_loss(pred, delta, targets, mask, settings['residual_penalty'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        if step % 30 == 0 or step == settings['epochs']:
            curve.append(dict(update=step, objective=float(loss.detach()),
                              cosine_loss=float(prediction.detach()), residual_squared_norm=float(residual.detach())))
        if step == settings['epochs']:
            break
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return model, dict(curve=curve, observed_target_count=int(mask.sum()),
                       norm_reference=normalization_reference(lookup, members), updates=settings['epochs'])


def predict(model, lookup, queries, norm, device):
    # No lookup of any query future is permitted here.
    baseline = torch.tensor(np.stack([lookup[(s, 'Week 12')] for s in queries]), device=device)
    with torch.no_grad():
        unit, _ = model(baseline, cohort_tensor(queries, device))
        value = (unit * norm).cpu().numpy().astype(np.float32)
    if not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=-1), norm, rtol=1e-6):
        raise ValueError('Invalid normalized forecast')
    return {f'{sid}_ts{slot+1}.npy': value[i, slot].copy()
            for i, sid in enumerate(queries) for slot in range(3)}


def partitions(fold, settings):
    train = fold['train_subjects']
    result = [([train[i] for i in fit], [train[i] for i in query])
              for fit, query in KFold(settings['inner_folds'], shuffle=True,
                                     random_state=settings['seed']).split(train)]
    return result + [(train, fold['test_subjects'])]


def build_fold(lookup, split, fold, settings=None, device='cpu', save_directory=None):
    settings = dict(SETTINGS, **(settings or {}))
    arrays, fits, assignments = {}, {}, {}
    for index, (members, queries) in enumerate(partitions(fold, settings)):
        key = f'fit_{index}'
        assert not set(members) & set(queries)
        assert set(members) <= set(fold['train_subjects'])
        model, detail = train_model(lookup, members, settings, settings['seed'] + index, device)
        predictions = predict(model, lookup, queries, detail['norm_reference'], device)
        fits[key] = dict(**detail, fit_subjects=members, query_subjects=queries,
                         training_digest=fit_digest(lookup, set(members)), seed=settings['seed'] + index,
                         genotype_conditioning=False, genotype_supervision=False,
                         export_mode='baseline_rollout')
        if save_directory is not None:
            directory = Path(save_directory); directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{key}.pt'
            torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            dimension=next(iter(predictions.values())).shape[0], settings=settings,
                            norm_reference=detail['norm_reference']), path)
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            restored = ResidualRollout(checkpoint['dimension'], settings['hidden']).to(device)
            restored.load_state_dict(checkpoint['state_dict'], strict=True); restored.eval()
            reproduced = predict(restored, lookup, queries, checkpoint['norm_reference'], device)
            assert all(np.array_equal(predictions[k], reproduced[k]) for k in predictions)
            fits[key]['checkpoint_sha256'] = file_sha256(path)
            fits[key]['reload_exact'] = True
        arrays.update(predictions); assignments.update({s: key for s in queries})
        if save_directory is not None:
            print(f'{fold["key"]}/{key}: {detail["observed_target_count"]} targets; final cosine loss '
                  f'{detail["curve"][-1]["cosine_loss"]:.5f}; reload exact', flush=True)
    manifest = dict(format_version=3, method='baseline_anchored_residual_full_rollout',
                    fold=fold, split_sha256=object_sha256(split), settings=settings,
                    fits=fits, queries=assignments, baseline_sha256={
                        s: hashlib.sha256(lookup[(s, 'Week 12')].tobytes()).hexdigest() for s in split['subjects']})
    return arrays, manifest


def verify_fold(directory, lookup, split, fold, settings):
    directory = Path(directory); manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['format_version'] != 3 or manifest['method'] != 'baseline_anchored_residual_full_rollout'
            or manifest['fold'] != fold or manifest['split_sha256'] != object_sha256(split)
            or manifest['settings'] != settings):
        raise ValueError('Residual forecast protocol mismatch')
    expected_files = {f'{s}_ts{k}.npy' for s in split['subjects'] for k in (1, 2, 3)}
    if set(manifest['files']) != expected_files or set(manifest['queries']) != set(split['subjects']):
        raise ValueError('Residual forecast coverage mismatch')
    expected = partitions(fold, settings)
    if set(manifest['fits']) != {f'fit_{i}' for i in range(len(expected))}:
        raise ValueError('Residual fit coverage mismatch')
    for i, (members, queries) in enumerate(expected):
        key = f'fit_{i}'; fit = manifest['fits'][key]
        if (fit['fit_subjects'] != members or fit['query_subjects'] != queries
                or fit['training_digest'] != fit_digest(lookup, set(members))
                or fit['genotype_conditioning'] is not False or fit['genotype_supervision'] is not False
                or fit['norm_reference'] != normalization_reference(lookup, members)
                or not fit['reload_exact'] or file_sha256(directory / f'{key}.pt') != fit['checkpoint_sha256']):
            raise ValueError('Residual fit membership, input, or checkpoint mismatch')
        for s in queries:
            if manifest['queries'][s] != key:
                raise ValueError('Residual query fit mismatch')
    for sid in split['subjects']:
        if hashlib.sha256(lookup[(sid, 'Week 12')].tobytes()).hexdigest() != manifest['baseline_sha256'][sid]:
            raise ValueError('Residual baseline changed')
    for name, digest in manifest['files'].items():
        if file_sha256(directory / name) != digest:
            raise ValueError('Residual forecast file changed')
    return manifest
