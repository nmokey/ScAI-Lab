"""Rebuild the existing forecaster strictly within the amended outer splits."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nested_forecasts import (DEFAULT_SETTINGS, embedding_lookup, fit_digest,
                                    build_pairs, train_fold, make_input, WEEK_ORDER)
from stratified_vlm_protocol import validate_split, verify_forecast
from utils.research_io import atomic_json, object_sha256, file_sha256


def build_forecasts(lookup, split, fold, settings=None, device='cpu'):
    settings = dict(DEFAULT_SETTINGS, **(settings or {}))
    train = fold['train_subjects']; subjects = split['subjects']
    partitions = []
    for fit_idx, query_idx in KFold(settings['inner_folds'], shuffle=True,
                                     random_state=settings['seed']).split(train):
        partitions.append(([train[i] for i in fit_idx], [train[i] for i in query_idx]))
    partitions.append((train, fold['test_subjects']))
    arrays = {}; fits = {}; queries = {}
    for index, (members, query_subjects) in enumerate(partitions):
        keys = sorted(k for k in lookup if k[0] in members)
        X, Y, _, _, _ = build_pairs(np.stack([lookup[k] for k in keys]),
                                    np.array([k[0] for k in keys]), np.array([k[1] for k in keys]),
                                    use_conditioning=True, use_genotype=False)
        seed = settings['seed'] + index
        torch.manual_seed(seed); np.random.seed(seed)
        model = train_fold(X, Y, X.shape[1], Y.shape[1], settings['hidden'], settings['lr'],
                           settings['epochs'], torch.device(device))
        norm = float(np.linalg.norm(Y, axis=1).mean()); key = f'fit_{index}'
        fits[key] = dict(fit_subjects=members, training_digest=fit_digest(lookup, set(members)),
                         training_pair_count=len(X), norm_reference=norm, seed=seed,
                         genotype_conditioning=False, export_mode='baseline_rollout')
        for sid in query_subjects:
            queries[sid] = key; cur = lookup[(sid, 'Week 12')].copy()
            for slot, (w0, w1) in enumerate(zip(WEEK_ORDER, WEEK_ORDER[1:]), 1):
                x = make_input(cur, sid.split('_')[0], None, w0, w1, True, False)
                with torch.no_grad():
                    cur = model(torch.tensor(x[None], device=device)).cpu().numpy()[0]
                magnitude = float(np.linalg.norm(cur))
                if not np.isfinite(cur).all() or magnitude <= 0:
                    raise ValueError('Invalid forecast')
                cur = np.asarray(cur * (norm / magnitude), dtype=np.float32)
                arrays[f'{sid}_ts{slot}.npy'] = cur.copy()
        del model
    manifest = dict(format_version=2, fold=fold, split_sha256=object_sha256(split), settings=settings,
                    fits=fits, queries=queries, baseline_sha256={
                        sid:hashlib.sha256(lookup[(sid, 'Week 12')].tobytes()).hexdigest() for sid in subjects})
    return arrays, manifest


def export_fold(lookup, split, fold, root, settings, device):
    import fcntl
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    directory = root / fold['key']
    request = dict(split=split, fold=fold, settings=settings,
                   training_digest=fit_digest(lookup, set(fold['train_subjects'])),
                   baselines={sid:hashlib.sha256(lookup[(sid, 'Week 12')].tobytes()).hexdigest()
                              for sid in split['subjects']},
                   sources={name:file_sha256(Path(__file__).with_name(name)) for name in
                            ('build_stratified_forecasts.py', 'stratified_vlm_protocol.py',
                             'build_nested_forecasts.py', 'train_longitudinal.py')})
    with open(root / f".{fold['key']}.lock", 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            manifest = verify_forecast(directory, fold, split)
            if manifest['request_sha256'] != object_sha256(request):
                raise ValueError('Forecast cache changed; choose a new root')
            return manifest
        arrays, manifest = build_forecasts(lookup, split, fold, settings, device)
        manifest['request_sha256'] = object_sha256(request)
        manifest['request'] = request
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{fold['key']}-") as tmp:
            temp = Path(tmp)
            for name, value in arrays.items():
                np.save(temp / name, value)
            manifest['files'] = {name:file_sha256(temp / name) for name in arrays}
            atomic_json(temp / 'manifest.json', manifest)
            verify_forecast(temp, fold, split)
            os.rename(temp, directory)
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embeddings', required=True); p.add_argument('--records', required=True)
    p.add_argument('--split', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    records = json.loads(Path(args.records).read_text()); split = json.loads(Path(args.split).read_text())
    validate_split(split, records)
    data = np.load(args.embeddings, allow_pickle=True)
    lookup = embedding_lookup(data['embeddings'], data['subject_ids'], data['weeks'])
    for fold in split['folds']:
        manifest = export_fold(lookup, split, fold, args.output_dir, DEFAULT_SETTINGS, args.device)
        print(f"Completed {fold['key']}: {len(manifest['files'])} tokens, {len(manifest['fits'])} fits", flush=True)
    atomic_json(Path(args.output_dir) / 'index.json', dict(format_version=2, split_sha256=object_sha256(split),
                settings=DEFAULT_SETTINGS, folds={f['key']:file_sha256(Path(args.output_dir) / f['key'] / 'manifest.json')
                                                 for f in split['folds']}))


if __name__ == '__main__':
    main()
