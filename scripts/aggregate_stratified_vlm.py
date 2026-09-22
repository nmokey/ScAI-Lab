"""Verify and report all six amended runs; never pool genotype scores across folds."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import hashlib

from stratified_vlm_protocol import REPO, validate_split, collect_parts, score_folds, verify_forecast
from utils.research_io import atomic_json, file_sha256, object_sha256


def verified_run(directory, records, truth_path, split):
    directory = Path(directory)
    manifest = json.loads((directory / 'run_manifest.json').read_text()); spec = manifest['spec']
    if manifest['run_id'] != object_sha256(spec):
        raise ValueError('Invalid run manifest identity')
    if spec['evaluation_protocol'] != split or spec['evaluation_protocol_sha256'] != object_sha256(split):
        raise ValueError('Unmatched evaluation splits')
    if spec['data_sha256'] != file_sha256(truth_path):
        raise ValueError('Ground truth changed')
    for name, digest in spec['source_sha256'].items():
        path = REPO / name if name.startswith('scripts/') else REPO / 'vlm' / name
        if file_sha256(path) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    for row in records:
        if file_sha256(row['embedding_path_ts0']) != spec['baseline_sha256'][row['pid']]:
            raise ValueError('Baseline input changed')
    root = spec['config']['data'].get('nested_forecast_root')
    if root:
        root = Path(root)
        if file_sha256(root / 'index.json') != spec['nested_index_sha256']:
            raise ValueError('Nested forecast index changed')
        index = json.loads((root / 'index.json').read_text())
        if index['split_sha256'] != object_sha256(split):
            raise ValueError('Nested split mismatch')
        for fold in split['folds']:
            if file_sha256(root / fold['key'] / 'manifest.json') != index['folds'][fold['key']]:
                raise ValueError('Nested forecast manifest changed')
            forecast = verify_forecast(root / fold['key'], fold, split)
            for sid, path in {r['pid']:r['embedding_path_ts0'] for r in records}.items():
                vector = load_file(path)['embeddings'].contiguous().numpy()
                if hashlib.sha256(vector.tobytes()).hexdigest() != forecast['baseline_sha256'][sid]:
                    raise ValueError('Forecast baseline mismatch')
    parts = collect_parts(directory, manifest['run_id'], records, split)
    result = score_folds(parts, records, split)
    if json.loads((directory / 'stratified_results.json').read_text()) != result:
        raise ValueError('Saved scores do not reproduce')
    expected = [r for fold in split['folds'] for r in parts[fold['key']]]
    if json.loads((directory / 'vqa_stratified.json').read_text()) != expected:
        raise ValueError('Saved aggregate differs from verified folds')
    return result, spec


def comparable(spec):
    value = copy.deepcopy(spec)
    value.pop('nested_index_sha256')
    value['config']['data'].pop('nested_forecast_root', None)
    value['config']['data'].pop('img_tokens')
    value['config']['data'].pop('data_seed')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True); p.add_argument('--truth', required=True)
    p.add_argument('--split', required=True); p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    records = json.loads(Path(args.truth).read_text()); split = json.loads(Path(args.split).read_text())
    validate_split(split, records)
    pairs = []; reference = None
    for seed in split['seeds']:
        pair = dict(seed=seed)
        for arm in ('base', 'long'):
            directory = Path(args.runs) / f'stratified_{arm}_seed{seed}'
            result, spec = verified_run(directory, records, args.truth, split)
            if spec['config']['data']['data_seed'] != seed or spec['config']['data']['img_tokens'] != (1 if arm == 'base' else 4):
                raise ValueError('Wrong seed or arm')
            current = comparable(spec)
            if reference is not None and current != reference:
                raise ValueError('Unmatched training or data contracts')
            reference = current
            pair[arm] = result
            pair[arm + '_run'] = str(directory)
        pair['delta_genotype_auc'] = (pair['long']['genotype_auc_within_fold_pair_weighted'] -
                                      pair['base']['genotype_auc_within_fold_pair_weighted'])
        pair['delta_subject_mae'] = pair['long']['subject_mae'] - pair['base']['subject_mae']
        pairs.append(pair)
    report = dict(protocol=split, truth_sha256=file_sha256(args.truth), per_seed=pairs,
                  mean_delta_genotype_auc=float(np.mean([p['delta_genotype_auc'] for p in pairs])),
                  mean_delta_subject_mae=float(np.mean([p['delta_subject_mae'] for p in pairs])),
                  status='complete', interpretation='Evaluation amendment after original LOSO results; no parameter search')
    lines = ['# Amended four-fold VLM evaluation', '',
             'All 24 fits passed checkpoint, input, split and prediction checks. This evaluation was specified after inspecting the original LOSO results; model settings and targets were retained.', '',
             'Genotype AUROC averages within-model test-fold AUROCs, weighted by the number of KO–WT pairs in each fold. It never ranks predictions from different fitted models against each other.', '',
             '| Seed | Baseline AUROC | Longitudinal AUROC | Difference | Baseline accuracy | Longitudinal accuracy | Training-majority accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        b, l = p['base'], p['long']
        lines.append(f"| {p['seed']} | {b['genotype_auc_within_fold_pair_weighted']:.4f} | {l['genotype_auc_within_fold_pair_weighted']:.4f} | {p['delta_genotype_auc']:+.4f} | {b['genotype_accuracy']:.4f} | {l['genotype_accuracy']:.4f} | {b['training_majority_accuracy']:.4f} |")
    lines += ['', '| Seed | Baseline subject MAE | Longitudinal subject MAE | Difference | Training-mean MAE |',
              '|---|---:|---:|---:|---:|']
    for p in pairs:
        lines.append(f"| {p['seed']} | {p['base']['subject_mae']:.4f} | {p['long']['subject_mae']:.4f} | {p['delta_subject_mae']:+.4f} | {p['base']['train_mean_subject_mae']:.4f} |")
    lines += ['', '## Every test fold', '',
              '| Seed | Fold | KO–WT pairs | Baseline AUROC | Longitudinal AUROC |', '|---|---|---:|---:|---:|']
    for p in pairs:
        for b, l in zip(p['base']['folds'], p['long']['folds']):
            lines.append(f"| {p['seed']} | {b['key']} | {b['positive_negative_pairs']} | {b['genotype_auc']:.4f} | {l['genotype_auc']:.4f} |")
    lines += ['', '## Every proxy horizon', '',
              '| Seed | Week | n | Baseline MAE | Longitudinal MAE | Training-mean MAE |', '|---|---|---:|---:|---:|---:|']
    for p in pairs:
        for week in ('15', '18', '20'):
            b, l = p['base']['horizons'][week], p['long']['horizons'][week]
            lines.append(f"| {p['seed']} | {week} | {b['n']} | {b['mae']:.4f} | {l['mae']:.4f} | {b['train_mean']['mae']:.4f} |")
    lines += ['', 'Seeds reuse the same 32 mice and split; they are not independent studies. There are 62 within-fold KO–WT comparisons per seed, which are also not independent observations. Training uses 24 mice per fold, versus 31 in the original LOSO study; differences from the old scores cannot be attributed solely to AUROC aggregation. Genotype and diet/acquisition group remain confounded, with only one WT acquisition component. No independent-group claim, significance test, calibration claim or biological validation of the PET proxy follows from these results.', '',
              '[Machine-readable results](paired_results.json) retain per-fold scores, all proxy horizons and subject errors. Original LOSO outputs remain unchanged.', '']
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'paired_results.json', report)
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    atomic_json(output / 'completion.json', dict(status='complete',
                results_sha256=file_sha256(output / 'paired_results.json'),
                markdown_sha256=file_sha256(output / 'RESULTS.md')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('protocol', 'per_seed')}), flush=True)


if __name__ == '__main__':
    main()
