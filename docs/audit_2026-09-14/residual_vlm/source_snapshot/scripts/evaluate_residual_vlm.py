"""Prepare and verify the matched three-seed residual-input VLM comparison."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from residual_vlm_protocol import REPO, ADDED_SOURCES, make_residual_spec, verify_residual_inputs
from aggregate_stratified_vlm import verified_run
from stratified_vlm_protocol import collect_parts, score_folds
from check_stratified_metrics import check_run
from utils.research_io import atomic_json, file_sha256, object_sha256

OUTPUT = REPO / 'docs/audit_2026-09-14/residual_vlm'
PARENT = REPO / 'docs/audit_2026-09-14/stratified'
FORECAST = REPO / 'docs/audit_2026-09-14/residual_rollout_experiment'
RUNS = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/runs')
TRUTH = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')


def prepare():
    records = json.loads(TRUTH.read_text()); split = json.loads((PARENT / 'PROTOCOL.json').read_text())
    forecasts = json.loads((FORECAST / 'PROTOCOL.json').read_text())
    for fold in split['folds']:
        verify_residual_inputs(forecasts['forecast_output'], fold, split, records)
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / 'configs').mkdir(exist_ok=True)
    references = {}; specs = {}
    for seed in split['seeds']:
        for arm in ('base', 'long'):
            directory = RUNS / f'stratified_{arm}_seed{seed}'
            result, original = verified_run(directory, records, str(TRUTH), split)
            references[directory.name] = dict(run_id=object_sha256(original),
                    results_sha256=file_sha256(directory / 'stratified_results.json'),
                    manifest_sha256=file_sha256(directory / 'run_manifest.json'))
            if arm == 'long':
                params = yaml.safe_load((PARENT / 'configs' / f'long_seed{seed}.yml').read_text())
                params['data']['nested_forecast_root'] = forecasts['forecast_output']
                params['exp']['output_dir'] = str(RUNS / f'stratified_residual_seed{seed}')
                params['inf']['model_name'] = str(Path(params['exp']['output_dir']) / params['train']['save_model_name'])
                config = OUTPUT / 'configs' / f'residual_seed{seed}.yml'
                if config.exists():
                    assert yaml.safe_load(config.read_text()) == params
                else:
                    config.write_text(yaml.safe_dump(params))
                specs[str(seed)] = make_residual_spec(params, records, str(TRUTH), split, original)
    protocol = dict(format_version=1, population=split, truth_path=str(TRUTH), truth_sha256=file_sha256(TRUTH),
                    forecast_protocol_sha256=file_sha256(FORECAST / 'PROTOCOL.json'),
                    forecast_index_sha256=file_sha256(Path(forecasts['forecast_output']) / 'index.json'),
                    references=references, expected_run_specs=specs,
                    selection='User authorized VLM follow-up after residual genotype AUROC improved but the joint forecast-fidelity gate failed',
                    design='Reuse six verified baseline/current runs; fit residual input only, all four folds and three seeds, fixed final 20 epochs',
                    changes='Only the forecast source and its verification route differ from the matching longitudinal VLM run',
                    reporting='All seeds, folds, proxy horizons, within-fold pair-weighted AUROC, fixed-zero-logit accuracy, subject-mean proxy MAE',
                    limitations='Development comparison on repeatedly inspected 32 mice; genotype/diet/acquisition confounding; unvalidated PET proxy; no independent-group inference')
    path = OUTPUT / 'PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen residual VLM protocol changed'
    else:
        atomic_json(path, protocol)
        snapshot = OUTPUT / 'source_snapshot'; snapshot.mkdir()
        for name, expected in specs['0']['source_sha256'].items():
            source = REPO / name if name.startswith('scripts/') else REPO / 'vlm' / name
            assert file_sha256(source) == expected
            target = snapshot / ('/' .join(['vlm', name]) if not name.startswith('scripts/') else name)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
    atomic_json(OUTPUT / 'preflight.json', dict(status='passed', matched_reference_runs=6,
                residual_forecast_folds=4, exact_dataset_token_rows=128, new_vlm_fits=12,
                experiment_protocol_sha256=object_sha256(protocol)))
    print('Preflight passed: all six comparators verified; 128 four-token dataset inputs match saved residual forecasts; three run specs matched.', flush=True)
    return records, split, protocol


def verify_residual_run(directory, records, split, expected):
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    if manifest['spec'] != expected or manifest['run_id'] != object_sha256(expected):
        raise ValueError('Residual VLM run does not match the frozen protocol')
    parts = collect_parts(directory, manifest['run_id'], records, split)
    result = score_folds(parts, records, split)
    if result != json.loads((directory / 'stratified_results.json').read_text()):
        raise ValueError('Residual aggregate scores do not reproduce')
    if [r for f in split['folds'] for r in parts[f['key']]] != json.loads((directory / 'vqa_stratified.json').read_text()):
        raise ValueError('Residual aggregate predictions differ from verified folds')
    return result


def evaluate(records, split, protocol):
    pairs = []; independent = []
    for seed in split['seeds']:
        entry = dict(seed=seed)
        for arm in ('base', 'long', 'residual'):
            directory = RUNS / f'stratified_{arm}_seed{seed}'
            if arm == 'residual':
                result = verify_residual_run(directory, records, split, protocol['expected_run_specs'][str(seed)])
            else:
                result, _ = verified_run(directory, records, str(TRUTH), split)
            independent.append(check_run(directory, records, split))
            entry[arm] = result
            entry[arm + '_run'] = str(directory)
        for comparator in ('base', 'long'):
            entry['residual_minus_' + comparator] = dict(
                genotype_auc=entry['residual']['genotype_auc_within_fold_pair_weighted'] - entry[comparator]['genotype_auc_within_fold_pair_weighted'],
                accuracy=entry['residual']['genotype_accuracy'] - entry[comparator]['genotype_accuracy'],
                subject_mae=entry['residual']['subject_mae'] - entry[comparator]['subject_mae'])
        pairs.append(entry)
    report = dict(status='complete', protocol_sha256=object_sha256(protocol), per_seed=pairs,
                  means={arm: {key: float(np.mean([p[arm][key] for p in pairs])) for key in
                      ('genotype_auc_within_fold_pair_weighted', 'genotype_accuracy', 'subject_mae', 'train_mean_subject_mae')}
                      for arm in ('base', 'long', 'residual')})
    atomic_json(OUTPUT / 'paired_results.json', report)
    atomic_json(OUTPUT / 'independent_metrics.json', dict(status='passed', checks=independent,
                method='Independent standard-library scoring via check_stratified_metrics.check_run for all nine runs'))
    render(report)
    paths = [OUTPUT / name for name in ('PROTOCOL.json', 'paired_results.json', 'RESULTS.md', 'independent_metrics.json')]
    atomic_json(OUTPUT / 'completion.json', dict(status='complete', new_vlm_fits=12, total_compared_vlm_fits=36,
                independent_runs_verified=9, files_sha256={str(p): file_sha256(p) for p in paths}))
    print(json.dumps(report['means'], indent=2), flush=True)


def render(report):
    lines = ['# Residual-embedding VLM comparison', '',
             'Twelve new residual-input fits use the same VLM, four subject folds, three seeds, supervision, '
             'and fixed 20-epoch schedule as the six verified baseline/current comparator runs. '
             'Only the forecast tokens and their verification route change. '
             'The user authorized this follow-up after the residual embedding probe improved genotype ranking '
             'but failed the joint forecast-fidelity gate. This is a development comparison.', '',
             '| Seed | Baseline AUROC | Current longitudinal AUROC | Residual AUROC | Baseline accuracy | Current accuracy | Residual accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for p in report['per_seed']:
        lines.append('| ' + str(p['seed']) + ' | ' + ' | '.join(f"{p[a][k]:.4f}" for k in
            ('genotype_auc_within_fold_pair_weighted', 'genotype_accuracy') for a in ('base', 'long', 'residual')) + ' |')
    lines += ['', 'AUROC averages only within-model KO-WT comparisons, weighted by 62 pairs per seed. '
              'Classification uses a fixed zero-logit threshold; the training-majority comparator scores 56.25%.', '',
              '| Seed | Baseline proxy MAE | Current longitudinal proxy MAE | Residual proxy MAE | Training-mean MAE |',
              '|---|---:|---:|---:|---:|']
    for p in report['per_seed']:
        lines.append(f"| {p['seed']} | {p['base']['subject_mae']:.4f} | {p['long']['subject_mae']:.4f} | {p['residual']['subject_mae']:.4f} | {p['base']['train_mean_subject_mae']:.4f} |")
    lines += ['', '## Every fold', '', '| Seed | Fold | Baseline AUROC | Current AUROC | Residual AUROC |', '|---|---|---:|---:|---:|']
    for p in report['per_seed']:
        for i, fold in enumerate(p['base']['folds']):
            lines.append(f"| {p['seed']} | {fold['key']} | " + ' | '.join(f"{p[a]['folds'][i]['genotype_auc']:.4f}" for a in ('base', 'long', 'residual')) + ' |')
    lines += ['', '## Every proxy horizon', '', '| Seed | Week | n | Baseline MAE | Current MAE | Residual MAE | Training-mean MAE |', '|---|---|---:|---:|---:|---:|---:|']
    for p in report['per_seed']:
        for week in ('15', '18', '20'):
            lines.append(f"| {p['seed']} | {week} | {p['base']['horizons'][week]['n']} | " +
                ' | '.join(f"{p[a]['horizons'][week]['mae']:.4f}" for a in ('base', 'long', 'residual')) +
                f" | {p['base']['horizons'][week]['train_mean']['mae']:.4f} |")
    lines += ['', 'All nine runs passed artifact verification and independent arithmetic checks. '
              'Seeds and within-fold pairs are not independent studies. These 32 mice and folds have been inspected repeatedly; '
              'acquisition and diet remain confounded with genotype. The PET outcome is an unvalidated distributional proxy. '
              'No significance, calibration, independent-acquisition, or validated disease-progression claim follows automatically.', '',
              '[Full results](paired_results.json), [independent scoring](independent_metrics.json), [frozen protocol](PROTOCOL.json).', '']
    (OUTPUT / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    records, split, protocol = prepare()
    if not args.prepare_only:
        evaluate(records, split, protocol)
