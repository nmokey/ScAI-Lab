"""Freeze and evaluate two fixed genotype VLM variants without test-set selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from run_genotype_variants import SOURCES, make_variant_spec
from vlm_genotype_variants import VARIANTS
from evaluate_residual_vlm import verify_residual_run
from aggregate_stratified_vlm import verified_run
from residual_vlm_protocol import verify_residual_inputs
from stratified_vlm_protocol import collect_parts, score_folds
from check_stratified_metrics import check_run
from utils.research_io import atomic_json, file_sha256, object_sha256

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'docs/audit_2026-09-14/genotype_vlm_variants'
PARENT = REPO / 'docs/audit_2026-09-14/residual_vlm'
RUNS = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/runs')
TRUTH = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')


def prepare():
    checks = json.loads((OUTPUT / 'runtime_checks.json').read_text())
    if checks['status'] != 'passed' or checks['model_source_sha256'] != file_sha256(REPO / 'scripts/vlm_genotype_variants.py'):
        raise ValueError('Current variant implementation has not passed the actual-backbone checks')
    parent_protocol = json.loads((PARENT / 'PROTOCOL.json').read_text())
    records = json.loads(TRUTH.read_text()); split = parent_protocol['population']
    specs = {}; references = {}
    (OUTPUT / 'configs').mkdir(exist_ok=True)
    for seed in split['seeds']:
        parent_run = RUNS / f'stratified_residual_seed{seed}'
        parent_spec = json.loads((parent_run / 'run_manifest.json').read_text())['spec']
        verify_residual_run(parent_run, records, split, parent_protocol['expected_run_specs'][str(seed)])
        for arm in ('base', 'long'):
            _, original_spec = verified_run(RUNS / f'stratified_{arm}_seed{seed}', records, str(TRUTH), split)
        references[str(seed)] = dict(residual_manifest_sha256=file_sha256(parent_run / 'run_manifest.json'),
            residual_results_sha256=file_sha256(parent_run / 'stratified_results.json'),
            base_results_sha256=file_sha256(RUNS / f'stratified_base_seed{seed}/stratified_results.json'))
        for variant in VARIANTS:
            params = yaml.safe_load((PARENT / 'configs' / f'residual_seed{seed}.yml').read_text())
            params['exp']['output_dir'] = str(RUNS / f'stratified_{variant}_seed{seed}')
            params['inf']['model_name'] = str(Path(params['exp']['output_dir']) / params['train']['save_model_name'])
            spec = make_variant_spec(params, records, str(TRUTH), split, variant, parent_spec, original_spec)
            specs[f'{variant}_seed{seed}'] = spec
            path = OUTPUT / 'configs' / f'{variant}_seed{seed}.yml'
            if path.exists():
                assert yaml.safe_load(path.read_text()) == params
            else:
                path.write_text(yaml.safe_dump(params))
    root = specs['direct_visual_seed0']['config']['data']['nested_forecast_root']
    for fold in split['folds']:
        verify_residual_inputs(root, fold, split, records)
    protocol = dict(format_version=1, population=split, truth_sha256=file_sha256(TRUTH),
        parent_protocol_sha256=file_sha256(PARENT / 'PROTOCOL.json'), references=references,
        expected_run_specs=specs, runtime_checks_sha256=file_sha256(OUTPUT / 'runtime_checks.json'),
        direct_visual='Existing language-state genotype logit plus zero-initialized bias-free linear score from all four standardized raw image tokens; original combined loss',
        genotype_only='Original genotype head and weighted BCE; language and proxy losses removed; all three original question records retained',
        normalization='Per-coordinate StandardScaler fitted on 24 unique training mice only; mean/scale saved as FP32 buffers',
        controls='Completed residual VLM, baseline VLM, and fixed direct-embedding logistic probe; visual/language components of the trained hybrid are descriptive ablations',
        selection='Two variants fixed in advance; all three seeds and four folds; final 20 epochs/180 updates only; no threshold or model selection',
        primary_focus='Genotype within-fold pair-weighted AUROC and fixed-zero-logit accuracy; balanced accuracy and class recall also reported',
        proxy_policy='Report supervised proxy metrics for direct_visual; genotype_only proxy outputs are unsupervised and not performance estimates',
        limitations='Repeatedly inspected development folds; acquisition and diet confounding; no independent-group inference or guarantee of 70% accuracy')
    path = OUTPUT / 'PROTOCOL.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen genotype-variant protocol changed'
    else:
        atomic_json(path, protocol)
        snapshot = OUTPUT / 'source_snapshot'; snapshot.mkdir()
        for name, digest in specs['direct_visual_seed0']['source_sha256'].items():
            source = REPO / name if name.startswith('scripts/') else REPO / 'vlm' / name
            assert file_sha256(source) == digest
            target = snapshot / (name if name.startswith('scripts/') else 'vlm/' + name)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
    atomic_json(OUTPUT / 'preflight.json', dict(status='passed', new_vlm_fits=24,
        runtime_variants_verified=2, dataset_input_rows_verified=128, protocol_sha256=object_sha256(protocol)))
    print('Genotype-variant preflight passed: matched inputs/settings, actual-backbone checks, 24 prescribed fits.', flush=True)
    return records, split, protocol


def class_metrics(rows, key='genotype_logit'):
    geno = [r for r in rows if r['content_type'] == 'genotype']
    assert len(geno) == 32 and len({r['pid'] for r in geno}) == 32
    tp = sum(r[key] > 0 and r['genotype_label'] == 1 for r in geno)
    tn = sum(r[key] <= 0 and r['genotype_label'] == 0 for r in geno)
    assert sum(r['genotype_label'] for r in geno) == 14
    return dict(correct=tp+tn, accuracy=(tp+tn)/32, ko_recall=tp/14, wt_recall=tn/18,
                balanced_accuracy=.5*(tp/14+tn/18))


def component_auc(parts, key):
    wins = 0.; pairs = 0
    for rows in parts.values():
        geno = [r for r in rows if r['content_type'] == 'genotype']
        pos = [r[key] for r in geno if r['genotype_label'] == 1]
        neg = [r[key] for r in geno if r['genotype_label'] == 0]
        wins += sum((p > n) + .5*(p == n) for p in pos for n in neg)
        pairs += len(pos)*len(neg)
    assert pairs == 62
    return wins/pairs


def evaluate(records, split, protocol):
    parents = json.loads((PARENT / 'paired_results.json').read_text())
    entries = []; independent = []; initials = {}
    for seed in split['seeds']:
        parent = next(p for p in parents['per_seed'] if p['seed'] == seed)
        entry = dict(seed=seed, base=parent['base'], residual=parent['residual'])
        for arm in ('base', 'residual'):
            directory = RUNS / f'stratified_{arm}_seed{seed}'
            entry[arm] = dict(entry[arm], classification=class_metrics(json.loads((directory/'vqa_stratified.json').read_text())))
            independent.append(check_run(directory, records, split))
        for variant in VARIANTS:
            directory = RUNS / f'stratified_{variant}_seed{seed}'
            manifest = json.loads((directory / 'run_manifest.json').read_text())
            expected = protocol['expected_run_specs'][f'{variant}_seed{seed}']
            if manifest['spec'] != expected or manifest['run_id'] != object_sha256(expected):
                raise ValueError('Variant run identity mismatch')
            parts = collect_parts(directory, manifest['run_id'], records, split)
            result = score_folds(parts, records, split)
            if result != json.loads((directory / 'stratified_results.json').read_text()):
                raise ValueError('Variant scoring does not reproduce')
            rows = [r for f in split['folds'] for r in parts[f['key']]]
            if rows != json.loads((directory / 'vqa_stratified.json').read_text()):
                raise ValueError('Variant predictions do not match complete folds')
            assert all(r['genotype_variant'] == variant and r['checkpoint_format_version'] == 3 for r in rows)
            for fold in split['folds']:
                audit = json.loads((directory / fold['key'] / 'variant_training_audit.json').read_text())
                assert audit['reload_exact'] and audit['updates'] == 180 and audit['epochs'] == 20
                key = (seed, fold['key'])
                if key in initials:
                    assert initials[key] == audit['initial_common_parameter_sha256']
                initials[key] = audit['initial_common_parameter_sha256']
                if variant == 'direct_visual':
                    assert audit['visual_normalization_subjects'] == sorted(fold['train_subjects'])
                    weights = torch.load(directory / fold['key'] / 'mouse_vlm_mdl/other_weights.bin', map_location='cpu', weights_only=True)
                    reference = np.load(REPO / 'docs/audit_2026-09-14/residual_rollout_experiment' /
                                        f"residual_combined_{fold['key']}.npz", allow_pickle=False)
                    np.testing.assert_array_equal(weights['visual_mean'].numpy(), reference['mean'].astype(np.float32))
                    np.testing.assert_array_equal(weights['visual_std'].numpy(), reference['scale'].astype(np.float32))
            independent.append(check_run(directory, records, split))
            result['classification'] = class_metrics(rows)
            result['proxy_evaluation_applicable'] = variant != 'genotype_only'
            if variant == 'direct_visual':
                for r in rows:
                    assert np.float32(r['genotype_logit']) == np.float32(r['language_genotype_logit']) + np.float32(r['visual_genotype_logit'])
                result['component_diagnostics'] = {k: dict(auc=component_auc(parts, k), **class_metrics(rows, k))
                    for k in ('language_genotype_logit', 'visual_genotype_logit')}
            entry[variant] = result
        entries.append(entry)
    probe_path = REPO / 'docs/audit_2026-09-14/residual_rollout_experiment/results.json'
    probe = json.loads(probe_path.read_text())['arms']['residual_combined']['summary']
    report = dict(status='complete', protocol_sha256=object_sha256(protocol), per_seed=entries,
        direct_embedding_control=probe, direct_embedding_source_sha256=file_sha256(probe_path),
        means={arm: dict(auc=float(np.mean([p[arm]['genotype_auc_within_fold_pair_weighted'] for p in entries])),
                        accuracy=float(np.mean([p[arm]['genotype_accuracy'] for p in entries])),
                        balanced_accuracy=float(np.mean([p[arm]['classification']['balanced_accuracy'] for p in entries])),
                        proxy_mae=None if arm == 'genotype_only' else float(np.mean([p[arm]['subject_mae'] for p in entries])))
               for arm in ('base', 'residual', *VARIANTS)})
    atomic_json(OUTPUT/'paired_results.json', report)
    atomic_json(OUTPUT/'independent_metrics.json', dict(status='passed', checks=independent,
        note='Genotype-only unsupervised proxy arithmetic is verified for artifact consistency only, not interpreted as performance'))
    render(report)
    files = [OUTPUT/name for name in ('PROTOCOL.json', 'paired_results.json', 'RESULTS.md', 'independent_metrics.json', 'runtime_checks.json')]
    atomic_json(OUTPUT/'completion.json', dict(status='complete', new_vlm_fits=24,
        verified_variant_checkpoint_reloads=24, independent_runs_verified=12,
        files_sha256={str(p):file_sha256(p) for p in files}))
    print(json.dumps(report['means'], indent=2), flush=True)


def render(report):
    lines = ['# Fixed genotype VLM variants', '',
        'Both prescribed variants completed all four folds and three seeds. Settings and thresholds were fixed before fitting. '
        'All new checkpoints passed exact inference reload and independent metric checks.', '',
        '| Seed | Input/model | AUROC | Correct / 32 | Balanced accuracy | KO recall | WT recall |',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in report['per_seed']:
        for arm in ('base', 'residual', *VARIANTS):
            r = p[arm]; c = r['classification']
            lines.append(f"| {p['seed']} | {arm} | {r['genotype_auc_within_fold_pair_weighted']:.4f} | {c['correct']}/32 | {c['balanced_accuracy']:.4f} | {c['ko_recall']:.4f} | {c['wt_recall']:.4f} |")
    control = report['direct_embedding_control']
    lines += ['', f"The fixed residual-embedding logistic classifier alone scores AUROC {control['auc']:.4f}, "
        f"accuracy {control['accuracy']:.4f} ({control['correct']}/32). It is a simpler comparator, not a matched-optimizer ablation. "
        'Always-WT accuracy is 18/32 (56.25%). AUROC counts 62 within-model KO-WT pairs; different fitted models are never ranked together.', '',
        '| Seed | Residual proxy MAE | Direct-visual proxy MAE | Training-mean MAE |', '|---|---:|---:|---:|']
    for p in report['per_seed']:
        lines.append(f"| {p['seed']} | {p['residual']['subject_mae']:.4f} | {p['direct_visual']['subject_mae']:.4f} | {p['residual']['train_mean_subject_mae']:.4f} |")
    lines += ['', '**Genotype-only proxy outputs are not supervised and are not performance estimates.** Their raw arithmetic remains in the machine-readable audit artifacts only.', '',
        '## Direct-visual component diagnostics', '',
        '| Seed | Combined AUROC | Language-state component AUROC | Direct-visual component AUROC |', '|---|---:|---:|---:|']
    for p in report['per_seed']:
        r = p['direct_visual']; c = r['component_diagnostics']
        lines.append(f"| {p['seed']} | {r['genotype_auc_within_fold_pair_weighted']:.4f} | {c['language_genotype_logit']['auc']:.4f} | {c['visual_genotype_logit']['auc']:.4f} |")
    lines += ['', 'Components come from the jointly trained model; removing a component at inference is diagnostic and '
        'does not replace training a standalone model. All folds, supervised proxy horizons, and raw scores are retained in '
        '[full results](paired_results.json). No model or threshold is selected for a replacement headline result.', '',
        'These are repeatedly inspected development folds from 32 mice with acquisition/diet confounding. Seeds are not independent studies, '
        'and improved genotype discrimination does not establish disease-progression prediction. '
        '[Protocol](PROTOCOL.json), [independent scoring](independent_metrics.json), [actual-backbone checks](runtime_checks.json).', '']
    (OUTPUT/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); records, split, protocol = prepare()
    if not args.prepare_only: evaluate(records, split, protocol)
