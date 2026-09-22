"""Run the two fixed genotype variants with matched residual inputs and folds."""
import argparse
import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from stratified_vlm_protocol import collect_parts, score_folds
from residual_vlm_protocol import make_residual_spec, verify_residual_inputs
from utils.research_io import atomic_json, file_sha256, object_sha256
from utils.run_contract import freeze_run, completed_fold, mark_fold_complete
from vlm_genotype_variants import VariantHelper, VARIANTS
from data.validated_metrics import score_predictions


SOURCES = ('vlm_genotype_variants.py', 'run_genotype_variants.py', 'evaluate_genotype_variants.py')


def make_variant_spec(params, records, truth, split, variant, parent_spec, original_spec):
    if variant not in VARIANTS:
        raise ValueError('Unknown genotype variant')
    spec = make_residual_spec(params, records, truth, split, original_spec)
    if spec != parent_spec:
        raise ValueError('Variant changed the residual VLM training or input contract')
    spec['genotype_variant'] = variant
    for name in SOURCES:
        spec['source_sha256']['scripts/' + name] = file_sha256(Path(__file__).with_name(name))
    return spec


def run_fold(fold, records, directory, base_params, split, run_id, variant):
    output = directory / fold['key']; output.mkdir(parents=True, exist_ok=True)
    truth = [r for r in records if r['pid'] in fold['test_subjects']]
    training = [r for r in records if r['pid'] in fold['train_subjects']]
    if completed_fold(output, run_id, truth) is not None:
        return
    params = copy.deepcopy(base_params)
    params['exp']['output_dir'] = str(output)
    split_dir = output / 'splits'; split_dir.mkdir(exist_ok=True)
    atomic_json(split_dir / 'train.json', training); atomic_json(split_dir / 'val.json', truth)
    params['data'].update(data_path=str(split_dir / 'train.json'), inf_data_path=str(split_dir / 'val.json'),
                          eval_data_path=None)
    if params['data']['img_tokens'] == 4:
        tokens = Path(params['data']['nested_forecast_root']) / fold['key']
        manifest = verify_residual_inputs(params['data']['nested_forecast_root'], fold, split, records)
        for sid, path in {r['pid']:r['embedding_path_ts0'] for r in records}.items():
            vector = load_file(path)['embeddings'].contiguous().numpy()
            if hashlib.sha256(vector.tobytes()).hexdigest() != manifest['baseline_sha256'][sid]:
                raise ValueError('Forecast baseline differs from VLM baseline')
        params['data']['predicted_emb_dir'] = str(tokens)
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
    if checkpoints:
        valid = [p for p in checkpoints if all((p / f).exists() for f in
                 ('vlm_config.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'))]
        if not valid:
            raise ValueError('No complete restart checkpoint for partial fold')
        params['train']['resume_from_checkpoint'] = str(valid[-1])
    params['inf'].update(model_name=str(output / params['train']['save_model_name']),
                          train_gt_file=str(split_dir / 'train.json'), save_file='vqa.json', results_file='val.json')
    config = split_dir / 'params.yml'; config.write_text(yaml.safe_dump(params))
    print(f"Starting {fold['key']}: {len(training)//3} train / {len(truth)//3} test mice", flush=True)
    trainer = VariantHelper(str(config), variant)
    trainer.setup(); trainer.run(); trainer.evaluate()
    predictions = json.loads((output / 'vqa.json').read_text())
    score_predictions(predictions, truth, training)
    mark_fold_complete(output, run_id, params['train']['save_model_name'])
    # Include exact memberships and effective config in the completion contract.
    state = json.loads((output / 'complete.json').read_text())
    for path in (split_dir / 'train.json', split_dir / 'val.json', config, output / 'exp.yml', output / 'variant_training_audit.json'):
        state['files'][str(path.relative_to(output))] = file_sha256(path)
    atomic_json(output / 'complete.json', state)
    del trainer; gc.collect(); torch.cuda.empty_cache()
    for path in checkpoints + list(output.glob('checkpoint-*')):
        if path.exists():
            shutil.rmtree(path)
    print(f"Completed {fold['key']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--yaml', required=True); p.add_argument('--split', required=True)
    p.add_argument('--all-data', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--original-run', required=True)
    p.add_argument('--parent-run', required=True)
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--end-fold', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.end_fold <= 4:
        raise ValueError('Invalid fold limit')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    records = json.loads(Path(args.all_data).read_text()); split = json.loads(Path(args.split).read_text())
    params = yaml.safe_load(Path(args.yaml).read_text()); directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    with open(directory / '.run.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_spec = json.loads((Path(args.original_run) / 'run_manifest.json').read_text())['spec']
        parent_spec = json.loads((Path(args.parent_run) / 'run_manifest.json').read_text())['spec']
        run_id = freeze_run(directory, make_variant_spec(params, records, args.all_data, split, args.variant, parent_spec, original_spec))
        for fold in split['folds'][:args.end_fold]:
            run_fold(fold, records, directory, params, split, run_id, args.variant)
            parts = collect_parts(directory, run_id, records, split)
            complete = len(parts) == len(split['folds'])
            atomic_json(directory / ('vqa_stratified.json' if complete else 'vqa_stratified_partial.json'),
                        [r for f in split['folds'] if f['key'] in parts for r in parts[f['key']]])
            atomic_json(directory / 'run_status.json', dict(status='complete' if complete else 'incomplete',
                        run_id=run_id, complete_folds=sorted(parts), expected_folds=[f['key'] for f in split['folds']]))
        if complete:
            result = score_folds(parts, records, split)
            atomic_json(directory / 'stratified_results.json', result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('folds', 'horizons', 'per_subject_mae',
                                                                      'per_subject_train_mean_mae')}), flush=True)


if __name__ == '__main__':
    main()
