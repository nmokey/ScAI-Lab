"""Recover retained adapter/head tensors into a NEW directory, without loading an LLM.

Accepts a legacy lora_adapter directory's parent or a Trainer checkpoint directory.
Requires the original experiment YAML; never changes pooling to relabel old training.
Missing normalization is recorded as incomplete, never silently replaced with defaults.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from utils.checkpoint_utils import canonicalize_legacy_adapter, read_adapter


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def repair(source, output, config_path):
    source, output, config_path = map(Path, (source, output, config_path))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    cfg = yaml.safe_load(config_path.read_text()); train, data = cfg['train'], cfg['data']
    full = source / 'model.safetensors'
    sources = [config_path]
    if full.exists():
        sources.append(full)
        with safe_open(full, framework='pt', device='cpu') as f:
            adapter = {k.removeprefix('language_model.'): f.get_tensor(k)
                       for k in f.keys() if k.startswith('language_model.') and 'lora_' in k}
            other = {k: f.get_tensor(k) for k in f.keys() if not k.startswith('language_model.')}
    else:
        directory = source / 'lora_adapter'
        adapter = read_adapter(directory)
        other_path = source / 'other_weights.bin'
        other = torch.load(other_path, map_location='cpu', weights_only=True)
        sources += list(directory.glob('adapter_*')) + [other_path]
    adapter = canonicalize_legacy_adapter(adapter)
    if not adapter or any('.default.' in k for k in adapter):
        raise ValueError('Unsupported or empty adapter state')
    for k, v in {**adapter, **other}.items():
        if v.is_meta or not torch.isfinite(v).all():
            raise ValueError(f'Invalid source tensor: {k}')
    adapter_config = LoraConfig(r=train['r'], lora_alpha=train['lora_alpha'],
                               target_modules=train['target_modules'], lora_dropout=train['lora_dropout'],
                               bias=train['bias'], task_type=train['task_type'],
                               base_model_name_or_path=train['llm_model_name'])
    # This recovery utility handles the exact r/qv-or-all/bias-none adapters used here.
    # Strict model reload is still required to validate complete backbone compatibility.
    if train['bias'] != 'none':
        raise ValueError('Recovery of bias-training configurations is not supported')
    expected_other = {'language_projection.weight','language_projection.bias'}
    if train['num_proj_layers'] != 1:
        raise ValueError('Recovery currently requires the historical one-layer projection')
    if train['add_multitask']:
        expected_other |= {'genotype_head.weight', 'genotype_head.bias', 'tbr_regression_head.0.weight',
                           'tbr_regression_head.0.bias', 'tbr_regression_head.2.weight','tbr_regression_head.2.bias'}
    missing = sorted((expected_other | {'tbr_mean','tbr_std'}) - set(other))
    if set(other) - (expected_other | {'tbr_mean','tbr_std'}):
        raise ValueError('Unexpected non-LLM state in source')
    tokenizer = AutoTokenizer.from_pretrained(data['tokenizer_name'], local_files_only=True)
    tokenizer.add_tokens(['<image>'])
    metadata = dict(format_version=2, img_token_id=tokenizer.convert_tokens_to_ids('<image>'),
                    img_tokens=data['img_tokens'], num_proj_layers=train['num_proj_layers'],
                    create_projection_layer=True, add_multitask=train['add_multitask'],
                    add_multitask_unknown=train.get('add_multitask_unknown',False),
                    multitask_wt=train['multitask_wt'], pool_at=train.get('pool_at','answer_eos'),
                    hidden_size=other['language_projection.weight'].shape[0],
                    base_model_name_or_path=train['llm_model_name'])
    report = dict(source=str(source.resolve()), original_config=str(config_path.resolve()),
                  source_sha256={str(p):digest(p) for p in sources},
                  adapter_tensor_count=len(adapter),
                  lora_B_nonzero_count=sum(bool(v.count_nonzero()) for k,v in adapter.items() if '.lora_B.' in k),
                  missing_state=missing, status='incomplete' if missing else 'recovered_requires_model_reload_check',
                  original_pool_at=metadata['pool_at'],
                  note='Weights preserved from this training step; no retraining, completed-fold or valid-metrics claim.')
    output.mkdir(parents=True)
    adapter_dir=output/'lora_adapter'; adapter_dir.mkdir()
    adapter_config.save_pretrained(adapter_dir)
    save_file({k:v.contiguous() for k,v in adapter.items()}, str(adapter_dir/'adapter_model.safetensors'))
    torch.save(other, output/'other_weights.bin')
    # Incomplete recovery cannot be mistaken for a loadable current checkpoint.
    if not missing:
        (output/'vlm_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    shutil.copyfile(config_path, output/'original_exp.yml')
    (output/'recovery.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    print(json.dumps(repair(args.source,args.output,args.config),indent=2))

if __name__=='__main__': main()
