"""Finish saved controls after fixing a probe-precision mismatch; no training.

The original controls completed all updates and saved final models, but their
reload assertion compared implicit Trainer autocast with ordinary inference.
Preserve the original source/reports; verify saved tensors and reproduce the
recorded training-curve endpoint at explicitly matching precision.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file

from run_real_mouse_learning_control import measure,component,state_digest
from model.viz_emb_trainer import VizEmbTrainer
from model.vision_language_model import VisionLanguageModel
from utils.research_io import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    args=p.parse_args();root=Path(args.directory)
    spec=importlib.util.spec_from_file_location('original_control_probe',root/'run_real_mouse_learning_control_source.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    transformers.set_seed(0)
    helper=VizEmbTrainer(str(root/'combined.yml'));helper._setup_tokenizer()
    dataset=helper.get_train_data()['train'];model,_=helper.load_train_model();model.to('cuda')
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    initial_hash=state_digest(initial);checks=[]
    for objective in ('combined','genotype_only'):
        folder=root/objective;report=json.loads((folder/'report.json').read_text())
        if (folder/'report_before_precision_repair.json').exists():
            raise ValueError('Preserve the existing repair; do not overwrite it')
        atomic_json(folder/'report_before_precision_repair.json',report)
        assert report['initial_trainable_sha256']==initial_hash
        assert report['learning_curve'][-1]['step']==300
        state=json.loads((folder/'checkpoint-300/trainer_state.json').read_text())
        assert state['global_step']==300 and state['epoch']==100.
        checkpoint=folder/'final_model'
        # Independently compare the final model to the last full Trainer save.
        for filename,loader in [('other_weights.bin',lambda p:torch.load(p,map_location='cpu',weights_only=True)),
                               ('lora_adapter/adapter_model.safetensors',lambda p:load_file(str(p)))]:
            a=loader(checkpoint/filename);b=loader(folder/'checkpoint-300'/filename)
            assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        loaded=VisionLanguageModel.from_pretrained(checkpoint,None,model.language_model,
                                                   helper.img_token_id,tokenizer=helper.tokenizer)
        loaded.eval()
        raw=original.measure(loaded,dataset);fixed=measure(loaded,dataset)
        recorded=report['learning_curve'][-1]
        error=max(abs(a['logit']-b['logit']) for a,b in zip(recorded['rows'],fixed['rows']))
        precision_delta=max(abs(a['logit']-b['logit']) for a,b in zip(recorded['rows'],raw['rows']))
        assert error==0,(objective,error)
        sums={}
        for name,param in loaded.named_parameters():
            if name in initial:
                group=component(name)
                sums[group]=sums.get(group,0.)+float((param.detach().cpu().float()-initial[name].float()).square().sum())
        norms={k:math.sqrt(v) for k,v in sums.items()}
        assert all(norms[k]>0 for k in ('language_projection','genotype_head','lora_A','lora_B'))
        assert (norms['tbr_regression_head']==0)==(objective=='genotype_only')
        report.update(status='complete',updates=300,epochs=100.,final=fixed,
                      parameter_change_norms=norms,reload_max_logit_error=error,
                      measurement_precision='BF16 model forward and FP32 genotype readout; explicit across Trainer and reloaded model',
                      ordinary_inference_final=raw,initial_unmatched_precision_max_logit_error=precision_delta,
                      final_matches_trainer_checkpoint_tensors=True,
                      capacity_criterion_met=fixed['by_question']['genotype']['accuracy']==1. and fixed['by_question']['genotype']['bce']<=.1,
                      measurement_repair=dict(no_training_performed=True,
                          original_training_source_sha256=file_sha256(root/'run_real_mouse_learning_control_source.py'),
                          repaired_probe_source_sha256=file_sha256(Path(__file__).with_name('run_real_mouse_learning_control.py')),
                          repair_source_sha256=file_sha256(__file__)))
        atomic_json(folder/'report.json',report)
        checks.append(dict(objective=objective,reload_max_logit_error=error,
                           initial_precision_mismatch=precision_delta,ordinary_inference=raw['by_question']['genotype']))
        model=loaded
    atomic_json(root/'precision_repair.json',dict(status='passed',training_rerun=False,checks=checks))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
