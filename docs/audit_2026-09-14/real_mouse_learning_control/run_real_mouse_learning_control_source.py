"""Training-only capacity control on eight existing mice; never a research score.

Two objectives share the production model, data, initialization, optimizer,
batching, head boundary and 300-update budget. Genotype-only removes language
and proxy losses through Trainer.compute_loss; the production model is unedited.
"""
import argparse
import copy
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import DefaultDataCollator, TrainerCallback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from model.viz_emb_trainer import VizEmbTrainer, VLMTrainer
from model.vision_language_model import VisionLanguageModel
from utils.research_io import atomic_json, file_sha256
from utils.target_contract import fold_target_statistics


class ControlTrainer(VLMTrainer):
    def __init__(self, *args, objective, **kwargs):
        self.control_objective = objective
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.control_objective == 'genotype_only':
            inputs = dict(inputs)
            inputs.pop('labels', None)
            inputs.pop('tbr_targets', None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)


def component(name):
    return next((s for s in ('language_projection','genotype_head','tbr_regression_head','lora_A','lora_B') if s in name),None)


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.float().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def measure(model, dataset):
    was_training=model.training; model.eval();rows=[]
    collate=DefaultDataCollator()
    with torch.random.fork_rng(devices=[0]):
        for start in range(0,len(dataset),2):
            batch={k:v.cuda() for k,v in collate([dataset[i] for i in range(start,min(start+2,len(dataset)))]).items()}
            # Match ordinary saved-model inference precision. Answers and losses
            # are absent from this forward; the explicit question boundary remains.
            inputs={k:batch[k] for k in ('input_ids','attention_mask','image_features','question_end_index')}
            out=model(**inputs)
            hidden=model._eos_hidden_state(batch['input_ids'],out.hidden_states,
                                           batch['question_end_index'],batch['attention_mask'])
            logits=model.genotype_head(hidden).squeeze(-1).float().cpu().tolist()
            for offset,value in enumerate(logits):
                record=dataset.data[start+offset]
                rows.append(dict(pid=record['pid'],qid=record['qid'],kind=record['content_type'],
                                 label=record['answer_vqa_numeric']['genotype'],logit=value))
    model.train(was_training)
    groups={}
    for kind in ('genotype','tbr','combined','all'):
        rr=[r for r in rows if kind=='all' or r['kind']==kind]
        y=np.array([r['label'] for r in rr]);s=np.array([r['logit'] for r in rr])
        positive=s[y==1];negative=s[y==0]
        auc=float(np.mean([float(a>b)+.5*float(a==b) for a in positive for b in negative]))
        groups[kind]=dict(n=len(rr),accuracy=float(np.mean((s>0)==y)),auc=auc,
                          bce=float(np.mean(np.logaddexp(0,s)-y*s)),
                          minimum_true_class_probability=float(np.min(1/(1+np.exp(-(2*y-1)*s)))))
    return dict(by_question=groups,rows=rows)


class Progress(TrainerCallback):
    def __init__(self,dataset,report,path):
        self.dataset,self.report,self.path=dataset,report,path

    def capture(self,model,step):
        values=measure(model,self.dataset);values['step']=step
        self.report['learning_curve'].append(values);atomic_json(self.path,self.report)
        print(json.dumps(dict(control_step=step,genotype=values['by_question']['genotype'],
                              all_questions=values['by_question']['all'])),flush=True)

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.capture(model,0)

    def on_step_end(self,args,state,control,model=None,**kwargs):
        if state.global_step in (30,60,120,180,240,300):
            self.capture(model,state.global_step)

    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        if state.global_step not in (0,59,179,299):return
        squares={}
        for name,p in model.named_parameters():
            group=component(name)
            if group and p.grad is not None:
                squares[group]=squares.get(group,0.)+float(p.grad.float().square().sum())
        self.report['gradient_checks'].append(dict(step=state.global_step+1,
                   component_gradient_norms={k:math.sqrt(v) for k,v in squares.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--objective',choices=('combined','genotype_only'),required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args();root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    if (root/'report.json').exists():raise ValueError('Preserve previous controls; use a new output directory')
    protocol=json.loads(Path(args.protocol).read_text())
    if protocol['updates']!=300 or protocol['seed']!=0:raise ValueError('Unexpected control protocol')
    transformers.set_seed(0)
    helper=VizEmbTrainer(args.config);helper.setup()
    dataset=helper.get_train_data()['train']
    if sorted({r['pid'] for r in dataset.data})!=protocol['selected_subjects']:
        raise ValueError('Control population mismatch')
    if {r['pid'] for r in dataset.data}&set(protocol['excluded_outer_test_subjects']):
        raise ValueError('Held-out subject entered learning control')
    model,_=helper.load_train_model();model.to('cuda')
    normal=fold_target_statistics(dataset.data)
    model.tbr_mean.copy_(torch.tensor(normal['mean'],device='cuda'))
    model.tbr_std.copy_(torch.tensor(normal['std'],device='cuda'))
    initial={name:value.detach().cpu().clone() for name,value in model.named_parameters() if value.requires_grad}
    report=dict(diagnostic_only=True,status='running',objective=args.objective,
                protocol=protocol,config_sha256=file_sha256(args.config),script_sha256=file_sha256(__file__),
                initial_trainable_sha256=state_digest(initial),initial_trainable_tensors=len(initial),
                torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
                target_statistics=normal,learning_curve=[],gradient_checks=[])
    atomic_json(root/'report.json',report)
    training_args=helper.get_training_args()
    training_args.max_steps=300
    callback=Progress(dataset,report,root/'report.json')
    trainer=ControlTrainer(model=model,tokenizer=helper.tokenizer,train_dataset=dataset,
                           data_collator=DefaultDataCollator(),args=training_args,
                           objective=args.objective,callbacks=[callback])
    # Check the loss actually used by the diagnostic against the intended
    # production calculation on identical eval-mode activations before fitting.
    model.eval()
    batch={k:v.cuda() for k,v in DefaultDataCollator()([dataset[0],dataset[1]]).items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        loss,out=trainer.compute_loss(model,batch,return_outputs=True)
        hidden=model._eos_hidden_state(batch['input_ids'],out.hidden_states,
                                       batch['question_end_index'],batch['attention_mask'])
        bce=F.binary_cross_entropy_with_logits(model.genotype_head(hidden).squeeze(-1),batch['genotype_label'])
        if args.objective=='genotype_only':expected=5*bce
        else:expected=model(**batch).loss
    if not torch.allclose(loss,expected,atol=1e-6,rtol=1e-6):
        raise AssertionError('The training control changed more than the declared objective')
    report['objective_check']=dict(actual=float(loss),expected=float(expected),status='passed')
    model.train(); trainer.train()
    report['updates']=trainer.state.global_step;report['epochs']=trainer.state.epoch
    sums={}
    for name,value in model.named_parameters():
        if name in initial:
            group=component(name)
            sums[group]=sums.get(group,0.)+float((value.detach().cpu().float()-initial[name].float()).square().sum())
    report['parameter_change_norms']={k:math.sqrt(v) for k,v in sums.items()}
    for group in ('language_projection','genotype_head','lora_A','lora_B'):
        if report['parameter_change_norms'].get(group,0)<=0:
            raise AssertionError(f'No parameter update in {group}')
    final=measure(model,dataset);report['final']=final
    checkpoint=root/'final_model';model.save_pretrained(checkpoint)
    # Reuse the frozen quantized backbone while strictly restoring every trained
    # adapter/head/projection tensor, then check outputs on the same input shape.
    loaded=VisionLanguageModel.from_pretrained(checkpoint,None,model.language_model,
                                               helper.img_token_id,tokenizer=helper.tokenizer)
    reloaded=measure(loaded,dataset)
    error=max(abs(a['logit']-b['logit']) for a,b in zip(final['rows'],reloaded['rows']))
    if error!=0:raise AssertionError(f'Control checkpoint did not reload exactly: {error}')
    report['reload_max_logit_error']=error
    summary=final['by_question']['genotype']
    report['capacity_criterion_met']=summary['accuracy']==1. and summary['bce']<=.1
    report['status']='complete';atomic_json(root/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('learning_curve','final','protocol')}),flush=True)


if __name__=='__main__':main()
