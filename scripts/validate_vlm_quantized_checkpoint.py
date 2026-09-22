"""Offline plumbing check on the actual 4-bit LLaMA and current mouse inputs.

Two training updates only; this produces no research-performance metrics.
Runs the production dataset/trainer loader/save/inference loader and checks exact
adapter and eval-output restoration. Optionally validates repaired old checkpoints.
"""
import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch
import transformers
import peft
from peft.utils import get_peft_model_state_dict
from transformers import DefaultDataCollator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from model.viz_emb_trainer import VizEmbTrainer
from model.vision_language_model import VisionLanguageModel


def measure(model, batch):
    model.eval()
    keys=('input_ids','attention_mask','image_features','question_end_index')
    b={k:batch[k] for k in keys}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out=model(**b)
        feat=model._eos_hidden_state(b['input_ids'],out.hidden_states,b['question_end_index'],b['attention_mask'])
        return [x.float().cpu() for x in (out.logits,model.genotype_head(feat),model.tbr_regression_head(feat))]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--report',required=True)
    p.add_argument('--repaired-root')
    args=p.parse_args(); torch.manual_seed(123)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required for the quantized-model gate')
    trainer=VizEmbTrainer(args.config);trainer._setup_tokenizer()
    datasets=trainer.get_train_data();collate=DefaultDataCollator()
    # Include a TBR-supervised record so both heads receive gradients.
    rows=[datasets['train'][i] for i in range(min(3,len(datasets['train'])))]
    batch={k:v.cuda() for k,v in collate(rows).items()}
    model,_=trainer.load_train_model(); model.to('cuda')
    mean,std=trainer._tbr_norm_stats(datasets['train'])
    model.tbr_mean.copy_(mean.cuda());model.tbr_std.copy_(std.cuda())
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
    losses=[];gradients={}
    model.train()
    for step in range(2):
        opt.zero_grad()
        with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(**batch).loss
        if not torch.isfinite(loss): raise AssertionError('Nonfinite training loss')
        loss.backward();losses.append(float(loss))
        if step==1:
            for component in ('language_projection','genotype_head','tbr_regression_head','lora_A','lora_B'):
                norms=[float(p.grad.float().norm()) for k,p in model.named_parameters() if component in k and p.grad is not None]
                gradients[component]=max(norms,default=0)
                assert gradients[component]>0,(component,gradients)
        opt.step()
    del opt
    before=measure(model,batch);repeat=measure(model,batch)
    repeat_error=[float((a-b).abs().max()) for a,b in zip(before,repeat)]
    assert all(v==0 for v in repeat_error),repeat_error
    # Same shape, image, question and mask; only supplied answer tokens change.
    changed={k:v.clone() for k,v in batch.items()}
    positions=torch.arange(batch['input_ids'].size(1),device='cuda')[None,:]
    answer_mask=(positions>batch['question_end_index'][:,None]) & batch['attention_mask'].bool()
    changed['input_ids'][answer_mask]=42
    altered=measure(model,changed)
    answer_error=[float((a-b).abs().max()) for a,b in zip(before[1:],altered[1:])]
    assert all(v==0 for v in answer_error),answer_error
    expected={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model.language_model).items()}
    other={k:v.detach().cpu().clone() for k,v in model._other_state().items()}
    with tempfile.TemporaryDirectory(prefix='scai-quantized-roundtrip-') as tmp:
        trainer.output_dir=tmp;trainer.save_model(model)
        checkpoint=str(Path(tmp)/trainer.params['train']['save_model_name'])
        del model;gc.collect();torch.cuda.empty_cache()
        trainer.params['inf']['model_name']=checkpoint
        loaded,_=trainer.load_inf_model(); loaded.eval()
        after=measure(loaded,batch)
        reload_error=[float((a-b).abs().max()) for a,b in zip(before,after)]
        assert all(v==0 for v in reload_error),reload_error
        actual=get_peft_model_state_dict(loaded.language_model)
        assert expected.keys()==actual.keys()
        assert all(torch.equal(v,actual[k].detach().cpu()) for k,v in expected.items())
        assert all(torch.equal(v,loaded._other_state()[k].detach().cpu()) for k,v in other.items())
        # Question-only inference uses generate()'s inferred boundary.
        qend=int(batch['question_end_index'][0]); question={k:batch[k][:1] for k in ('input_ids','attention_mask','image_features')}
        question['input_ids']=question['input_ids'][:,:qend+1]
        question['attention_mask']=question['attention_mask'][:,:qend+1]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            generated=loaded.generate(**question,max_new_tokens=2,do_sample=False)
        assert all(torch.isfinite(generated[k]).all() for k in ('genotype_logits','tbr_logits'))
        repaired=[]
        if args.repaired_root:
            for config in sorted(Path(args.repaired_root).rglob('vlm_config.json')):
                directory=config.parent
                recovered=VisionLanguageModel.from_pretrained(directory,None,loaded.language_model,
                           trainer.img_token_id,tokenizer=trainer.tokenizer).eval()
                values=measure(recovered,batch)
                assert all(torch.isfinite(v).all() for v in values)
                repaired.append(dict(checkpoint=str(directory),strict_reload='passed',pool_at=recovered.pool_at))
    report=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
                gpu=torch.cuda.get_device_name(),training_updates=2,losses=losses,gradient_max_norms=gradients,
                adapter_tensors=len(expected),nonzero_B=sum(bool(v.count_nonzero()) for k,v in expected.items() if '.lora_B.' in k),
                repeated_eval_max_errors=repeat_error,reload_max_errors=reload_error,
                answer_perturbation_head_max_errors=answer_error,repaired_checkpoints=repaired,status='passed')
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
