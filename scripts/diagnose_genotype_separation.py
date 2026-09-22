"""Read-only train/test, default-response and precision diagnostics of saved fits.

No fitting or correction of published scores. Training results, centroids and
alternate-precision outputs are explicitly diagnostic, not selected metrics.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from model.viz_emb_trainer import VizEmbTrainer
from model.vision_language_model import VisionLanguageModel
from utils.research_io import atomic_json, file_sha256
from diagnose_genotype_fold_offsets import score


def stats(rows, key='score'):
    y = np.array([r['label'] for r in rows]); s = np.array([r[key] for r in rows])
    p = 1 / (1 + np.exp(-s))
    return dict(n=len(y), auc=float(roc_auc_score(y, s)), accuracy=float(np.mean((s>0)==y)),
                logit_sd=float(np.std(s)), probability_min=float(p.min()), probability_max=float(p.max()),
                probability_mean=float(p.mean()), probability_sd=float(np.std(p)),
                probability_ko_minus_wt=float(p[y==1].mean()-p[y==0].mean()),
                logit_ko_minus_wt=float(s[y==1].mean()-s[y==0].mean()),
                bce=float(np.mean(np.logaddexp(0,s)-y*s)), positive_predictions=int((s>0).sum()),
                tp=int(((s>0)&(y==1)).sum()), fp=int(((s>0)&(y==0)).sum()),
                tn=int(((s<=0)&(y==0)).sum()), fn=int(((s<=0)&(y==1)).sum()))


def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    report = dict(diagnostic_only=True, status='running',
                  scope='All 24 saved four-fold models; genotype-only questions; no training or metric replacement',
                  folds=[])
    model = None
    for arm in ('base','long'):
        for seed in (0,1,2):
            run = Path(args.runs) / f'stratified_{arm}_seed{seed}'
            for fold in sorted(run.glob('fold_*')):
                trainer = VizEmbTrainer(str(fold / 'exp.yml')); trainer._setup_tokenizer()
                if model is None:
                    model, _ = trainer.load_inf_model()
                else:
                    model = VisionLanguageModel.from_pretrained(
                        str(fold / 'mouse_vlm_mdl'), None, model.language_model,
                        trainer.img_token_id, tokenizer=trainer.tokenizer)
                model.eval()
                training = json.loads((fold / 'splits' / 'train.json').read_text())
                test = json.loads((fold / 'splits' / 'val.json').read_text())
                train_ids = {r['pid'] for r in training}
                data = trainer.get_inf_data()
                data.data = [r for r in training + test if r['content_type']=='genotype']
                samples = [data[i] for i in range(len(data))]
                assert len({s['question'] for s in samples}) == 1
                saved = {r['pid']:r['genotype_logit'] for r in json.loads((fold/'vqa.json').read_text())
                         if r['content_type']=='genotype'}
                mean_features = torch.stack([s['image_features'] for s in samples if s['pid'] in train_ids]).mean(dim=0)
                neutral = copy.copy(samples[0]); neutral['image_features'] = mean_features
                default_score = score(model, trainer, neutral)
                class_baselines = {}
                for label in (0,1):
                    class_baselines[label] = np.mean([s['image_features'][0,0].numpy().astype(float)
                        for s in samples if s['pid'] in train_ids and s['answer_vqa_numeric']['genotype']==label],axis=0)
                rows = []
                for sample in samples:
                    sid = sample['pid']; value = score(model, trainer, sample)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        bf16 = score(model, trainer, sample)
                    if sid in saved and value != saved[sid]:
                        raise AssertionError(f'Saved-score reproduction failed: {run.name}/{fold.name}/{sid}: {value-saved[sid]}')
                    baseline = sample['image_features'][0,0].numpy().astype(float)
                    rows.append(dict(pid=sid, label=sample['answer_vqa_numeric']['genotype'],
                                     membership='train' if sid in train_ids else 'test', score=value,
                                     bf16_score=bf16, deviation_from_training_centroid=value-default_score,
                                     baseline_cosine_to_ko_minus_wt=cosine(baseline,class_baselines[1])-cosine(baseline,class_baselines[0])))
                train_rows = [r for r in rows if r['membership']=='train']
                test_rows = [r for r in rows if r['membership']=='test']
                prior = sum(r['label'] for r in train_rows) / len(train_rows)
                result = dict(arm=arm, seed=seed, fold=fold.name,
                              checkpoint_marker_sha256=file_sha256(fold/'complete.json'),
                              saved_logits_exact=True, training_positive_fraction=prior,
                              training_prior_logit=math.log(prior/(1-prior)),
                              training_centroid_logit=default_score,
                              train=stats(train_rows), test=stats(test_rows),
                              train_bf16=stats(train_rows,'bf16_score'), test_bf16=stats(test_rows,'bf16_score'),
                              train_baseline_centroid_diagnostic=stats(train_rows,'baseline_cosine_to_ko_minus_wt'),
                              test_baseline_centroid_diagnostic=stats(test_rows,'baseline_cosine_to_ko_minus_wt'),
                              test_precision_threshold_flips=sum((r['score']>0)!=(r['bf16_score']>0) for r in test_rows),
                              test_precision_max_logit_change=max(abs(r['score']-r['bf16_score']) for r in test_rows),
                              rows=rows)
                report['folds'].append(result); atomic_json(args.output,report)
                print(json.dumps({k:result[k] for k in ('arm','seed','fold','train','test','test_bf16',
                                                        'test_precision_threshold_flips','test_precision_max_logit_change')}),flush=True)
    report['status'] = 'complete'; atomic_json(args.output,report)


if __name__ == '__main__':
    main()
