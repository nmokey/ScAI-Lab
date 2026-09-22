"""Read-only saved-model diagnostic; outputs are NOT new performance estimates.

Compare each model's held-out score, in-sample discrimination, and response to
one identical reference input across folds. The reference is the first fold's
held-out input, chosen by directory order, with no label-based selection.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from model.viz_emb_trainer import VizEmbTrainer
from model.vision_language_model import VisionLanguageModel


@torch.inference_mode()
def score(model, trainer, sample):
    inputs = trainer.apply_tokenizer(trainer.prepare_question(sample['question'])).to('cuda')
    boundary = inputs['attention_mask'].sum(dim=1) - 1
    combined, mask, _ = model.get_image_and_text_embeddings(
        **inputs, image_features=sample['image_features'].to('cuda', dtype=torch.float))
    out = model.language_model(inputs_embeds=combined, attention_mask=mask,
                               output_hidden_states=True, use_cache=False)
    feat = model._eos_hidden_state(inputs['input_ids'], out.hidden_states,
                                  boundary, inputs['attention_mask'])
    return float(model.genotype_head(feat).squeeze().cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folds = sorted(args.run.glob('fold_*'))
    report = dict(run=str(args.run), diagnostic_only=True,
                  reference_definition='First fold held-out genotype input, identical across models; not a performance control',
                  folds=[])
    model = None
    reference = None
    for fold in folds:
        trainer = VizEmbTrainer(str(fold / 'exp.yml'))
        trainer._setup_tokenizer()
        if model is None:
            model, _ = trainer.load_inf_model()
        else:
            model = VisionLanguageModel.from_pretrained(
                str(fold / 'mouse_vlm_mdl'), None, model.language_model,
                trainer.img_token_id, tokenizer=trainer.tokenizer)
        model.eval()
        data = trainer.get_inf_data()
        held = next(data[i] for i, row in enumerate(data.data) if row['content_type'] == 'genotype')
        if reference is None:
            reference = held
            report['reference_pid'] = held['pid']
        held_score = score(model, trainer, held)
        saved = next(r for r in json.loads((fold / 'vqa.json').read_text()) if r['content_type'] == 'genotype')
        error = abs(held_score - saved['genotype_logit'])
        if error > 1e-5:
            raise AssertionError(f'Held-out reproduction failed: {fold.name}, {error}')
        ref_score = score(model, trainer, reference)
        data.data = json.loads((fold / 'splits' / 'train.json').read_text())
        train_scores = []
        for i, row in enumerate(data.data):
            if row['content_type'] != 'genotype':
                continue
            sample = data[i]
            label = sample['answer_vqa_numeric']['genotype']
            assert label == int('_KO_' in sample['pid'])
            train_scores.append(dict(pid=sample['pid'], label=label, score=score(model, trainer, sample)))
        labels = [r['label'] for r in train_scores]
        scores = [r['score'] for r in train_scores]
        result = dict(fold=fold.name, held_pid=held['pid'], held_label=saved['genotype_label'],
                      held_score=held_score, saved_score_error=error, reference_score=ref_score,
                      training_positive_fraction=float(np.mean(labels)),
                      training_auc=float(roc_auc_score(labels, scores)),
                      training_accuracy=float(np.mean((np.asarray(scores)>0)==np.asarray(labels))),
                      training_score_sd=float(np.std(scores)), training_scores=train_scores)
        report['folds'].append(result)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='training_scores'}), flush=True)
    rows = report['folds']
    held_scores = np.array([r['held_score'] for r in rows])
    ref_scores = np.array([r['reference_score'] for r in rows])
    report['summary'] = dict(
        held_auc=float(roc_auc_score([r['held_label'] for r in rows], held_scores)),
        reference_auc_by_held_label=float(roc_auc_score([r['held_label'] for r in rows], ref_scores)),
        held_reference_correlation=float(np.corrcoef(held_scores, ref_scores)[0,1]),
        held_score_sd=float(np.std(held_scores)), reference_score_sd=float(np.std(ref_scores)),
        median_training_score_sd=float(np.median([r['training_score_sd'] for r in rows])),
        median_training_auc=float(np.median([r['training_auc'] for r in rows])),
        min_training_auc=min(r['training_auc'] for r in rows),
        max_training_auc=max(r['training_auc'] for r in rows),
        reference_ko_fold_mean=float(np.mean([r['reference_score'] for r in rows if r['held_label']==1])),
        reference_wt_fold_mean=float(np.mean([r['reference_score'] for r in rows if r['held_label']==0])))
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
