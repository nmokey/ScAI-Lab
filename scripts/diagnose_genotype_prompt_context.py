"""Read-only per-question genotype objective diagnostic on a saved run."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_genotype_separation import stats
from diagnose_genotype_fold_offsets import score
from model.viz_emb_trainer import VizEmbTrainer
from model.vision_language_model import VisionLanguageModel
from utils.research_io import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); model = None
    report = dict(diagnostic_only=True, status='running',
                  definition='Saved eval-mode genotype outputs by training question type; bias gradients are BCE-only at fixed features, without training dropout',
                  folds=[])
    for fold in sorted(Path(args.run).glob('fold_*')):
        trainer = VizEmbTrainer(str(fold/'exp.yml')); trainer._setup_tokenizer()
        if model is None:
            model,_ = trainer.load_inf_model()
        else:
            model = VisionLanguageModel.from_pretrained(str(fold/'mouse_vlm_mdl'),None,
                        model.language_model,trainer.img_token_id,tokenizer=trainer.tokenizer)
        model.eval(); train = json.loads((fold/'splits/train.json').read_text())
        test = json.loads((fold/'splits/val.json').read_text()); train_ids = {r['pid'] for r in train}
        data = trainer.get_inf_data(); data.data = train+test; rows=[]
        saved = {(r['pid'],str(r['qid'])):r['genotype_logit'] for r in json.loads((fold/'vqa.json').read_text())}
        for i in range(len(data)):
            sample = data[i]; value = score(model,trainer,sample)
            if sample['pid'] not in train_ids:
                assert value == saved[sample['pid'],str(sample['qid'])]
            rows.append(dict(pid=sample['pid'],qid=sample['qid'],kind=sample['content_type'],
                             membership='train' if sample['pid'] in train_ids else 'test',
                             label=sample['answer_vqa_numeric']['genotype'],score=value))
        prior = np.mean([r['answer_vqa_numeric']['genotype'] for r in train])
        groups = {}
        for kind in ('genotype','tbr','combined'):
            rr=[r for r in rows if r['membership']=='train' and r['kind']==kind]
            groups[kind] = stats(rr)
            groups[kind]['genotype_bias_bce_gradient'] = groups[kind]['probability_mean']-float(prior)
        all_train = stats([r for r in rows if r['membership']=='train'])
        result = dict(fold=fold.name,training_positive_fraction=float(prior),training_by_question=groups,
                      training_all_questions=all_train,
                      all_questions_genotype_bias_bce_gradient=all_train['probability_mean']-float(prior),rows=rows)
        report['folds'].append(result);atomic_json(args.output,report)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    report['status']='complete';atomic_json(args.output,report)


if __name__=='__main__':main()
