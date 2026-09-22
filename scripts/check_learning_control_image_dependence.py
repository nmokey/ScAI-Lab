"""Read-only image-swap and identical-input checks of the completed controls."""
import argparse
import json
from pathlib import Path

import torch

from run_real_mouse_learning_control import measure
from model.viz_emb_trainer import VizEmbTrainer
from model.vision_language_model import VisionLanguageModel
from utils.research_io import atomic_json


class ModifiedInputs:
    def __init__(self,dataset,mode):
        self.dataset=dataset;self.data=dataset.data;self.mode=mode
        self.subjects=sorted({r['pid'] for r in self.data})
        self.indices={(r['pid'],r['content_type']):i for i,r in enumerate(self.data)}
        self.donors={s:self.subjects[(i+4)%8] for i,s in enumerate(self.subjects)}
        self.mean=torch.stack([dataset[self.indices[s,'genotype']]['image_features'] for s in self.subjects]).mean(0)

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        item=dict(self.dataset[index]);record=self.data[index]
        if self.mode=='swap':
            donor=self.indices[self.donors[record['pid']],record['content_type']]
            item['image_features']=self.dataset[donor]['image_features']
        else:item['image_features']=self.mean
        return item


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    args=p.parse_args();root=Path(args.directory);model=None;checks=[]
    for objective in ('combined','genotype_only'):
        report=json.loads((root/objective/'report.json').read_text())
        assert report['status']=='complete'
        helper=VizEmbTrainer(str(root/f'{objective}.yml'));helper._setup_tokenizer()
        data=helper.get_train_data()['train']
        helper.params['inf']['model_name']=str(root/objective/'final_model')
        if model is None:model,_=helper.load_inf_model()
        else:model=VisionLanguageModel.from_pretrained(root/objective/'final_model',None,
                       model.language_model,helper.img_token_id,tokenizer=helper.tokenizer)
        model.eval();ordinary=measure(model,data)
        assert ordinary==report['final']
        values={(r['pid'],r['kind']):r['logit'] for r in ordinary['rows']}
        swapped_data=ModifiedInputs(data,'swap');swapped=measure(model,swapped_data)
        error=max(abs(r['logit']-values[swapped_data.donors[r['pid']],r['kind']]) for r in swapped['rows'])
        if error>1e-5:raise AssertionError(f'Swapped outputs do not follow donor images: {error}')
        mean=measure(model,ModifiedInputs(data,'mean'))
        ranges={kind:max(r['logit'] for r in mean['rows'] if r['kind']==kind)-
                     min(r['logit'] for r in mean['rows'] if r['kind']==kind)
                for kind in ('genotype','tbr','combined')}
        if max(ranges.values())>1e-5:raise AssertionError('Identical images retain subject-specific head scores')
        checks.append(dict(objective=objective,fresh_reload_exact=True,
                           image_swap_donor_logit_max_error=error,identical_image_logit_ranges=ranges,
                           donor_subjects=swapped_data.donors))
    result=dict(status='passed',diagnostic_only=True,checks=checks,
                interpretation='Outputs follow swapped images; identical images remove between-mouse score differences while questions, answer tokens, masks and labels remain unchanged')
    atomic_json(root/'image_dependence.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
