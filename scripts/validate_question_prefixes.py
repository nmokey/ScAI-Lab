"""Check training/inference token prefixes on every frozen fixed-horizon record."""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from data.vqa_dataset import MouseTrajDataset
from utils.huggingface_utils import load_tokenizer_from_huggingface
from utils.research_io import atomic_json,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')
    parser.add_argument('--forecasts',default='/data1/Processed_NIfTI_Test/embeddings/longitudinal_nested_20260914/NaF_KO_01')
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    tokenizer=load_tokenizer_from_huggingface('meta-llama/Meta-Llama-3.1-8B-Instruct')
    tokenizer.add_bos_token=False;tokenizer.add_tokens(['<image>'])
    kwargs=dict(tokenizer=tokenizer,prompt_type='standard',beg_prompt='',mid_prompt='',end_prompt='',
                data_path=args.dataset,img_tokens=4,predicted_emb_dir=args.forecasts,seq_length=150)
    training=MouseTrajDataset(**kwargs,mode='train');testing=MouseTrajDataset(**kwargs,mode='test')
    maximum=0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',RuntimeWarning)
        for index in range(len(training)):
            train,test=training[index],testing[index]
            prefix=tokenizer(test['question'])['input_ids'];end=train['question_end_index']+1
            if list(train['input_ids'][:end])!=prefix or end!=len(prefix):
                raise ValueError('Training and inference head prefixes differ')
            if not np.all(np.asarray(train['labels'][:end])==-100):
                raise ValueError('Question tokens unexpectedly have language targets')
            if not np.array_equal(train['image_features'].numpy(),test['image_features'][0].numpy()):
                raise ValueError('Training and inference image tokens differ')
            maximum=max(maximum,int(np.sum(train['attention_mask'])))
        truncations=[str(w.message) for w in captured if 'truncat' in str(w.message).lower()]
    if truncations:raise ValueError(f'The fixed dataset has truncated language targets: {truncations}')
    atomic_json(args.report,dict(status='passed',records=len(training),all_training_prefixes_equal_inference=True,
                all_image_tokens_equal=True,seq_length=150,max_combined_tokens=maximum,truncated_records=0,
                data_sha256=file_sha256(args.dataset)))
    print(f'All {len(training)} fixed-dataset prefixes and image inputs agree; maximum {maximum}/150 tokens; no truncation.')


if __name__=='__main__':main()
