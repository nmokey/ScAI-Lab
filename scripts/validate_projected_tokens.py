"""Measure observed, forecast and trained-projection token norms on a fixed fold."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,file_sha256,verify_forecast_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',required=True)
    parser.add_argument('--forecasts',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    fold=Path(args.fold)
    training=json.loads((fold/'splits/train.json').read_text())
    test=json.loads((fold/'splits/val.json').read_text())
    unique={r['pid']:r for r in training+test};held=test[0]['pid']
    manifest=verify_forecast_manifest(args.forecasts,held,sorted(unique))
    weights_path=fold/'mouse_vlm_mdl/other_weights.bin'
    state=torch.load(weights_path,map_location='cpu',weights_only=True)
    weight=state['language_projection.weight'].float();bias=state['language_projection.bias'].float()
    rows=[]
    for sid,row in sorted(unique.items()):
        baseline=load_file(row['embedding_path_ts0'])['embeddings'].reshape(-1).float()
        fit=manifest['fits'][manifest['queries'][sid]]
        for tag in range(4):
            vector=baseline if tag==0 else torch.from_numpy(np.load(Path(args.forecasts)/f'{sid}_ts{tag}.npy')).reshape(-1).float()
            projected=torch.nn.functional.linear(vector,weight,bias)
            raw_norm=float(vector.norm());projected_norm=float(projected.norm())
            if not np.isfinite([raw_norm,projected_norm]).all() or min(raw_norm,projected_norm)<=0:
                raise ValueError('Invalid token norm')
            if tag and not np.isclose(raw_norm,fit['norm_reference'],rtol=1e-6):
                raise ValueError('Forecast norm differs from its training-fit reference')
            rows.append(dict(subject=sid,tag=tag,held_out=sid==held,raw_norm=raw_norm,
                             projected_norm=projected_norm,norm_reference=fit['norm_reference'] if tag else None))
    summary={}
    for tag in range(4):
        subset=[r for r in rows if r['tag']==tag]
        summary[f'ts{tag}']={key:dict(min=min(r[key] for r in subset),
                                   mean=float(np.mean([r[key] for r in subset])),max=max(r[key] for r in subset))
                            for key in ('raw_norm','projected_norm')}
    atomic_json(args.report,dict(status='passed',held_out=held,checkpoint_weights_sha256=file_sha256(weights_path),
                arithmetic='Float32 diagnostic with the saved trained projection; not an attention-mechanism causal test.',
                summary=summary,observations=rows))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
