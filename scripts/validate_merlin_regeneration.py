"""Verify regenerated Merlin inputs and recompute the existing comparison metrics.

Point estimates use the existing evaluator. Genotype group-label inference uses
all 120 full-refit arrangements. No fixed-score permutation p-values are used.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,file_sha256
import evaluate_embeddings as evaluator
from eval_stats import join_mouse_group_ids,permutation_refit_p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='/data1/Processed_NIfTI_Test/embeddings/merlin_validated_20260914')
    parser.add_argument('--manifest',default='/data1/Processed_NIfTI_Test/mouse_manifest.csv')
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    root=Path(args.directory);archive=root/'merlin_embeddings.npz'
    provenance=json.loads((root/'input_manifest.json').read_text())
    if not provenance['complete'] or provenance['embedding_sha256']!=file_sha256(archive):
        raise ValueError('Merlin artifact incomplete or changed')
    if provenance['source_manifest_sha256']!=file_sha256(args.manifest):
        raise ValueError('Crop manifest changed')
    data=np.load(archive,allow_pickle=True)
    embs=data['embeddings'];subjects=data['subject_ids'].astype(str);weeks=data['weeks'].astype(str)
    keys=list(zip(subjects,weeks))
    reference=np.load(root.parent/'raddino/raddino_embeddings.npz',allow_pickle=True)
    if len(keys)!=len(set(keys)) or set(keys)!=set(zip(reference['subject_ids'],reference['weeks'])):
        raise ValueError('Encoder populations differ or include duplicate visits')
    if embs.shape!=(229,2048) or not np.isfinite(embs).all() or (np.linalg.norm(embs,axis=1)==0).any():
        raise ValueError('Invalid Merlin vectors')
    for path,digest in zip(data['paths'].astype(str),data['source_sha256'].astype(str)):
        if provenance['source_sha256'].get(path)!=digest or file_sha256(path)!=digest:
            raise ValueError(f'Encoder input changed: {path}')
    groups=join_mouse_group_ids(subjects,weeks,args.manifest)
    genotypes=np.array([s.split('_')[1] for s in subjects])
    cohorts=np.array([s.split('_')[0] for s in subjects])
    labels=(genotypes=='KO').astype(int)
    metrics={}
    metrics.update(evaluator.run_t2a_week_classification(embs,weeks,subjects))
    for group_name,group in [('subject',subjects),('mousegroup',groups)]:
        for task,mask,y in [('genotype',np.ones(len(embs),bool),labels),
                            ('early_late',np.isin(weeks,['Week 12','Week 20']),(weeks=='Week 20').astype(int))]:
            acc,_,truth,scores=evaluator._loso_logistic(embs[mask],y[mask],group[mask],task,return_proba=True)
            if len(truth)!=int(mask.sum()):raise ValueError('Evaluator omitted held-out observations')
            metrics[f'{task}_{group_name}']=dict(n=len(truth),accuracy=float(acc),auc=float(roc_auc_score(truth,scores)))
    metrics.update(evaluator.run_t2d_naf_vs_fdg(embs,subjects,cohorts))
    metrics.update(evaluator.run_t3a_temporal_ordering(embs,weeks,subjects))
    metrics.update(evaluator.run_t3b_subject_retrieval(embs,subjects))
    metrics.update(evaluator.run_t3c_week_retrieval(embs,weeks,k=5))
    count=0
    def fit_predict(x,y,g):
        nonlocal count
        count+=1
        with contextlib.redirect_stdout(io.StringIO()):
            _,_,truth,scores=evaluator._loso_logistic(x,y,g,'genotype refit',return_proba=True)
        if len(truth)!=len(y):raise ValueError('Refit omitted held-out observations')
        if count%20==0:print(f'Genotype full refits {count}/121',flush=True)
        return np.asarray(truth),np.asarray(scores)
    observed,p,null,meta=permutation_refit_p(fit_predict,embs,labels,groups)
    metrics['genotype_mousegroup'].update(exact_refit_p=p,refit_null95=list(null),permutation=meta)
    if not np.isclose(observed,metrics['genotype_mousegroup']['auc']):raise ValueError('Refit point estimate differs')
    report=dict(status='passed',embedding_path=str(archive),embedding_sha256=file_sha256(archive),
                source_manifest_sha256=file_sha256(args.manifest),n_scans=len(embs),n_subjects=len(set(subjects)),
                n_groups=len(set(groups)),metrics=metrics,
                evaluator_sha256=file_sha256(Path(evaluator.__file__)),
                limitations=['Off-the-shelf human-scale preprocessing; not an isolated architecture comparison.',
                             'Genotype and diet are confounded; no disease localization or confirmatory encoder-selection claim.',
                             'Week and acquisition date coincide.'])
    atomic_json(args.report,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
