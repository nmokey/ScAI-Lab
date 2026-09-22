"""Evaluate actual baseline-only nested exports against persistence and train-week means."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,verify_forecast_manifest


def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecasts',required=True);p.add_argument('--embeddings',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();root=Path(args.forecasts);index=json.loads((root/'index.json').read_text());subjects=index['subjects']
    data=np.load(args.embeddings,allow_pickle=True)
    lookup={(str(s),str(w)):e for s,w,e in zip(data['subject_ids'],data['weeks'],data['embeddings'])}
    rows=[]
    for sid in subjects:
        manifest=verify_forecast_manifest(root/sid,sid,subjects)
        fit=manifest['fits'][manifest['queries'][sid]]
        for tag,week in enumerate(('Week 15','Week 18','Week 20'),1):
            forecast=np.load(root/sid/f'{sid}_ts{tag}.npy')
            norm=float(np.linalg.norm(forecast))
            if not np.isclose(norm,fit['norm_reference'],rtol=1e-6):raise ValueError('Forecast normalization mismatch')
            if (sid,week) not in lookup:continue
            target=lookup[(sid,week)];baseline=lookup[(sid,'Week 12')]
            centroid=np.mean([lookup[(other,week)] for other in subjects if other!=sid and (other,week) in lookup],axis=0)
            rows.append(dict(subject=sid,week=week,forecast_cosine=cosine(forecast,target),
                             persistence_cosine=cosine(baseline,target),train_week_centroid_cosine=cosine(centroid,target),
                             forecast_norm=norm,observed_baseline_norm=float(np.linalg.norm(baseline)),norm_reference=fit['norm_reference']))
    summary={week:{'n':sum(r['week']==week for r in rows),**{key:float(np.mean([r[key] for r in rows if r['week']==week]))
                      for key in ('forecast_cosine','persistence_cosine','train_week_centroid_cosine')}} for week in ('Week 15','Week 18','Week 20')}
    report=dict(status='passed',summary=summary,observations=rows,
                interpretation='Actual autoregressive baseline-only exports. Rescaling occurs before recurrent feedback; it changes later directions as well as token magnitudes. Centroids use all eligible training-week observations and are a practical diagnostic, not an identical-pair ablation.')
    atomic_json(args.report,report);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
