"""Independently check saved VLM metrics using only standard-library arithmetic.

Reads complete raw predictions and structured ground truth. Does not import the
production scorer, fit models, alter predictions or change the fixed protocol.
"""
import argparse
import hashlib
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(runs,truth_path,paired_path):
    report=json.loads(Path(paired_path).read_text());truth=json.loads(Path(truth_path).read_text())
    targets={r['pid']:r['answer_vqa_numeric']['tbr'] for r in truth if r['content_type']=='tbr'}
    expected={(r['pid'],str(r['qid'])) for r in truth};checks=[]
    for result in report['per_seed']:
        for directory_arm,arm in [('base','baseline'),('long','longitudinal')]:
            path=Path(runs)/f"validated_{directory_arm}_seed{result['seed']}"/'vqa_loso.json'
            rows=json.loads(path.read_text())
            keys=[(r['pid'],str(r['qid'])) for r in rows]
            if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Invalid raw prediction coverage')
            predictions=defaultdict(lambda:defaultdict(list));genotype=[]
            for row in rows:
                if row['content_type']=='genotype':genotype.append((row['genotype_label'],row['genotype_logit']))
                else:
                    for slot in range(3):
                        if targets[row['pid']][slot]>=0:predictions[row['pid']][slot].append(row['tbr_regression'][slot])
            errors={sid:{k:stats.mean(v)-targets[sid][k] for k,v in slots.items()} for sid,slots in predictions.items()}
            primary=stats.mean(stats.mean(abs(v) for v in slots.values()) for slots in errors.values())
            pos=[s for y,s in genotype if y==1];neg=[s for y,s in genotype if y==0]
            auc=sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))
            mean_errors={sid:stats.mean(abs(stats.mean(v[k] for other,v in targets.items() if other!=sid and v[k]>=0)-t)
                         for k,t in enumerate(values[:3]) if t>=0) for sid,values in targets.items()}
            comparator=stats.mean(mean_errors.values());saved=result[arm]
            if not math.isclose(primary,saved['tbr_reg_subject_mae'],abs_tol=1e-12):raise ValueError('Primary MAE differs')
            if not math.isclose(auc,saved['genotype_auc'],abs_tol=1e-12):raise ValueError('Pairwise AUROC differs')
            # Production means are stored as float32 checkpoint buffers.
            if not math.isclose(comparator,saved['train_mean_comparator']['subject_mae'],abs_tol=2e-6):
                raise ValueError('Training-mean comparator differs')
            horizons={};dispersion={}
            for k,delta,week in [(0,3,15),(1,6,18),(2,8,20)]:
                residuals=[v[k] for v in errors.values() if k in v]
                subjects=[sid for sid,v in errors.items() if k in v]
                values=[targets[sid][k] for sid in subjects];predicted=[stats.mean(predictions[sid][k]) for sid in subjects]
                mae=stats.mean(abs(v) for v in residuals);mse=stats.mean(v*v for v in residuals)
                r2=1-sum(v*v for v in residuals)/sum((v-stats.mean(values))**2 for v in values)
                measured=dict(mae=mae,mse=mse,r2=r2)
                for metric,value in measured.items():
                    if not math.isclose(value,saved['tbr_reg_by_slot'][f'delta_{delta}wk'][metric],abs_tol=1e-12):
                        raise ValueError(f'Horizon {week} {metric} differs')
                horizons[f'delta_{delta}wk']=measured
                dispersion[f'W{week}']=dict(target_sd=stats.pstdev(values),prediction_sd=stats.pstdev(predicted))
            checks.append(dict(arm=arm,seed=result['seed'],subject_mae=primary,genotype_auc=auc,
                               mean_predictor_subject_mae=comparator,horizon_metrics=horizons,
                               genotype_positive_predictions=sum(s>0 for _,s in genotype),
                               genotype_ko_mean_logit=stats.mean(pos),genotype_wt_mean_logit=stats.mean(neg),
                               prediction_dispersion=dispersion,prediction_sha256=sha(path)))
    return dict(status='passed',method='Independent standard-library arithmetic; no production scoring imports.',
                checks=checks,truth_sha256=sha(truth_path),paired_results_sha256=sha(paired_path),checker_sha256=sha(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='/data1/Processed_NIfTI_Test/embeddings/vlm/runs')
    parser.add_argument('--truth',default='/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')
    parser.add_argument('--paired',default='docs/audit_2026-09-14/paired_vlm_results.json')
    parser.add_argument('--report',default='docs/audit_2026-09-14/independent_metric_check.json')
    args=parser.parse_args();report=check(args.runs,args.truth,args.paired)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(f"Independently verified all {len(report['checks'])} runs: primary MAE, pairwise AUROC, per-horizon MAE/MSE/R² and training-mean comparator.")


if __name__=='__main__':main()
