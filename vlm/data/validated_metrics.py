"""Strict, subject-weighted scoring for the repaired fixed-horizon protocol.

Intervals describe resampling of fixed cross-validated predictions, not new
training runs or independent acquisition groups. No fixed-score permutation
p-value is presented as a test of the learning procedure.
"""
import math
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.target_contract import numeric_targets, unique_subject_targets, fold_target_statistics


def finite_vector(values, name, size=4):
    if values is None or len(values) != size or not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"Missing, nonfinite or incorrectly sized {name}")
    return np.asarray(values, dtype=float)


def regression_metrics(truth, predictions):
    truth, predictions = np.asarray(truth,float),np.asarray(predictions,float)
    if not len(truth):
        return dict(n=0,mae=None,mse=None,r2=None,pearson_r=None)
    error=predictions-truth
    variance=float(np.sum((truth-truth.mean())**2))
    correlation=(float(np.corrcoef(truth,predictions)[0,1])
                 if len(truth)>1 and np.std(truth)>0 and np.std(predictions)>0 else None)
    return dict(n=len(truth),mae=float(np.mean(abs(error))),mse=float(np.mean(error**2)),
                r2=1-float(np.sum(error**2))/variance if variance>0 else None,pearson_r=correlation)


def conditional_mean_interval(values, seed=0, draws=5000):
    values=np.asarray(values,float)
    if len(values)<2:return None
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def collapse_predictions(predictions, truth, require_complete=True):
    expected={(r['pid'],str(r['qid'])):r for r in truth}
    if len(expected)!=len(truth):raise ValueError('Duplicate ground-truth records')
    population=unique_subject_targets(truth)
    seen=set(); result={sid:dict(genotype_label=v['genotype'],genotype_scores=[],
                               tbr_targets=v['tbr'],tbr_predictions=defaultdict(list),
                               train_mean=None) for sid,v in population.items()}
    for row in predictions:
        key=(row.get('pid'),str(row.get('qid')))
        if key in seen or key not in expected:raise ValueError(f'Duplicate or unexpected prediction: {key}')
        seen.add(key); source=expected[key]; sid=key[0]; entry=result[sid]
        if row.get('content_type')!=source.get('content_type'):
            raise ValueError(f'Prediction routing differs from ground truth: {key}')
        if row.get('genotype_label')!=entry['genotype_label']:
            raise ValueError(f'Conflicting genotype target: {key}')
        if source['content_type']=='genotype':
            score=row.get('genotype_logit')
            if score is None or not math.isfinite(float(score)):raise ValueError(f'Invalid genotype score: {key}')
            entry['genotype_scores'].append(float(score))
        targets=finite_vector(row.get('tbr_targets'),'TBR targets')
        if not np.allclose(targets,numeric_targets(source),atol=1e-6,rtol=0):
            raise ValueError(f'Prediction target differs from ground truth: {key}')
        values=finite_vector(row.get('tbr_regression'),'TBR predictions')
        means=finite_vector(row.get('train_mean_prediction'),'training-fold mean predictions')
        if entry['train_mean'] is not None and not np.array_equal(entry['train_mean'],means):
            raise ValueError(f'Conflicting training means: {sid}')
        entry['train_mean']=means
        for slot,target in enumerate(targets[:3]):
            if target>=0:entry['tbr_predictions'][slot].append(float(values[slot]))
    if require_complete and seen!=set(expected):raise ValueError('Incomplete prediction coverage')
    for sid,entry in result.items():
        if not entry['genotype_scores']:raise ValueError(f'Missing genotype prediction: {sid}')
        entry['genotype_score']=float(np.mean(entry.pop('genotype_scores')))
        entry['tbr_predictions']={slot:float(np.mean(values)) for slot,values in entry['tbr_predictions'].items()}
        for slot,target in enumerate(entry['tbr_targets'][:3]):
            if target>=0 and slot not in entry['tbr_predictions']:raise ValueError(f'Missing TBR prediction: {sid}/{slot}')
    return result


def score_predictions(predictions, truth, train_records=None):
    collapsed=collapse_predictions(predictions,truth)
    # Validate the deployable comparator against the actual training population.
    # Aggregate LOSO scoring reconstructs the outer-training mean independently.
    for sid,entry in collapsed.items():
        training=train_records if train_records is not None else [r for r in truth if r['pid']!=sid]
        if sid in {r['pid'] for r in training}:raise ValueError('Test subject occurs in training mean reference')
        stats=fold_target_statistics(training)
        if not np.allclose(entry['train_mean'],stats['mean'],atol=2e-5,rtol=1e-6):
            raise ValueError(f'Training mean does not match the declared training fold for {sid}')
    subject_mae={};mean_subject_mae={}; by_slot={}; all_y=[];all_p=[];all_m=[]
    for slot,delta in enumerate((3,6,8)):
        rows=[(s,e['tbr_targets'][slot],e['tbr_predictions'][slot],e['train_mean'][slot])
              for s,e in collapsed.items() if e['tbr_targets'][slot]>=0]
        y=[r[1] for r in rows]; p=[r[2] for r in rows]; m=[r[3] for r in rows]
        by_slot[f'delta_{delta}wk']=dict(**regression_metrics(y,p),train_mean=regression_metrics(y,m))
        all_y+=y;all_p+=p;all_m+=m
    for sid,entry in collapsed.items():
        slots=list(entry['tbr_predictions'])
        if slots:
            subject_mae[sid]=float(np.mean([abs(entry['tbr_predictions'][k]-entry['tbr_targets'][k]) for k in slots]))
            mean_subject_mae[sid]=float(np.mean([abs(entry['train_mean'][k]-entry['tbr_targets'][k]) for k in slots]))
    y=[e['genotype_label'] for e in collapsed.values()];s=[e['genotype_score'] for e in collapsed.values()]
    auc=float(roc_auc_score(y,s)) if len(set(y))==2 else None
    overall=regression_metrics(all_y,all_p); comparator=regression_metrics(all_y,all_m)
    return dict(protocol_version=1,total=len(predictions),genotype_n=len(y),genotype_auc=auc,
                genotype_acc=float(np.mean((np.asarray(s)>0)==np.asarray(y))),
                genotype_threshold=0.0,genotype_calibration_status='not_assessed',
                genotype_auc_perm_p=None,genotype_permutation_status='not_run_requires_full_refitting',
                tbr_reg_subject_mae=float(np.mean(list(subject_mae.values()))) if subject_mae else None,
                tbr_reg_subject_mae_conditional_ci95=conditional_mean_interval(list(subject_mae.values())),
                tbr_reg_overall_mae=overall['mae'],tbr_reg_overall_mse=overall['mse'],
                tbr_reg_overall_r2=overall['r2'],tbr_reg_overall_r=overall['pearson_r'],
                tbr_reg_n_subjects=len(subject_mae),tbr_reg_n_observations=len(all_y),tbr_reg_by_slot=by_slot,
                train_mean_comparator=dict(**comparator,subject_mae=float(np.mean(list(mean_subject_mae.values()))) if mean_subject_mae else None),
                per_subject_mae=subject_mae,per_subject_train_mean_mae=mean_subject_mae,
                uncertainty='Conditional fixed-prediction subject resampling; does not model refitting or acquisition-group dependence.',
                population_limit='NaF has three acquisition components and one WT component; no independent-group genotype generalization claim.')


def paired_comparison(base_predictions,long_predictions,truth):
    base=score_predictions(base_predictions,truth);long=score_predictions(long_predictions,truth)
    subjects=sorted(base['per_subject_mae'])
    if subjects!=sorted(long['per_subject_mae']):raise ValueError('Unpaired subject masks')
    delta={s:long['per_subject_mae'][s]-base['per_subject_mae'][s] for s in subjects}
    return dict(baseline=base,longitudinal=long,primary='Longitudinal minus baseline subject-averaged MAE; negative favors longitudinal',
                delta_subject_mae=float(np.mean(list(delta.values()))),per_subject_delta=delta,
                delta_conditional_ci95=conditional_mean_interval(list(delta.values())),
                genotype_auc_delta=long['genotype_auc']-base['genotype_auc'] if long['genotype_auc'] is not None and base['genotype_auc'] is not None else None)
