import copy
import json

import numpy as np
import pytest

from utils.target_contract import fold_target_statistics,numeric_targets
from data.validated_metrics import score_predictions,paired_comparison
from utils.run_contract import freeze_run,collect_complete_folds
from utils.research_io import atomic_json


def examples():
    truth=[];pred=[]
    for i in range(4):
        for kind in ('genotype','tbr','combined'):
            targets=[float(i+1),float(i+2) if i<2 else -1.,-1.,-1.]
            if kind=='genotype':targets=[-1.]*4
            row=dict(pid=f'm{i}',qid=f'{i}_{kind}',content_type=kind,target_schema_version=1,
                     answer_vqa_numeric=dict(genotype=i%2,tbr=targets))
            truth.append(row)
    for row in truth:
        means=fold_target_statistics([r for r in truth if r['pid']!=row['pid']])['mean']
        pred.append(dict(pid=row['pid'],qid=row['qid'],content_type=row['content_type'],
                         genotype_label=row['answer_vqa_numeric']['genotype'],genotype_logit=0.,
                         tbr_targets=row['answer_vqa_numeric']['tbr'],
                         tbr_regression=[v+1 if v>=0 else 0 for v in row['answer_vqa_numeric']['tbr']],
                         train_mean_prediction=means))
    return truth,pred


def test_structured_targets_ignore_wording_and_normalize_unique_observations():
    truth,_=examples()
    for row in truth:row['question']='predict an outcome without the old keyword';row['answer']='irrelevant prose'
    assert numeric_targets(truth[1])==[1.,2.,-1.,-1.]
    stats=fold_target_statistics(truth)
    assert stats['counts']==[4,2,0,0]
    assert stats['mean']==[2.5,2.5,0.,0.]
    assert stats['std'][0]==pytest.approx(np.std([1,2,3,4],ddof=0))
    # A missing target is masked; singleton observations still have their correct mean.
    assert fold_target_statistics(truth[:3])['mean'][:2]==[1.,2.]


def test_subject_weighted_metrics_pairing_and_training_mean_comparator():
    truth,pred=examples();result=score_predictions(pred,truth)
    assert result['tbr_reg_subject_mae']==1
    assert result['tbr_reg_n_observations']==6
    assert result['genotype_auc']==.5  # tied predictions
    better=copy.deepcopy(pred)
    for row in better:row['tbr_regression']=[v if v>=0 else 0 for v in row['tbr_targets']]
    comparison=paired_comparison(pred,better,truth)
    assert comparison['delta_subject_mae']==-1
    assert comparison['delta_conditional_ci95']==[-1.,-1.]
    assert result['genotype_auc_perm_p'] is None


@pytest.mark.parametrize('damage',['duplicate','missing','nan','target','mean','routing'])
def test_bad_predictions_are_rejected(damage):
    truth,pred=examples()
    if damage=='duplicate':pred.append(copy.deepcopy(pred[0]))
    if damage=='missing':pred.pop()
    if damage=='nan':pred[1]['tbr_regression'][0]=float('nan')
    if damage=='target':pred[1]['tbr_targets']=list(pred[1]['tbr_targets']);pred[1]['tbr_targets'][0]+=1
    if damage=='mean':
        for r in pred:r['train_mean_prediction']=[0.,0.,0.,0.]
    if damage=='routing':pred[0]['content_type']='combined'
    with pytest.raises(ValueError):score_predictions(pred,truth)


def test_fixed_future_query_does_not_reveal_missing_visits(scripts):
    module=scripts('create_mouse_traj_dataset')
    sid='NaF_KO_01'
    embs={sid:{'Week 12':np.ones(768),'Week 15':np.ones(768),'Week 20':np.ones(768)}}
    tbr={sid:{'Week 15':2.,'Week 20':3.}}
    records,_=module.get_records_for_subject(sid,embs,tbr,{},0)
    altered=copy.deepcopy(embs);altered[sid]['Week 18']=np.ones(768)
    with_visit,_=module.get_records_for_subject(sid,altered,{sid:{**tbr[sid],'Week 18':4.}},{},0)
    assert [r['question'] for r in records]==[r['question'] for r in with_visit]
    assert records[1]['future_weeks']==['Week 15','Week 18','Week 20']
    assert records[1]['answer_vqa_numeric']['tbr']==[2.,-1.,3.,-1.]


def test_stale_resume_rejected_and_partial_output_never_published_as_complete(tmp_path):
    spec={'protocol':1};rid=freeze_run(tmp_path,spec)
    assert freeze_run(tmp_path,spec)==rid
    with pytest.raises(ValueError,match='fingerprints'):freeze_run(tmp_path,{'protocol':2})
    rows,complete=collect_complete_folds(tmp_path,['m0'],[],rid)
    assert not complete and rows==[]
    assert not (tmp_path/'vqa_loso.json').exists()
    assert json.loads((tmp_path/'run_status.json').read_text())['status']=='incomplete'


def test_loso_training_prior_demonstrates_why_fixed_score_shuffle_is_not_refitting():
    from sklearn.metrics import roc_auc_score
    y=np.array([0,0,1,1])
    scores=(y.sum()-y)/(len(y)-1)
    assert roc_auc_score(y,scores)==0
    shuffled=1-y
    assert roc_auc_score(shuffled,scores)==1
    assert roc_auc_score(shuffled,(shuffled.sum()-shuffled)/(len(y)-1))==0


def test_cache_key_changes_when_file_contents_or_preprocessing_change(tmp_path):
    from utils.image_cache import image_cache_key
    source=tmp_path/'image.nii';source.write_bytes(b'old scan')
    first=image_cache_key({'image':str(source)},'v1')
    source.write_bytes(b'new scan')
    second=image_cache_key({'image':str(source)},'v1')
    assert first!=second
    assert second!=image_cache_key({'image':str(source)},'v2')


def test_physical_tbr3_mapping_respects_origin_and_rejects_missing_geometry(scripts):
    module=scripts('extract_tbr_features')
    ct=np.zeros((120,100,60));ct[35:45,25:35,:]=800
    pet=np.random.default_rng(0).gamma(4,.25,size=(30,25,15))
    pet[9:12,14:17,:]=50
    ct_affine=np.diag([.1,.1,.1,1]);pet_affine=np.diag([.4,.4,.4,1])
    original=module.tbr_strategy_3(ct,pet,ct_affine,pet_affine)
    shifted_affine=pet_affine.copy();shifted_affine[0,3]=-2.
    shifted=module.tbr_strategy_3(ct,np.roll(pet,5,axis=0),ct_affine,shifted_affine)
    assert original['tbr3_tbr']>2
    assert original['tbr3_tbr']==pytest.approx(shifted['tbr3_tbr'])
    with pytest.raises(ValueError,match='affines'):module.tbr_strategy_3(ct,pet)


def test_fold_resume_is_idempotent_and_detects_changed_predictions(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    import yaml
    from utils.target_contract import fold_target_statistics
    from utils.run_contract import collect_complete_folds
    path=Path(__file__).resolve().parents[2]/'vlm/run/run_mouse_vlm_loso.py'
    spec=importlib.util.spec_from_file_location('strict_loso_test',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    truth,_=examples();subjects=sorted({r['pid'] for r in truth});calls=[]
    class Lifecycle:
        def __init__(self,path):self.params=yaml.safe_load(Path(path).read_text())
        def setup(self):pass
        def run(self):
            calls.append('train')
            out=Path(self.params['exp']['output_dir'])
            checkpoint=out/self.params['train']['save_model_name'];checkpoint.mkdir(exist_ok=True)
            atomic_json(checkpoint/'vlm_config.json',{'test_double':True})
            train=json.loads(Path(self.params['data']['data_path']).read_text())
            atomic_json(out/'target_statistics.json',fold_target_statistics(train))
        def evaluate(self):
            held=json.loads(Path(self.params['data']['inf_data_path']).read_text())
            training=json.loads(Path(self.params['data']['data_path']).read_text())
            means=fold_target_statistics(training)['mean'];predictions=[]
            for r in held:
                targets=numeric_targets(r)
                predictions.append(dict(pid=r['pid'],qid=r['qid'],content_type=r['content_type'],
                     genotype_label=r['answer_vqa_numeric']['genotype'],genotype_logit=0.,
                     tbr_targets=targets,tbr_regression=[max(0.,v) for v in targets],train_mean_prediction=means))
            out=Path(self.params['exp']['output_dir'])
            atomic_json(out/'vqa.json',predictions)
            atomic_json(out/'val.json',score_predictions(predictions,held,training))
    monkeypatch.setattr(runner,'get_model',lambda _,path:Lifecycle(path))
    params={'exp':{},'data':{'img_tokens':1},'inf':{},'train':{'save_model_name':'model'}}
    for index,sid in enumerate(subjects):
        runner.run_fold(sid,index,truth,str(tmp_path),params,subjects=subjects,run_id='fixed')
        runner.run_fold(sid,index,truth,str(tmp_path),params,subjects=subjects,run_id='fixed')
    assert len(calls)==len(subjects)
    rows,complete=collect_complete_folds(tmp_path,subjects,truth,'fixed')
    assert complete and len(rows)==len(truth)
    rows2,_=collect_complete_folds(tmp_path,subjects,truth,'fixed')
    assert rows==rows2
    first=tmp_path/'fold_00_m0/vqa.json'
    altered=json.loads(first.read_text());altered[0]['genotype_logit']=99
    atomic_json(first,altered)
    with pytest.raises(ValueError,match='artifact changed'):
        runner.run_fold(subjects[0],0,truth,str(tmp_path),params,subjects=subjects,run_id='fixed')


@pytest.mark.parametrize('damage',['source','forecast','train','missing_seed'])
def test_aggregate_rejects_mixed_versions_across_seeds(scripts,monkeypatch,damage):
    module=scripts('aggregate_seeds')
    truth,pred=examples()
    common=dict(data_sha256='data',baseline_sha256='images',source_sha256='source',
                environment={},backbone_revision='fixed',subjects=['m0','m1','m2','m3'])
    specs={}
    for arm,tokens in [('base',1),('long',4)]:
        for seed in (0,1,2):
            specs[f'{arm}{seed}']=dict(common,nested_index_sha256='nested' if tokens==4 else None,
                 config=dict(data=dict(data_seed=seed,img_tokens=tokens),train=dict(epochs=20)))
    if damage=='source':specs['base1']['source_sha256']='changed'
    if damage=='forecast':specs['long2']['nested_index_sha256']='changed'
    if damage=='train':specs['base2']['config']['train']['epochs']=21
    monkeypatch.setattr(module,'read_complete_run',lambda directory,_:(specs[directory],pred))
    bases=['base0','base1','base2'];longs=['long0','long1','long2']
    if damage=='missing_seed':longs.pop()
    with pytest.raises(ValueError,match='Unmatched|Incomplete'):
        module.aggregate_paired(bases,longs,truth)


def test_aggregate_keeps_seeds_separate_and_reproduces_paired_errors(scripts,monkeypatch):
    module=scripts('aggregate_seeds');truth,pred=examples()
    def read(directory,_):
        arm,seed=directory.split(':');tokens=1 if arm=='base' else 4
        spec=dict(data_sha256='data',baseline_sha256='images',source_sha256='source',environment={},
                  backbone_revision='fixed',subjects=['m0','m1','m2','m3'],
                  nested_index_sha256='nested' if tokens==4 else None,
                  config=dict(data=dict(data_seed=int(seed),img_tokens=tokens),train=dict(epochs=20)))
        rows=copy.deepcopy(pred)
        if arm=='long':
            for r in rows:r['tbr_regression']=[max(0.,v) for v in r['tbr_targets']]
        return spec,rows
    monkeypatch.setattr(module,'read_complete_run',read)
    result=module.aggregate_paired([f'base:{s}' for s in (0,1,2)],[f'long:{s}' for s in (0,1,2)],truth)
    assert result['n_subjects']==4 and len(result['per_seed'])==3
    assert result['mean_paired_delta_subject_mae']==-1
    assert result['paired_delta_seed_sd']==0


@pytest.mark.parametrize('damage',['index','fold_manifest'])
def test_completed_run_rejects_forecasts_changed_after_training(scripts,tmp_path,damage):
    from utils.research_io import file_sha256,object_sha256
    module=scripts('aggregate_seeds')
    forecasts=tmp_path/'forecasts';(forecasts/'m0').mkdir(parents=True)
    atomic_json(forecasts/'m0/manifest.json',{'original':'manifest'})
    atomic_json(forecasts/'index.json',{'subjects':['m0'],
                'folds':{'m0':file_sha256(forecasts/'m0/manifest.json')}})
    spec=dict(subjects=['m0'],baseline_sha256={},nested_index_sha256=file_sha256(forecasts/'index.json'),
              config={'data':{'img_tokens':4,'nested_forecast_root':str(forecasts)}})
    rid=object_sha256(spec)
    run=tmp_path/'run';atomic_json(run/'run_manifest.json',{'run_id':rid,'spec':spec})
    atomic_json(run/'run_status.json',{'status':'complete','run_id':rid})
    if damage=='index':atomic_json(forecasts/'index.json',{'changed':'index'})
    else:atomic_json(forecasts/'m0/manifest.json',{'changed':'manifest'})
    with pytest.raises(ValueError,match='Nested forecast index changed|Nested fold manifest differs'):
        module.read_complete_run(run,[])
