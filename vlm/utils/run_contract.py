"""Frozen run identities and idempotent fold aggregation for the fixed protocol."""
import copy
import importlib.metadata
import json
from pathlib import Path

from utils.research_io import atomic_json, file_sha256, object_sha256
from utils.target_contract import FUTURE_WEEKS, unique_subject_targets


def validate_records(records):
    subjects=unique_subject_targets(records)
    for row in records:
        if row.get('target_schema_version')!=1 or row.get('future_weeks')!=FUTURE_WEEKS or row.get('input_week')!='Week 12':
            raise ValueError('Validated runs require structured targets and fixed Week-12-to-15/18/20 queries')
        if row.get('content_type') not in ('genotype','tbr','combined'):
            raise ValueError('Unknown question type')
    for sid in subjects:
        kinds=[r['content_type'] for r in records if r['pid']==sid]
        if sorted(kinds)!=['combined','genotype','tbr']:
            raise ValueError(f'Expected the same three questions for every subject: {sid}')
    return sorted(subjects)


def make_run_spec(params, records, all_data_path):
    subjects=validate_records(records)
    config=copy.deepcopy(params)
    config.pop('exp',None)
    for key in ('model_name','train_gt_file'):
        config['inf'].pop(key,None)
    for key in ('data_path','inf_data_path','eval_data_path'):
        config['data'].pop(key,None)
    config['train'].pop('resume_from_checkpoint',None)
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):file_sha256(p) for p in root.rglob('*.py')}
    repo=root.parent
    for name in ('build_nested_forecasts.py','train_longitudinal.py','create_mouse_traj_dataset.py'):
        sources['scripts/'+name]=file_sha256(repo/'scripts'/name)
    baseline={r['pid']:file_sha256(r['embedding_path_ts0']) for r in records}
    nested=params['data'].get('nested_forecast_root')
    forecast_index=file_sha256(Path(nested)/'index.json') if nested else None
    model_name=params['train']['llm_model_name']
    cache=Path.home()/'.cache/huggingface/hub'/('models--'+model_name.replace('/','--'))
    revision=(cache/'refs/main').read_text().strip() if (cache/'refs/main').exists() else None
    if revision is None:raise ValueError('Cached backbone revision is required for reproducible runs')
    environment={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','bitsandbytes','numpy','safetensors','accelerate')}
    return dict(format_version=1,subjects=subjects,record_count=len(records),config=config,
                data_sha256=file_sha256(all_data_path),baseline_sha256=baseline,
                nested_index_sha256=forecast_index,source_sha256=sources,environment=environment,
                backbone=model_name,backbone_revision=revision,
                primary_metric='subject_averaged_future_proxy_MAE',training_selection='fixed_final_epoch')


def freeze_run(directory, spec):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'run_manifest.json'
    identity=object_sha256(spec)
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('run_id')!=identity:raise ValueError('Run fingerprints changed; use a new output directory')
    else:
        if any(directory.glob('fold_*')) or (directory/'vqa_loso.json').exists():
            raise ValueError('Unversioned results exist; refusing to mix historical and repaired folds')
        atomic_json(path,dict(run_id=identity,spec=spec))
    return identity


def completed_fold(directory,run_id,truth):
    from data.validated_metrics import collapse_predictions
    directory=Path(directory)
    marker=directory/'complete.json'
    if not marker.exists():return None
    state=json.loads(marker.read_text())
    if state.get('run_id')!=run_id:raise ValueError('Completed fold belongs to a different run')
    for name,digest in state['files'].items():
        if file_sha256(directory/name)!=digest:raise ValueError(f'Completed fold artifact changed: {name}')
    rows=json.loads((directory/'vqa.json').read_text())
    collapse_predictions(rows,truth)
    return rows


def mark_fold_complete(directory,run_id,checkpoint_name):
    directory=Path(directory)
    files=[directory/'vqa.json',directory/'val.json',directory/'target_statistics.json']
    files+=sorted((directory/checkpoint_name).rglob('*'))
    files=[p for p in files if p.is_file()]
    if not (directory/checkpoint_name/'vlm_config.json').exists():raise ValueError('Missing final checkpoint')
    atomic_json(directory/'complete.json',dict(run_id=run_id,files={str(p.relative_to(directory)):file_sha256(p) for p in files}))


def collect_complete_folds(directory,subjects,records,run_id):
    rows=[];complete=[]
    for index,sid in enumerate(subjects):
        truth=[r for r in records if r['pid']==sid]
        part=completed_fold(Path(directory)/f'fold_{index:02d}_{sid}',run_id,truth)
        if part is not None:rows+=part;complete.append(sid)
    keys=[(r['pid'],str(r['qid'])) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate aggregate predictions')
    is_complete=complete==list(subjects)
    atomic_json(Path(directory)/('vqa_loso.json' if is_complete else 'vqa_loso_partial.json'),rows)
    atomic_json(Path(directory)/'run_status.json',dict(status='complete' if is_complete else 'incomplete',
                run_id=run_id,complete_subjects=complete,expected_subjects=list(subjects)))
    return rows,is_complete
