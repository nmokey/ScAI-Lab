"""Aggregate complete, paired validated VLM runs without pooling seeds as animals.

Use --baseline and --longitudinal for the fixed comparison. Positional historical
runs only print their existing per-run metrics; they are never certified here.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,object_sha256,file_sha256,verify_forecast_manifest
from utils.run_contract import completed_fold
from data.validated_metrics import paired_comparison,conditional_mean_interval


def read_complete_run(directory,truth):
    directory=Path(directory)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    if manifest.get('run_id')!=object_sha256(manifest['spec']):
        raise ValueError(f'Run manifest fingerprint does not match its specification: {directory}')
    status=json.loads((directory/'run_status.json').read_text())
    if status.get('status')!='complete' or status.get('run_id')!=manifest['run_id']:
        raise ValueError(f'Incomplete or mismatched run: {directory}')
    spec=manifest['spec']
    for row in truth:
        if file_sha256(row['embedding_path_ts0'])!=spec['baseline_sha256'].get(row['pid']):
            raise ValueError('Baseline input differs from the frozen run')
    if spec['config']['data']['img_tokens']==4:
        root=Path(spec['config']['data']['nested_forecast_root'])
        if file_sha256(root/'index.json')!=spec['nested_index_sha256']:
            raise ValueError('Nested forecast index changed after training')
        index=json.loads((root/'index.json').read_text())
        if index['subjects']!=spec['subjects']:raise ValueError('Nested index population differs')
        for sid in spec['subjects']:
            if file_sha256(root/sid/'manifest.json')!=index['folds'].get(sid):
                raise ValueError('Nested fold manifest differs from the frozen index')
            verify_forecast_manifest(root/sid,sid,spec['subjects'])
    predictions=[]
    for index,sid in enumerate(spec['subjects']):
        part=completed_fold(directory/f'fold_{index:02d}_{sid}',manifest['run_id'],
                            [r for r in truth if r['pid']==sid])
        if part is None:raise ValueError(f'Incomplete fold coverage: {directory}')
        predictions.extend(part)
    # Scoring is read-only: the saved complete aggregate must match its verified
    # fold artifacts. The training runner alone creates/rebuilds that aggregate.
    aggregate=json.loads((directory/'vqa_loso.json').read_text())
    if aggregate!=predictions:raise ValueError('Saved aggregate differs from completed fold predictions')
    return manifest['spec'],predictions


def aggregate_paired(baseline_dirs,longitudinal_dirs,truth,seeds=(0,1,2)):
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError('Prespecified seeds must be nonempty and unique')
    arms=[]
    for directories in (baseline_dirs,longitudinal_dirs):
        arm={}
        for directory in directories:
            spec,predictions=read_complete_run(directory,truth)
            seed=int(spec['config']['data']['data_seed'])
            if seed in arm:raise ValueError(f'Duplicate seed {seed}')
            arm[seed]=(spec,predictions,str(Path(directory).resolve()))
        if set(arm)!=set(seeds):raise ValueError(f'Incomplete prespecified seeds: expected {seeds}, found {sorted(arm)}')
        reference=arm[seeds[0]][0]
        for spec,_,_ in arm.values():
            for key in ('data_sha256','baseline_sha256','source_sha256','environment','backbone_revision','subjects','nested_index_sha256'):
                if spec[key]!=reference[key]:raise ValueError(f'Unmatched {key} across seeds')
            # Initialization is the only permitted within-arm configuration change.
            def without_seed(config):
                import copy
                normalized=copy.deepcopy(config)
                normalized['data'].pop('data_seed',None)
                return normalized
            if without_seed(spec['config'])!=without_seed(reference['config']):
                raise ValueError('Unmatched configuration across seeds')
        arms.append(arm)
    results=[]
    for seed in seeds:
        base,long=arms[0][seed],arms[1][seed]
        for key in ('data_sha256','baseline_sha256','source_sha256','environment','backbone_revision','subjects'):
            if base[0][key]!=long[0][key]:raise ValueError(f'Unmatched {key} across arms')
        if base[0]['config']['train']!=long[0]['config']['train']:raise ValueError('Unmatched training settings')
        if base[0]['config']['data']['img_tokens']!=1 or long[0]['config']['data']['img_tokens']!=4:
            raise ValueError('Expected baseline one-token and longitudinal four-token arms')
        comparison=paired_comparison(base[1],long[1],truth)
        results.append(dict(seed=seed,baseline_directory=base[2],longitudinal_directory=long[2],**comparison))
    subjects=sorted(results[0]['per_subject_delta'])
    per_subject={sid:float(np.mean([r['per_subject_delta'][sid] for r in results])) for sid in subjects}
    deltas=[r['delta_subject_mae'] for r in results]
    return dict(protocol_version=1,seeds=list(seeds),n_subjects=len(subjects),per_seed=results,
                mean_paired_delta_subject_mae=float(np.mean(deltas)),
                paired_delta_seed_sd=statistics.stdev(deltas) if len(deltas)>1 else None,
                fixed_prediction_subject_interval95=conditional_mean_interval(list(per_subject.values())),
                per_subject_seed_mean_delta=per_subject,
                interpretation='Negative MAE difference favors longitudinal. Seeds are repeated fits on the same mice, not additional animals. The interval is descriptive, conditional on fitted predictions, and ignores acquisition-group dependence.',
                significance_test=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='*')
    parser.add_argument('--baseline',nargs='+');parser.add_argument('--longitudinal',nargs='+')
    parser.add_argument('--truth');parser.add_argument('--output');parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--rescore',action='store_true',help='Historical rescoring is intentionally disabled; use the explicit validated comparison')
    args=parser.parse_args()
    if args.baseline or args.longitudinal:
        if not all((args.baseline,args.longitudinal,args.truth,args.output)):parser.error('Paired scoring requires both arms, --truth and --output')
        truth=json.loads(Path(args.truth).read_text())
        report=aggregate_paired(args.baseline,args.longitudinal,truth,args.seeds)
        atomic_json(args.output,report)
        lines=['# Corrected paired VLM comparison','',
               'Endpoint: existing PET intensity proxy; subject-held-out evaluation. Negative MAE difference favors longitudinal.','',
               '| Seed | Baseline subject MAE | Longitudinal subject MAE | Difference | Baseline genotype AUROC | Longitudinal genotype AUROC |',
               '|---|---:|---:|---:|---:|---:|']
        for row in report['per_seed']:
            b,l=row['baseline'],row['longitudinal']
            fmt=lambda value:'n/a' if value is None else f'{value:.4f}'
            lines.append(f"| {row['seed']} | {fmt(b['tbr_reg_subject_mae'])} | {fmt(l['tbr_reg_subject_mae'])} | {row['delta_subject_mae']:+.4f} | {fmt(b['genotype_auc'])} | {fmt(l['genotype_auc'])} |")
        lines+=['',report['interpretation'],'','No independent-group genotype generalization or anatomically validated aortic endpoint is claimed. Per-horizon results and training-mean comparators are in the accompanying JSON.']
        Path(args.output).with_suffix('.md').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines))
    else:
        if args.rescore:parser.error('Historical runs must not be overwritten or interpreted as corrected results')
        print('Historical per-run values only; no validated comparison, pooled-rank AUROC or significance claim.')
        for directory in args.runs:
            p=Path(directory)/'loso_results.json'
            row=json.loads(p.read_text())
            print(directory,{k:row.get(k) for k in ('genotype_auc','tbr_reg_overall_mae','tbr_reg_overall_r2')})


if __name__=='__main__':main()
