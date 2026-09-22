"""Verify all six fixed runs and produce the authoritative paired result package.

The optional wait mode only waits for the prespecified jobs; it does not change
their settings or choose which seeds, subjects or endpoints to report.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,file_sha256
from aggregate_seeds import aggregate_paired


def render(report,merlin):
    fmt=lambda value:'n/a' if value is None else f'{value:.4f}'
    rows=['# Validated research results','',
          'All six prespecified VLM runs are complete: 32 held-out mice × two arms × three seeds. Checkpoint hashes, source/input fingerprints and paired prediction coverage were verified before scoring. The endpoint is the existing PET intensity proxy, not a validated aortic measurement.','',
          '**Current genotype evaluation:** use the [completed four-fold amendment](audit_2026-09-14/stratified/RESULTS.md), which avoids ranking scores across different fitted models. The table below preserves the original LOSO study; its pooled genotype AUROCs are not the amended discrimination estimates. See also [why the remaining small separations reverse](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md).','',
          '## Baseline versus longitudinal VLM','',
          '| Seed | Baseline subject MAE | Longitudinal subject MAE | Longitudinal − baseline | Baseline genotype AUROC | Longitudinal genotype AUROC |',
          '|---|---:|---:|---:|---:|---:|']
    for result in report['per_seed']:
        base,long=result['baseline'],result['longitudinal']
        rows.append(f"| {result['seed']} | {fmt(base['tbr_reg_subject_mae'])} | {fmt(long['tbr_reg_subject_mae'])} | {result['delta_subject_mae']:+.4f} | {fmt(base['genotype_auc'])} | {fmt(long['genotype_auc'])} |")
    delta=report['mean_paired_delta_subject_mae'];interval=report['fixed_prediction_subject_interval95']
    signs=[r['delta_subject_mae']<0 for r in report['per_seed']]
    interpretation=('Longitudinal has lower primary MAE in every seed.' if all(signs) else
                    'Longitudinal does not have lower primary MAE in any seed.' if not any(signs) else
                    'The direction of the primary difference varies across seeds.')
    rows+=['',f"Mean paired MAE difference: **{delta:+.4f}**; initialization SD {fmt(report['paired_delta_seed_sd'])}. {interpretation}",
           f"The descriptive fixed-prediction subject interval for the seed-mean difference is [{interval[0]:+.4f}, {interval[1]:+.4f}]. It does not account for refitting or acquisition-group dependence and is not a significance test.",'',
           'Primary MAE averages valid-horizon errors within each mouse and then across mice. Negative differences favor longitudinal. Seeds are repeated fits on the same 32 mice, not extra animals. All 67 observed future subject/horizon targets are included (W15: 32; W18: 15; W20: 20).','',
           '## Per-horizon results','',
           'Values below are means of the three per-seed metrics; R² is ordinary held-out R². Every horizon is shown.','',
           '| Horizon | n | Baseline MAE | Longitudinal MAE | Training-mean MAE | Baseline MSE | Longitudinal MSE | Training-mean MSE | Baseline R² | Longitudinal R² | Training-mean R² |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,week in [('delta_3wk','W15'),('delta_6wk','W18'),('delta_8wk','W20')]:
        b=[r['baseline']['tbr_reg_by_slot'][key] for r in report['per_seed']]
        l=[r['longitudinal']['tbr_reg_by_slot'][key] for r in report['per_seed']]
        avg=lambda arm,k:float(np.mean([v[k] for v in arm]))
        m=b[0]['train_mean']
        values=[avg(b,'mae'),avg(l,'mae'),m['mae'],avg(b,'mse'),avg(l,'mse'),m['mse'],avg(b,'r2'),avg(l,'r2'),m['r2']]
        rows.append(f"| {week} | {b[0]['n']} | "+' | '.join(fmt(v) for v in values)+' |')
    comparator=report['per_seed'][0]['baseline']['train_mean_comparator']['subject_mae']
    base_mean=float(np.mean([r['baseline']['tbr_reg_subject_mae'] for r in report['per_seed']]))
    long_mean=float(np.mean([r['longitudinal']['tbr_reg_subject_mae'] for r in report['per_seed']]))
    n_improved=sum(v<0 for v in report['per_subject_seed_mean_delta'].values())
    rows+=['',f"The actual training-fold mean predictor has subject MAE {comparator:.4f}. Baseline's seed-mean MAE is {base_mean:.4f}; longitudinal's is {long_mean:.4f}. Longitudinal reduces seed-mean absolute error for {n_improved}/32 mice. The accompanying JSON preserves every mouse's paired error and the observation-weighted continuity metrics.",'',
           'Per-horizon MAE differences by seed (longitudinal minus baseline):','',
           '| Horizon | Seed 0 | Seed 1 | Seed 2 |','|---|---:|---:|---:|']
    horizon_consistency=[]
    for key,week in [('delta_3wk','W15'),('delta_6wk','W18'),('delta_8wk','W20')]:
        differences=[r['longitudinal']['tbr_reg_by_slot'][key]['mae']-r['baseline']['tbr_reg_by_slot'][key]['mae'] for r in report['per_seed']]
        rows.append(f'| {week} | '+' | '.join(f'{value:+.4f}' for value in differences)+' |')
        horizon_consistency.append(f"{week}: {sum(value<0 for value in differences)}/{len(differences)} seeds")
    rows+=['','Longitudinal has lower MAE at '+', '.join(horizon_consistency)+'. These are secondary per-horizon results (32, 15 and 20 observed mice respectively), and do not override the primary comparison.','',
           '## Validity and limits','',
           '**Genotype evaluation amendment complete:** the genotype AUROCs above pool different leave-one-out models and remain historical diagnostics of that protocol. The verified four-fold amendment compares only predictions from the same fitted model: baseline AUROC 0.5161 / 0.4839 / 0.4355 versus longitudinal 0.4194 / 0.4516 / 0.4516. Both independent scorers agree; no consistent longitudinal gain is established. See [the amended results](audit_2026-09-14/stratified/RESULTS.md) and [the investigation of weak/reversed class separation](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md). The original numerical outputs are unchanged.','',
           'Both arms use the same question-boundary heads, targets, folds, seeds and final-state training schedule. The longitudinal tokens are generated through five inner cross-fits plus an outer-training forecaster per held-out mouse. The installed Trainer executes 220 updates (reported epoch approximately 19.53) under the fixed requested 20-epoch configuration. No setting was selected from these held-out scores.','',
           'The NaF subset has three acquisition components (KO 4, KO 10, WT 18), with one WT component. These subject-held-out results cannot establish independent-group genotype generalization. Calibration is not assessed; threshold accuracy uses a fixed zero logit. Genotype and diet are confounded. The PET proxy has not been validated against source ROI measurements or histology.','',
           'This comparison tests adding three predicted tokens to the existing baseline. It does not establish the mechanism of a gain or personalized future modeling. Actual nested rollout cosine exceeds persistence but trails the training-week centroid at all three horizons. Extra observed visits, alternative endpoints and additional VLM control arms were outside this validation task.','',
           '## Other verified results','',
           '* All 111 PET-2 source measurements reproduce exactly; corrected RAD-DINO crops and baseline features reproduce exactly.',
           '* RAD-DINO all-week group-held-out genotype/diet AUROC remains 0.82819, exact full-refit group-label p=0.00833 (229 scans, 78 mice, ten groups). This is an association, not disease localization; encoder selection limits confirmatory wording.']
    if merlin:
        m=merlin['metrics']['genotype_mousegroup']
        rows.append(f"* Fresh Merlin extraction includes all 229 scans. Its group-held-out genotype/diet AUROC is {m['auc']:.4f}, exact full-refit p={m['exact_refit_p']:.4f} over 120 arrangements. The old cached Merlin rows are superseded.")
        metrics=merlin['metrics']
        rows+=['','### Refreshed Merlin comparison rows','',
               '| Existing metric | Fresh result |','|---|---:|']
        for name,value in [('Four-class week accuracy, subject holdout',metrics['T2a_accuracy']),
                           ('Early/late AUROC, subject holdout',metrics['early_late_subject']['auc']),
                           ('Early/late AUROC, group holdout',metrics['early_late_mousegroup']['auc']),
                           ('Genotype/diet accuracy, subject holdout',metrics['genotype_subject']['accuracy']),
                           ('Genotype/diet AUROC, subject holdout',metrics['genotype_subject']['auc']),
                           ('Genotype/diet AUROC, group holdout',m['auc']),
                           ('Cohort AUROC, subject holdout',metrics['T2d_auc']),
                           ('Temporal-ordering accuracy',metrics['T3a_pairwise_ordering_acc']),
                           ('Subject retrieval Recall@1',metrics['T3b_recall_at_1']),
                           ('Week retrieval mAP@5',metrics['T3c_map_at_5'])]:
            rows.append(f'| {name} | {value:.5f} |')
    rows+=['','## Evidence and reproduction','',
           '[Paired machine-readable results](audit_2026-09-14/paired_vlm_results.json), [closure report](audit_2026-09-14/PIPELINE_CLOSURE.md), [fixed protocol](audit_2026-09-14/EXPERIMENT_PROTOCOL.json), [commands and run paths](experiments.md), [Merlin metrics](audit_2026-09-14/merlin_regenerated_metrics.json), [RAD-DINO refit evidence](audit_2026-09-14/group_refit.log). Each paired result records its source run directories. Each run retains raw predictions, split files, checkpoint files and verification manifests.','',
           'Historical VLM summaries, including those previously labeled “leak-free,” are archived and excluded from this corrected result.']
    return '\n'.join(rows)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='/data1/Processed_NIfTI_Test/embeddings/vlm/runs')
    parser.add_argument('--truth',default='/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json')
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];audit=repo/'docs/audit_2026-09-14'
    roots=[[Path(args.runs)/f'validated_{arm}_seed{seed}' for seed in (0,1,2)] for arm in ('base','long')]
    previous=None
    while True:
        states=[]
        for directory in sum(roots,[]):
            path=directory/'run_status.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            states.append((directory.name,state.get('status'),len(state.get('complete_subjects',[]))))
        if states!=previous:print(json.dumps(states),flush=True);previous=states
        if all(s[1]=='complete' for s in states):break
        if not args.wait:raise ValueError('All six complete runs are required')
        time.sleep(30)
    truth=json.loads(Path(args.truth).read_text())
    # Validate current immutable inputs, not only equality between paired manifests.
    for directory in sum(roots,[]):
        spec=json.loads((directory/'run_manifest.json').read_text())['spec']
        if file_sha256(args.truth)!=spec['data_sha256']:raise ValueError('Truth data changed')
        for row in truth:
            if file_sha256(row['embedding_path_ts0'])!=spec['baseline_sha256'][row['pid']]:
                raise ValueError('Baseline feature changed after training')
        for name,digest in spec['source_sha256'].items():
            source=repo/name if name.startswith('scripts/') else repo/'vlm'/name
            if file_sha256(source)!=digest:raise ValueError(f'Frozen production source changed: {source}')
    report=aggregate_paired(roots[0],roots[1],truth)
    if report['n_subjects']!=32 or any(r[a]['total']!=96 or r[a]['tbr_reg_n_observations']!=67
                                     for r in report['per_seed'] for a in ('baseline','longitudinal')):
        raise ValueError('Prespecified population or target coverage differs')
    report['truth_sha256']=file_sha256(args.truth)
    atomic_json(audit/'paired_vlm_results.json',report)
    merlin_path=audit/'merlin_regenerated_metrics.json'
    merlin=json.loads(merlin_path.read_text()) if merlin_path.exists() else None
    text=render(report,merlin)
    # Only this successfully verified complete package replaces the current ledger.
    output=repo/'docs/results.md';temporary=output.with_suffix('.md.tmp')
    temporary.write_text(text);temporary.replace(output)
    companion=text.replace('audit_2026-09-14/','').replace('](experiments.md)','](../experiments.md)')
    (audit/'paired_vlm_results.md').write_text(companion)
    atomic_json(audit/'final_package_status.json',dict(status='complete',runs=[str(p) for p in sum(roots,[])],
                paired_results_sha256=file_sha256(audit/'paired_vlm_results.json'),results_md_sha256=file_sha256(output)))
    closure=audit/'PIPELINE_CLOSURE.md'
    if closure.exists():
        content=closure.read_text()
        content=content.replace('The matched three-seed study is running. **Complete corrected VLM metrics are not yet available for all prescribed seeds.**',
                                'The matched three-seed study is complete. **All six runs passed final artifact verification; corrected paired metrics are available.**')
        content=content.replace('The six full VLM runs and final paired aggregation are the remaining computational closure steps.',
                                'All six full VLM runs and final paired aggregation are complete; see [the verified package](paired_vlm_results.json).')
        closure.write_text(content)
    plan=repo/'docs/MASTER_PLAN.md'
    if plan.exists():
        content=plan.read_text()
        for phrase in ('Run matching folds/settings for all selected seeds;',
                       'Aggregate only complete matched arms/seeds;',
                       'Produce one authoritative baseline/longitudinal table',
                       'Include an explicit check against training-fold mean predictions',
                       'Save reproducible scoring commands, environment/config fingerprints,'):
            content=content.replace('- [ ] '+phrase,'- [x] '+phrase)
        content=content.replace('Full paired runs and final aggregation remain in progress.',
                                'All six paired runs and final aggregation are complete; [verified results](results.md).')
        plan.write_text(content)
    print(f"Complete paired MAE difference: {report['mean_paired_delta_subject_mae']:+.6f}",flush=True)


if __name__=='__main__':main()
