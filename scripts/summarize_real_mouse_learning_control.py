"""Verify paired training-only capacity controls and write their interpretation."""
import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True)
    args=p.parse_args();root=Path(args.directory)
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    records=json.loads((root/'training_subset.json').read_text())
    labels={(r['pid'],str(r['qid'])):r for r in records}
    reports={k:json.loads((root/k/'report.json').read_text()) for k in ('combined','genotype_only')}
    for name,report in reports.items():
        assert report['status']=='complete' and report['objective']==name
        assert report['protocol']==protocol and report['updates']==300
        assert report['reload_max_logit_error']==0
        assert report['objective_check']['status']=='passed'
        for point in report['learning_curve']+[report['final']]:
            assert {(r['pid'],str(r['qid'])) for r in point['rows']}==set(labels)
            assert len(point['rows'])==len(labels)==24
            assert all(r['label']==labels[r['pid'],str(r['qid'])]['answer_vqa_numeric']['genotype'] for r in point['rows'])
            for kind in ('genotype','tbr','combined','all'):
                rows=[r for r in point['rows'] if kind=='all' or r['kind']==kind]
                s=[r['logit'] for r in rows];y=[r['label'] for r in rows]
                assert all(math.isfinite(v) for v in s)
                accuracy=statistics.mean((v>0)==label for v,label in zip(s,y))
                bce=statistics.mean(max(v,0)+math.log1p(math.exp(-abs(v)))-label*v for v,label in zip(s,y))
                assert math.isclose(accuracy,point['by_question'][kind]['accuracy'],abs_tol=1e-12)
                assert math.isclose(bce,point['by_question'][kind]['bce'],abs_tol=1e-12)
    a,b=reports['combined'],reports['genotype_only']
    assert a['initial_trainable_sha256']==b['initial_trainable_sha256']
    assert a['learning_curve'][0]==b['learning_curve'][0]
    assert a['target_statistics']==b['target_statistics']
    assert b['parameter_change_norms']['tbr_regression_head']==0
    assert a['parameter_change_norms']['tbr_regression_head']>0
    lines=['# Real-mouse learning control — training diagnostics only','',
           'Eight existing training mice (four KO, four WT), 24 question records, the longitudinal inputs, seed 0, identical initial parameters and 300 updates per objective. No held-out performance was measured. Both final checkpoints reproduce all control logits exactly after strict reload.','',
           'The control changes only the objective: the combined run uses the existing language + proxy + genotype losses; genotype-only retains the same weighted genotype loss and removes the other two. Both use the same batches, learning rate, schedule and extended training budget.','',
           '| Objective | Correct genotype labels | Mean genotype BCE | Lowest true-class score | Capacity criterion met |',
           '|---|---:|---:|---:|---|']
    for name,report in reports.items():
        f=report['final']['by_question']['genotype']
        lines.append(f"| {name} | {round(f['accuracy']*8)}/8 | {f['bce']:.6f} | {f['minimum_true_class_probability']:.4f} | {report['capacity_criterion_met']} |")
    lines+=['','The declared capacity criterion is all eight genotype-question labels correct at a zero-logit threshold and mean BCE ≤ 0.1. It is a debugging criterion, not a statistical test.','',
            '| Update | Combined correct / 8 | Combined BCE | Genotype-only correct / 8 | Genotype-only BCE |',
            '|---|---:|---:|---:|---:|']
    for x,y in zip(a['learning_curve'],b['learning_curve']):
        assert x['step']==y['step']
        u,v=x['by_question']['genotype'],y['by_question']['genotype']
        lines.append(f"| {x['step']} | {round(u['accuracy']*8)} | {u['bce']:.6f} | {round(v['accuracy']*8)} | {v['bce']:.6f} |")
    pa,pb=a['capacity_criterion_met'],b['capacity_criterion_met']
    if pa and pb:
        interpretation='Both objectives can learn this small set of actual image/label pairs. The combined objective does not prevent memorization under this extended budget. This weakens a gross training-plumbing failure as an explanation of the full-study result, but does not establish adequate optimization or generalization on the full dataset.'
    elif pb and not pa:
        interpretation='Genotype-only meets the learning criterion and the combined objective does not under the matched budget. This implicates an optimization interaction with the other objectives on this subset. It does not establish which competing component dominates or that changing the loss will improve held-out performance.'
    elif pa and not pb:
        interpretation='The combined objective meets the learning criterion and genotype-only does not. Removing the other supervision did not solve the capacity test under this schedule; those objectives may provide useful training signal. This is a subset-specific diagnostic, not a held-out comparison.'
    else:
        interpretation='Neither objective meets the declared criterion under the fixed control budget. This calls for further localization of optimization/input limitations; failure at one chosen schedule is not proof that the implementation is defective or that the images contain no usable information.'
    lines+=['',interpretation,'',
            'Passing this extended training-only control does not show that increasing the full-study epoch count would improve held-out performance.','',
            'The subset was fixed by sorted subject order before fitting. Its KO mice all belong to one acquisition component, so successful fitting could use acquisition-related cues. The controls establish only the ability to learn these selected labels. The original held-out scores, model checkpoints and training settings remain unchanged.','',
            'The 300-update budget corresponds to 100 passes through this tiny subset. The measurements at 60 updates correspond to 20 subset epochs, but the learning-rate schedule is defined over the full 300 updates; they are not a reproduction of the original 20-epoch experiment.','',
            '[Protocol](PROTOCOL.json), [input fingerprints](frozen_inputs.json), [combined report](combined/report.json), [genotype-only report](genotype_only/report.json). Each report includes learning curves, gradient checks, parameter changes and exact checkpoint-reload results.','']
    if (root/'precision_repair.json').exists():
        lines += ['## Measurement precision check','',
                  'The initial reload assertion compared the Trainer’s automatic BF16 forward/FP32 readout with plain inference precision. It therefore stopped after both models had already finished all 300 updates and saved their final weights. The control probe now makes its precision explicit. Replaying the saved models at the original probe precision reproduces every recorded endpoint logit exactly; their tensors also equal the checkpoint at update 300. No training was repeated or model weights changed. Ordinary inference separately retains 8/8 correct labels in both controls.','',
                  '[Precision repair evidence](precision_repair.json) preserves the measured discrepancy and checks. Original unfinished reports and execution source are retained; `run_real_mouse_learning_control_source.py` is the source corresponding to the launch-time fingerprint. The current script includes the measurement correction.','']
    if (root/'image_dependence.json').exists():
        check=json.loads((root/'image_dependence.json').read_text());assert check['status']=='passed'
        lines += ['## Image dependence','',
                  'Swapping the mouse image-token inputs makes the outputs follow the donor images, while leaving questions, supplied answer tokens, masks and labels unchanged. Giving every mouse identical image tokens removes between-mouse genotype-score differences. This confirms that the learned distinction depends on the image inputs. [Checks](image_dependence.json).','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    summary=dict(status='complete',diagnostic_only=True,independent_arithmetic='passed',
                 identical_initial_parameters=True,identical_initial_predictions=True,
                 interpretation=interpretation,final={name:r['final']['by_question']['genotype'] for name,r in reports.items()})
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
