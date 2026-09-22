"""Independently reproduce amended metrics using only the Python standard library."""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def check_run(directory, truth, split):
    subject_truth = {}
    for row in truth:
        entry = subject_truth.setdefault(row['pid'], {'label':row['answer_vqa_numeric']['genotype'], 'targets':[-1.]*4})
        for i, value in enumerate(row['answer_vqa_numeric']['tbr']):
            if value >= 0:
                entry['targets'][i] = value
    total_pairs = 0; wins = 0.; correct = 0; majority_correct = 0
    subject_errors = {}; mean_errors = {}; folds = []
    for fold in split['folds']:
        rows = json.loads((directory / fold['key'] / 'vqa.json').read_text())
        expected = {(r['pid'], str(r['qid'])) for r in truth if r['pid'] in fold['test_subjects']}
        actual = [(r['pid'], str(r['qid'])) for r in rows]
        assert len(actual) == len(set(actual)) and set(actual) == expected
        genotype = [r for r in rows if r['content_type'] == 'genotype']
        assert len(genotype) == len(fold['test_subjects'])
        assert all(r['genotype_label'] == subject_truth[r['pid']]['label'] for r in genotype)
        assert all(math.isfinite(r['genotype_logit']) for r in genotype)
        pos = [r['genotype_logit'] for r in genotype if r['genotype_label']==1]
        neg = [r['genotype_logit'] for r in genotype if r['genotype_label']==0]
        pairs = len(pos) * len(neg); assert pairs
        fold_wins = sum(float(p>n) + .5*float(p==n) for p in pos for n in neg)
        total_pairs += pairs; wins += fold_wins
        correct += sum((r['genotype_logit']>0)==r['genotype_label'] for r in genotype)
        majority = int(sum(subject_truth[s]['label'] for s in fold['train_subjects']) > len(fold['train_subjects']) / 2)
        majority_correct += sum(r['genotype_label']==majority for r in genotype)
        means = []
        for slot in range(4):
            values = [subject_truth[s]['targets'][slot] for s in fold['train_subjects'] if subject_truth[s]['targets'][slot]>=0]
            means.append(statistics.mean(values) if values else 0.)
        predictions = defaultdict(lambda:defaultdict(list)); saved_means = {}
        for row in rows:
            sid = row['pid']; saved_means[sid] = row['train_mean_prediction']
            assert all(math.isclose(a,b,rel_tol=1e-6,abs_tol=2e-5) for a,b in zip(means,saved_means[sid]))
            if row['content_type'] in ('tbr','combined'):
                for slot, target in enumerate(subject_truth[sid]['targets'][:3]):
                    if target >= 0:
                        predictions[sid][slot].append(row['tbr_regression'][slot])
        for sid, slots in predictions.items():
            subject_errors[sid] = statistics.mean(abs(statistics.mean(values)-subject_truth[sid]['targets'][slot])
                                                   for slot,values in slots.items())
            mean_errors[sid] = statistics.mean(abs(saved_means[sid][slot]-subject_truth[sid]['targets'][slot]) for slot in slots)
        folds.append(dict(key=fold['key'], pairs=pairs, wins=fold_wins, auc=fold_wins/pairs))
    n = len(subject_truth)
    expected = dict(genotype_auc_within_fold_pair_weighted=wins/total_pairs,
                    genotype_accuracy=correct/n, training_majority_accuracy=majority_correct/n,
                    subject_mae=statistics.mean(subject_errors.values()),
                    train_mean_subject_mae=statistics.mean(mean_errors.values()))
    saved = json.loads((directory / 'stratified_results.json').read_text())
    for key, value in expected.items():
        if not math.isclose(saved[key], value, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f'{directory.name}: independent {key} disagrees')
    assert total_pairs == saved['genotype_pairs'] == 62
    for fold, original in zip(folds, saved['folds']):
        assert fold['key'] == original['key']
        assert math.isclose(fold['auc'], original['genotype_auc'], abs_tol=1e-12)
    return dict(run=str(directory), status='passed', **expected, folds=folds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True); p.add_argument('--truth', required=True)
    p.add_argument('--split', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    truth = json.loads(Path(args.truth).read_text()); split = json.loads(Path(args.split).read_text())
    checks = [check_run(Path(args.runs) / f'stratified_{arm}_seed{seed}', truth, split)
              for arm in ('base','long') for seed in (0,1,2)]
    report = dict(status='passed', checks=checks, method='Independent standard-library arithmetic; no production scorer imported')
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print('All six runs: independent within-fold pair counting, accuracy and proxy-MAE checks passed')


if __name__ == '__main__':
    main()
