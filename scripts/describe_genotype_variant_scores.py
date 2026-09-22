"""Describe verified prediction confidence without fitting or selecting thresholds."""
import hashlib
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'docs/audit_2026-09-14/genotype_vlm_variants'
RUNS = Path('/data1/Processed_NIfTI_Test/embeddings/vlm/runs')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability(logit):
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    value = math.exp(logit)
    return value / (1 + value)


def describe(rows):
    scores = [r['genotype_logit'] for r in rows]
    assert all(math.isfinite(x) for x in scores)
    probabilities = [probability(x) for x in scores]
    correct = [(r['genotype_logit'] > 0) == r['genotype_label'] for r in rows]
    mistaken_confidence = [max(p, 1-p) for p, ok in zip(probabilities, correct) if not ok]
    return dict(n=len(rows), correct=sum(correct), predicted_ko=sum(x > 0 for x in scores),
                probability_min=min(probabilities), probability_max=max(probabilities),
                probability_04_to_06=sum(.4 <= p <= .6 for p in probabilities),
                confidence_at_least_09=sum(max(p, 1-p) >= .9 for p in probabilities),
                mistaken_mean_confidence=statistics.mean(mistaken_confidence) if mistaken_confidence else None)


def main():
    completion_path = OUTPUT / 'completion.json'
    completion = json.loads(completion_path.read_text())
    assert completion['status'] == 'complete'
    for name, digest in completion['files_sha256'].items():
        assert sha(Path(name)) == digest, f'Completed result changed: {name}'
    protocol = json.loads((OUTPUT / 'PROTOCOL.json').read_text())
    results = json.loads((OUTPUT / 'paired_results.json').read_text())
    entries = []; sources = {}
    for seed in protocol['population']['seeds']:
        reference_path = RUNS / f'stratified_residual_seed{seed}/vqa_stratified.json'
        reference = {r['pid']: r for r in json.loads(reference_path.read_text()) if r['content_type'] == 'genotype'}
        sources[str(reference_path)] = sha(reference_path)
        for arm in ('base', 'residual', 'direct_visual', 'genotype_only'):
            path = RUNS / f'stratified_{arm}_seed{seed}/vqa_stratified.json'
            sources[str(path)] = sha(path)
            rows = [r for r in json.loads(path.read_text()) if r['content_type'] == 'genotype']
            assert len(rows) == len({r['pid'] for r in rows}) == 32
            assert {r['pid'] for r in rows} == set(reference)
            entry = dict(seed=seed, arm=arm, summary=describe(rows), folds=[])
            expected = next(r for r in results['per_seed'] if r['seed'] == seed)[arm]
            assert entry['summary']['correct'] == expected['classification']['correct']
            improved = []; worsened = []
            for r in rows:
                other = reference[r['pid']]
                assert r['genotype_label'] == other['genotype_label']
                correct = (r['genotype_logit'] > 0) == r['genotype_label']
                previous = (other['genotype_logit'] > 0) == other['genotype_label']
                if correct and not previous:
                    improved.append(r['pid'])
                if previous and not correct:
                    worsened.append(r['pid'])
            entry['versus_residual'] = dict(improved=sorted(improved), worsened=sorted(worsened))
            for fold in protocol['population']['folds']:
                subset = [r for r in rows if r['pid'] in fold['test_subjects']]
                assert len(subset) == 8
                row = dict(fold=fold['key'], **describe(subset), components={})
                for key in ('genotype_logit', 'language_genotype_logit', 'visual_genotype_logit'):
                    if key not in subset[0]:
                        continue
                    ko = [r[key] for r in subset if r['genotype_label'] == 1]
                    wt = [r[key] for r in subset if r['genotype_label'] == 0]
                    row['components'][key] = dict(mean=statistics.mean(r[key] for r in subset),
                        ko_mean=statistics.mean(ko), wt_mean=statistics.mean(wt),
                        ko_minus_wt_mean=statistics.mean(ko)-statistics.mean(wt))
                entry['folds'].append(row)
            entries.append(entry)
    report = dict(status='complete', completion_sha256=sha(completion_path), source_sha256=sha(Path(__file__)),
        method='Descriptive inspection of all completed predictions; sigmoid of saved logits; no fitting, threshold selection, inversion, or new model evaluation',
        limitation='Confidence means raw sigmoid score, not verified calibration. Seeds reuse the same mice; paired changes are descriptive, not independent observations.',
        prediction_sha256=sources, runs=entries)
    output = OUTPUT / 'score_diagnostics.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    for entry in entries:
        print(entry['arm'], entry['seed'], entry['summary'])


if __name__ == '__main__':
    main()
