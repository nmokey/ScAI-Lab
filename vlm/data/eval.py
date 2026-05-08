"""
Evaluation metrics for the mouse trajectory VLM.
Genotype: exact WT/KO match accuracy.
TBR: per-week MAE and Pearson correlation between predicted and ground-truth values.
"""

import json
import re
from collections import defaultdict
from math import sqrt


_TBR_RE = re.compile(r"Week\s+(\d+):\s*(\d+(?:\.\d+)?)")


def _parse_tbr(text):
    """Extract {week_int: float} from model output or ground truth text."""
    return {int(m.group(1)): float(m.group(2)) for m in _TBR_RE.finditer(text)}


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sqrt(sum((x - mx) ** 2 for x in xs)) * sqrt(sum((y - my) ** 2 for y in ys))
    return num / den if den > 0 else float("nan")


def calculate_mouse_metrics(gt_file, train_gt_file, pred_file, out_file):
    with open(pred_file) as f:
        preds = json.load(f)

    total, correct_geno, geno_n = 0, 0, 0
    tbr_pairs = defaultdict(lambda: {"gt": [], "pred": []})  # week -> lists
    tbr_q_n, combined_q_n = 0, 0

    for p in preds:
        answer    = p.get("answer", "")
        model_ans = p.get("model_answer", "")
        question  = p.get("orig_question", p.get("question", ""))
        total += 1

        is_geno     = "genotype" in question.lower() and "TBR" not in question
        is_tbr      = "TBR" in question and "genotype" not in question.lower()
        is_combined = "TBR" in question and "genotype" in question.lower()

        if is_geno:
            gt_geno   = "KO" if "KO" in answer else "WT"
            pred_geno = "KO" if "KO" in model_ans.upper() else ("WT" if "WT" in model_ans.upper() else "")
            correct_geno += int(gt_geno == pred_geno)
            geno_n += 1

        if is_tbr or is_combined:
            if is_tbr:
                tbr_q_n += 1
            else:
                combined_q_n += 1
            gt_tbr   = _parse_tbr(answer)
            pred_tbr = _parse_tbr(model_ans)
            for wk, gt_val in gt_tbr.items():
                if wk in pred_tbr:
                    tbr_pairs[wk]["gt"].append(gt_val)
                    tbr_pairs[wk]["pred"].append(pred_tbr[wk])

    # Per-week TBR metrics
    tbr_results = {}
    all_gt, all_pred = [], []
    for wk in sorted(tbr_pairs):
        gt   = tbr_pairs[wk]["gt"]
        pred = tbr_pairs[wk]["pred"]
        mae  = sum(abs(g - p) for g, p in zip(gt, pred)) / len(gt)
        r    = _pearson(gt, pred)
        tbr_results[f"week_{wk}"] = {"mae": round(mae, 3), "pearson_r": round(r, 3), "n": len(gt)}
        all_gt.extend(gt)
        all_pred.extend(pred)

    overall_mae = (sum(abs(g - p) for g, p in zip(all_gt, all_pred)) / len(all_gt)
                   if all_gt else float("nan"))
    overall_r   = _pearson(all_gt, all_pred)

    results = {
        "total":            total,
        "genotype_acc":     round(correct_geno / max(geno_n, 1), 3),
        "genotype_n":       geno_n,
        "tbr_n":            tbr_q_n,
        "combined_n":       combined_q_n,
        "tbr_overall_mae":  round(overall_mae, 3) if all_gt else None,
        "tbr_overall_r":    round(overall_r, 3)   if len(all_gt) >= 2 else None,
        "tbr_by_week":      tbr_results,
    }

    print("\n=== Mouse Trajectory VLM Evaluation ===")
    print(f"  Genotype accuracy : {results['genotype_acc']:.3f}  (n={geno_n})")
    if all_gt:
        print(f"  TBR overall MAE   : {results['tbr_overall_mae']:.3f}  (n={len(all_gt)} predictions)")
        print(f"  TBR overall r     : {results['tbr_overall_r']}")
        for wk, m in tbr_results.items():
            print(f"    {wk}: MAE={m['mae']:.3f}, r={m['pearson_r']}, n={m['n']}")
    else:
        print(f"  TBR questions     : {tbr_q_n}  (no parseable predictions)")
    print(f"  Combined questions: {combined_q_n}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_file}")
