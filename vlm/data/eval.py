"""
Evaluation metrics for the mouse trajectory VLM.
Genotype: head-based accuracy (primary) with text-match fallback.
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


def _r2(ys_true, ys_pred):
    n = len(ys_true)
    if n < 2:
        return float("nan")
    mean_true = sum(ys_true) / n
    ss_tot = sum((y - mean_true) ** 2 for y in ys_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(ys_true, ys_pred))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def calculate_mouse_metrics(gt_file, train_gt_file, pred_file, out_file):
    with open(pred_file) as f:
        preds = json.load(f)

    total = 0

    # --- Genotype: head-based (primary) ---
    # Records that have a saved genotype_logit come from the multitask head.
    # Records without it fall back to text matching.
    head_correct, head_n   = 0, 0
    text_correct, text_n   = 0, 0

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
            gt_label = 1 if "KO" in answer else 0

            if "genotype_logit" in p and p.get("genotype_label") is not None:
                # Primary: sigmoid(logit) > 0.5 → KO
                pred_label = 1 if p["genotype_logit"] > 0.0 else 0
                head_correct += int(gt_label == pred_label)
                head_n += 1
            else:
                # Fallback: text matching
                pred_geno = "KO" if "KO" in model_ans.upper() else ("WT" if "WT" in model_ans.upper() else "")
                gt_geno   = "KO" if gt_label == 1 else "WT"
                text_correct += int(gt_geno == pred_geno)
                text_n += 1

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

    # Genotype accuracy: head-based when available, text-match otherwise
    geno_n          = head_n + text_n
    correct_geno    = head_correct + text_correct
    head_based      = head_n > 0

    # Per-week TBR metrics
    tbr_results = {}
    all_gt, all_pred = [], []
    for wk in sorted(tbr_pairs):
        gt   = tbr_pairs[wk]["gt"]
        pred = tbr_pairs[wk]["pred"]
        mae  = sum(abs(g - p) for g, p in zip(gt, pred)) / len(gt)
        r    = _pearson(gt, pred)
        r2   = _r2(gt, pred)
        tbr_results[f"week_{wk}"] = {
            "mae": round(mae, 3), "pearson_r": round(r, 3), "r2": round(r2, 3), "n": len(gt)
        }
        all_gt.extend(gt)
        all_pred.extend(pred)

    overall_mae = (sum(abs(g - p) for g, p in zip(all_gt, all_pred)) / len(all_gt)
                   if all_gt else float("nan"))
    overall_r   = _pearson(all_gt, all_pred)
    overall_r2  = _r2(all_gt, all_pred)

    results = {
        "total":                total,
        "genotype_acc":         round(correct_geno / max(geno_n, 1), 3),
        "genotype_n":           geno_n,
        "genotype_acc_source":  "multitask_head" if head_based else "text_match",
        "genotype_head_n":      head_n,
        "genotype_text_n":      text_n,
        "tbr_n":                tbr_q_n,
        "combined_n":           combined_q_n,
        "tbr_overall_mae":      round(overall_mae, 3) if all_gt else None,
        "tbr_overall_r":        round(overall_r, 3)   if len(all_gt) >= 2 else None,
        "tbr_overall_r2":       round(overall_r2, 3)  if len(all_gt) >= 2 else None,
        "tbr_by_week":          tbr_results,
    }

    print("\n=== Mouse Trajectory VLM Evaluation ===")
    src = "multitask head" if head_based else "text match"
    print(f"  Genotype accuracy : {results['genotype_acc']:.3f}  (n={geno_n}, source={src})")
    if head_based and text_n > 0:
        print(f"    head-based: {head_correct}/{head_n}  text-match: {text_correct}/{text_n}")
    if all_gt:
        print(f"  TBR overall MAE   : {results['tbr_overall_mae']:.3f}  (n={len(all_gt)} predictions)")
        print(f"  TBR overall r     : {results['tbr_overall_r']}")
        print(f"  TBR overall R²    : {results['tbr_overall_r2']}")
        for wk, m in tbr_results.items():
            print(f"    {wk}: MAE={m['mae']:.3f}, r={m['pearson_r']}, R²={m['r2']}, n={m['n']}")
    else:
        print(f"  TBR questions     : {tbr_q_n}  (no parseable predictions)")
    print(f"  Combined questions: {combined_q_n}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_file}")
