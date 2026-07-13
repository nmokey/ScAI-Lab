"""
Evaluation metrics for the mouse trajectory VLM.
Genotype: head-based accuracy (primary) with text-match fallback.
TBR text: per-week MAE and Pearson correlation from parsed model text output.
TBR regression head: direct numeric MAE/r from the multitask regression head logits.
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


def _auroc(labels, scores):
    """Binary AUROC via the rank-based (Mann-Whitney U) formula, with tie handling.

    labels: 0/1 ground-truth; scores: continuous decision values (higher → class 1).
    The genotype head emits one logit, a monotonic function of P(KO), so the raw
    logit works directly as the score. Returns NaN if either class is absent.
    """
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks (1-based), tie groups share the mean of their positions.
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # mean of positions i..j, converted to 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(r for r, lbl in zip(ranks, labels) if lbl == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calculate_mouse_metrics(gt_file, train_gt_file, pred_file, out_file):
    with open(pred_file) as f:
        preds = json.load(f)

    total = 0

    # --- Genotype: head-based (primary) ---
    # Records that have a saved genotype_logit come from the multitask head.
    # Records without it fall back to text matching.
    head_correct, head_n   = 0, 0
    text_correct, text_n   = 0, 0
    geno_labels, geno_scores = [], []  # for head-based AUROC (label, raw logit)

    tbr_pairs = defaultdict(lambda: {"gt": [], "pred": []})  # week -> lists
    tbr_q_n, combined_q_n = 0, 0

    for p in preds:
        answer    = p.get("answer", "")
        model_ans = p.get("model_answer", "")
        question  = p.get("orig_question", p.get("question", ""))
        total += 1

        is_geno     = "status" in question.lower() and "TBR" not in question
        is_tbr      = "TBR" in question and "status" not in question.lower()
        is_combined = "TBR" in question and "status" in question.lower()

        if is_geno:
            raw_gl = p.get("genotype_label")
            if raw_gl is None:
                continue
            gt_label = int(raw_gl)

            if "genotype_logit" in p:
                # Primary: sigmoid(logit) > 0.5 → KO
                pred_label = 1 if p["genotype_logit"] > 0.0 else 0
                head_correct += int(gt_label == pred_label)
                head_n += 1
                geno_labels.append(gt_label)
                geno_scores.append(p["genotype_logit"])
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
    # AUROC is only defined for the head-based scores (needs continuous logits + both classes)
    geno_auc        = _auroc(geno_labels, geno_scores) if head_based else float("nan")

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

    # --- TBR regression head (direct numeric output) ---
    # tbr_regression is a list of up to 4 floats (future-week order: Δ3, Δ6, Δ8, pad=-1).
    # tbr_targets mirrors the same layout. Only positions where target != -1 are valid.
    reg_pairs = defaultdict(lambda: {"gt": [], "pred": []})  # slot index -> lists
    for p in preds:
        reg  = p.get("tbr_regression")
        tgts = p.get("tbr_targets")
        if reg is None or tgts is None:
            continue
        for slot, (pred_val, gt_val) in enumerate(zip(reg, tgts)):
            if gt_val != -1:
                reg_pairs[slot]["gt"].append(gt_val)
                reg_pairs[slot]["pred"].append(pred_val)

    # Slot indices map to relative-week offsets: 0→Δ3wk, 1→Δ6wk, 2→Δ8wk
    _SLOT_LABEL = {0: "delta_3wk", 1: "delta_6wk", 2: "delta_8wk", 3: "delta_pad"}
    reg_results = {}
    reg_all_gt, reg_all_pred = [], []
    for slot in sorted(reg_pairs):
        gt   = reg_pairs[slot]["gt"]
        pred = reg_pairs[slot]["pred"]
        mae  = sum(abs(g - p) for g, p in zip(gt, pred)) / len(gt)
        r    = _pearson(gt, pred)
        r2   = _r2(gt, pred)
        label = _SLOT_LABEL.get(slot, f"slot_{slot}")
        reg_results[label] = {
            "mae": round(mae, 3), "pearson_r": round(r, 3), "r2": round(r2, 3), "n": len(gt)
        }
        reg_all_gt.extend(gt)
        reg_all_pred.extend(pred)

    reg_overall_mae = (sum(abs(g - p) for g, p in zip(reg_all_gt, reg_all_pred)) / len(reg_all_gt)
                       if reg_all_gt else None)
    reg_overall_r   = _pearson(reg_all_gt, reg_all_pred) if len(reg_all_gt) >= 2 else None
    reg_overall_r2  = _r2(reg_all_gt, reg_all_pred)      if len(reg_all_gt) >= 2 else None

    results = {
        "total":                    total,
        "genotype_acc":             round(correct_geno / max(geno_n, 1), 3),
        "genotype_auc":             round(geno_auc, 3) if geno_auc == geno_auc else None,
        "genotype_n":               geno_n,
        "genotype_acc_source":      "multitask_head" if head_based else "text_match",
        "genotype_head_n":          head_n,
        "genotype_text_n":          text_n,
        "tbr_n":                    tbr_q_n,
        "combined_n":               combined_q_n,
        "tbr_overall_mae":          round(overall_mae, 3) if all_gt else None,
        "tbr_overall_r":            round(overall_r, 3)   if len(all_gt) >= 2 else None,
        "tbr_overall_r2":           round(overall_r2, 3)  if len(all_gt) >= 2 else None,
        "tbr_by_week":              tbr_results,
        "tbr_reg_overall_mae":      round(reg_overall_mae, 3) if reg_overall_mae is not None else None,
        "tbr_reg_overall_r":        round(reg_overall_r, 3)   if reg_overall_r  is not None else None,
        "tbr_reg_overall_r2":       round(reg_overall_r2, 3)  if reg_overall_r2 is not None else None,
        "tbr_reg_by_slot":          reg_results,
    }

    print("\n=== Mouse Trajectory VLM Evaluation ===")
    src = "multitask head" if head_based else "text match"
    print(f"  Genotype accuracy : {results['genotype_acc']:.3f}  (n={geno_n}, source={src})")
    if results["genotype_auc"] is not None:
        print(f"  Genotype AUROC    : {results['genotype_auc']:.3f}  (head-based, n={head_n}, chance=0.5)")
    if head_based and text_n > 0:
        print(f"    head-based: {head_correct}/{head_n}  text-match: {text_correct}/{text_n}")
    if all_gt:
        print(f"  TBR (text) MAE    : {results['tbr_overall_mae']:.3f}  (n={len(all_gt)} predictions)")
        print(f"  TBR (text) r      : {results['tbr_overall_r']}")
        print(f"  TBR (text) R²     : {results['tbr_overall_r2']}")
        for wk, m in tbr_results.items():
            print(f"    {wk}: MAE={m['mae']:.3f}, r={m['pearson_r']}, R²={m['r2']}, n={m['n']}")
    else:
        print(f"  TBR questions     : {tbr_q_n}  (no parseable predictions)")
    if reg_all_gt:
        print(f"  TBR (reg head) MAE: {results['tbr_reg_overall_mae']:.3f}  (n={len(reg_all_gt)} predictions)")
        print(f"  TBR (reg head) r  : {results['tbr_reg_overall_r']}")
        print(f"  TBR (reg head) R² : {results['tbr_reg_overall_r2']}")
        for slot, m in reg_results.items():
            print(f"    {slot}: MAE={m['mae']:.3f}, r={m['pearson_r']}, R²={m['r2']}, n={m['n']}")
    else:
        print(f"  TBR reg head      : no predictions saved (multitask head may be off)")
    print(f"  Combined questions: {combined_q_n}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_file}")
