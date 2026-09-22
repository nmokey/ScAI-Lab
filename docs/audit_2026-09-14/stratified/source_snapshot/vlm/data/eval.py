"""
Evaluation metrics for the mouse trajectory VLM.
Genotype: head-based accuracy (primary) with text-match fallback.
TBR text: per-week MAE and Pearson correlation from parsed model text output.
TBR regression head: direct numeric MAE/r from the multitask regression head logits.
"""

import json
import re
import warnings
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


def _bootstrap_ci(pairs_by_group, stat_fn, n_boot=5000, seed=0, alpha=0.05):
    """
    Percentile bootstrap CI, resampling GROUPS (subjects) with replacement.
    pairs_by_group: {group: [(a, b), ...]}; stat_fn takes flat lists (as, bs).
    Subject-level because rows from one subject are not independent (F14).
    """
    import random
    groups = sorted(pairs_by_group)
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [rng.choice(groups) for _ in groups]
        a_s, b_s = [], []
        for g in pick:
            for a, b in pairs_by_group[g]:
                a_s.append(a); b_s.append(b)
        v = stat_fn(a_s, b_s)
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int((alpha / 2) * (len(vals) - 1))]
    hi = vals[int((1 - alpha / 2) * (len(vals) - 1))]
    return (lo, hi)


def _perm_p_auroc(labels, scores, n_perm=10000, seed=0):
    """
    One-sided permutation p for AUROC with fixed scores (F1's estimator null):
    fraction of random relabellings scoring at least as high. Returns
    (p, null_lo, null_hi). One record per subject here, so row = subject.
    """
    import random
    obs = _auroc(labels, scores)
    if obs != obs:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    lab = list(labels)
    null = []
    for _ in range(n_perm):
        rng.shuffle(lab)
        v = _auroc(lab, scores)
        if v == v:
            null.append(v)
    if not null:
        return (float("nan"), float("nan"), float("nan"))
    null.sort()
    p = (sum(1 for v in null if v >= obs) + 1) / (len(null) + 1)
    return (p, null[int(0.025 * (len(null) - 1))], null[int(0.975 * (len(null) - 1))])


def _route(question, content_type=None):
    """
    Classify a record as genotype / tbr / combined.

    Prefers `content_type` ("genotype" | "tbr" | "combined", set by
    create_mouse_traj_dataset.py and threaded through to predictions
    unchanged) since it is ground truth and immune to question wording.

    Falls back to text heuristics (case-normalised) only when content_type
    is absent, e.g. prediction files from before F15 added the field. That
    fallback is still unreliable against the actual GPT-augmented question
    bank in crump_aug_dataset.csv: ~36/44 "combined" paraphrases and 6/58
    "genotype" paraphrases use synonyms ("outcome", "classification", ...)
    instead of the literal word "status", so they'd be misrouted or dropped
    under text-only matching. content_type is what makes routing correct
    for that bank -- see FINDINGS.md F21.
    """
    if content_type in ("genotype", "tbr", "combined"):
        return "geno" if content_type == "genotype" else content_type
    q        = question.lower()
    has_tbr  = "tbr" in q
    has_geno = "status" in q
    if has_tbr and has_geno:
        return "combined"
    if has_tbr:
        return "tbr"
    if has_geno:
        return "geno"
    return None


def calculate_mouse_metrics(gt_file, train_gt_file, pred_file, out_file):
    with open(pred_file) as f:
        preds = json.load(f)

    if any(p.get("target_schema_version") == 1 for p in preds):
        from data.validated_metrics import score_predictions
        from utils.research_io import atomic_json
        if not gt_file:
            raise ValueError("Validated scoring requires the explicit ground-truth population")
        with open(gt_file) as f:
            truth = json.load(f)
        training = None
        if train_gt_file:
            with open(train_gt_file) as f:
                training = json.load(f)
        results = score_predictions(preds, truth, training)
        atomic_json(out_file, results)
        print(f"Validated scoring: subject MAE={results['tbr_reg_subject_mae']}, "
              f"genotype AUROC={results['genotype_auc']}; saved {out_file}")
        return results

    total = 0
    unrouted = 0

    # --- Genotype: head-based (primary) ---
    # Records that have a saved genotype_logit come from the multitask head.
    # Records without it fall back to text matching.
    head_correct, head_n   = 0, 0
    text_correct, text_n   = 0, 0
    geno_labels, geno_scores = [], []  # for head-based AUROC (label, raw logit)

    # Text-TBR pairs are keyed by (week, pid) then averaged per subject, so a
    # subject that answers both a "tbr" and a "combined" question contributes
    # one observation per week rather than two (see the regression-head note
    # below -- the same double-counting applied here).
    tbr_obs = defaultdict(lambda: defaultdict(lambda: {"gt": [], "pred": []}))  # week -> pid -> lists
    tbr_q_n, combined_q_n = 0, 0

    for p in preds:
        answer    = p.get("answer", "")
        model_ans = p.get("model_answer", "")
        question  = p.get("orig_question", p.get("question", ""))
        pid       = p.get("pid")
        total += 1

        route = _route(question, p.get("content_type"))
        if route is None:
            unrouted += 1
            continue
        is_geno     = route == "geno"
        is_tbr      = route == "tbr"
        is_combined = route == "combined"

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
                    tbr_obs[wk][pid]["gt"].append(gt_val)
                    tbr_obs[wk][pid]["pred"].append(pred_tbr[wk])

    if unrouted:
        warnings.warn(
            f"{unrouted}/{total} prediction records matched neither the genotype "
            f"nor the TBR question pattern and were excluded from every metric. "
            f"Check the question wording against _route().",
            RuntimeWarning, stacklevel=2,
        )

    # Collapse per-subject repeats: one (gt, pred) per (week, subject).
    tbr_pairs = defaultdict(lambda: {"gt": [], "pred": []})
    for wk, per_pid in tbr_obs.items():
        for _pid, vals in per_pid.items():
            tbr_pairs[wk]["gt"].append(sum(vals["gt"]) / len(vals["gt"]))
            tbr_pairs[wk]["pred"].append(sum(vals["pred"]) / len(vals["pred"]))

    # Genotype accuracy. Head-based and text-match accuracies are reported
    # separately -- previously they were summed into one number while
    # `genotype_acc_source` named only one of them.
    geno_n          = head_n + text_n
    correct_geno    = head_correct + text_correct
    head_based      = head_n > 0
    # AUROC is only defined for the head-based scores (needs continuous logits + both classes)
    geno_auc        = _auroc(geno_labels, geno_scores) if head_based else float("nan")
    # F1: never report an AUROC at n=32 without its interval and a permutation p.
    if head_based and geno_auc == geno_auc:
        _g = {i: [(l, sc)] for i, (l, sc) in enumerate(zip(geno_labels, geno_scores))}
        geno_auc_ci = _bootstrap_ci(_g, lambda a, b: _auroc(a, b))
        geno_perm_p, geno_null_lo, geno_null_hi = _perm_p_auroc(geno_labels, geno_scores)
    else:
        geno_auc_ci = (float("nan"), float("nan"))
        geno_perm_p = geno_null_lo = geno_null_hi = float("nan")

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
    # Every subject emits BOTH a "tbr" and a "combined" record carrying the same
    # tbr_targets, so iterating records directly counted each subject twice --
    # 134 pairs from 67 independent observations. That barely moves the point
    # estimates but halves the effective n, which matters for any bootstrap CI
    # or permutation test computed downstream. Group by (slot, pid) and average
    # the head's predictions so each subject contributes exactly one observation.
    reg_obs = defaultdict(lambda: defaultdict(lambda: {"gt": [], "pred": []}))  # slot -> pid -> lists
    for p in preds:
        reg  = p.get("tbr_regression")
        tgts = p.get("tbr_targets")
        if reg is None or tgts is None:
            continue
        pid = p.get("pid")
        for slot, (pred_val, gt_val) in enumerate(zip(reg, tgts)):
            if gt_val != -1:
                reg_obs[slot][pid]["gt"].append(gt_val)
                reg_obs[slot][pid]["pred"].append(pred_val)

    reg_pairs = defaultdict(lambda: {"gt": [], "pred": []})  # slot index -> lists
    for slot, per_pid in reg_obs.items():
        for _pid, vals in per_pid.items():
            gts = vals["gt"]
            if max(gts) - min(gts) > 1e-6:
                warnings.warn(
                    f"Subject {_pid} has conflicting TBR targets for slot {slot}: "
                    f"{gts}. Averaging, but the dataset records disagree.",
                    RuntimeWarning, stacklevel=2,
                )
            reg_pairs[slot]["gt"].append(sum(gts) / len(gts))
            reg_pairs[slot]["pred"].append(sum(vals["pred"]) / len(vals["pred"]))

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

    # F1: subject-level bootstrap for the pooled regression metrics.
    _by_pid = defaultdict(list)
    for slot, per_pid in reg_obs.items():
        for _pid, vals in per_pid.items():
            _by_pid[_pid].append((sum(vals["gt"]) / len(vals["gt"]), sum(vals["pred"]) / len(vals["pred"])))
    reg_r_ci   = _bootstrap_ci(_by_pid, lambda a, b: _pearson(a, b))    if len(_by_pid) >= 2 else (None, None)
    reg_mae_ci = _bootstrap_ci(_by_pid, lambda a, b: sum(abs(x - y) for x, y in zip(a, b)) / len(a)) if len(_by_pid) >= 2 else (None, None)

    reg_overall_mae = (sum(abs(g - p) for g, p in zip(reg_all_gt, reg_all_pred)) / len(reg_all_gt)
                       if reg_all_gt else None)
    reg_overall_r   = _pearson(reg_all_gt, reg_all_pred) if len(reg_all_gt) >= 2 else None
    reg_overall_r2  = _r2(reg_all_gt, reg_all_pred)      if len(reg_all_gt) >= 2 else None

    results = {
        "analysis_status":          "historical_protocol_not_validated",
        "uncertainty_scope":        "Fixed-prediction resampling only; label shuffling is not a full-refit pipeline test.",
        "total":                    total,
        "unrouted_records":         unrouted,
        "genotype_acc":             round(correct_geno / max(geno_n, 1), 3),
        # Matching question context fixes supervision; calibration is not thereby established.
        "genotype_acc_calibrated":  None,
        "genotype_calibration_status": "not_assessed",
        "genotype_primary_metric":  "genotype_auc",
        "genotype_auc":             round(geno_auc, 3) if geno_auc == geno_auc else None,
        "genotype_auc_ci95":        [round(v, 3) for v in geno_auc_ci] if geno_auc_ci[0] == geno_auc_ci[0] else None,
        "genotype_auc_perm_p":      round(geno_perm_p, 4) if geno_perm_p == geno_perm_p else None,
        "genotype_auc_null95":      [round(geno_null_lo, 3), round(geno_null_hi, 3)] if geno_null_lo == geno_null_lo else None,
        "genotype_n":               geno_n,
        "genotype_acc_source":      "multitask_head" if head_based else "text_match",
        "genotype_head_acc":        round(head_correct / head_n, 3) if head_n else None,
        "genotype_text_acc":        round(text_correct / text_n, 3) if text_n else None,
        "genotype_head_n":          head_n,
        "genotype_text_n":          text_n,
        "tbr_n":                    tbr_q_n,
        "combined_n":               combined_q_n,
        "tbr_overall_mae":          round(overall_mae, 3) if all_gt else None,
        "tbr_overall_r":            round(overall_r, 3)   if len(all_gt) >= 2 else None,
        "tbr_overall_r2":           round(overall_r2, 3)  if len(all_gt) >= 2 else None,
        "tbr_by_week":              tbr_results,
        "tbr_reg_overall_mae":      round(reg_overall_mae, 3) if reg_overall_mae is not None else None,
        "tbr_reg_overall_mae_ci95": [round(v, 3) for v in reg_mae_ci] if reg_mae_ci[0] is not None and reg_mae_ci[0] == reg_mae_ci[0] else None,
        "tbr_reg_overall_r_ci95":   [round(v, 3) for v in reg_r_ci]   if reg_r_ci[0]   is not None and reg_r_ci[0]   == reg_r_ci[0]   else None,
        "tbr_reg_n_subjects":       len(_by_pid),
        "tbr_reg_overall_r":        round(reg_overall_r, 3)   if reg_overall_r  is not None else None,
        "tbr_reg_overall_r2":       round(reg_overall_r2, 3)  if reg_overall_r2 is not None else None,
        "tbr_reg_by_slot":          reg_results,
    }

    print("\n=== Mouse Trajectory VLM Evaluation ===")
    src = "multitask head" if head_based else "text match"
    if unrouted:
        print(f"  [!] {unrouted}/{total} records did not route to any question type")
    if results["genotype_auc"] is not None:
        ci = results["genotype_auc_ci95"]; nl = results["genotype_auc_null95"]
        print(f"  Genotype AUROC    : {results['genotype_auc']:.3f}  95% CI {ci}  "
              f"perm p={results['genotype_auc_perm_p']}  no-signal 95% {nl}  (n={head_n})"
              f"  <- primary genotype metric")
    print(f"  Genotype accuracy : {results['genotype_acc']:.3f}  (n={geno_n}, source={src}, "
          f"UNCALIBRATED threshold — see F3)")
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
        print(f"  TBR (reg head) MAE: {results['tbr_reg_overall_mae']:.3f}  95% CI {results['tbr_reg_overall_mae_ci95']}"
              f"  (n={len(reg_all_gt)} obs, {results['tbr_reg_n_subjects']} subjects)")
        print(f"  TBR (reg head) r  : {results['tbr_reg_overall_r']}  95% CI {results['tbr_reg_overall_r_ci95']}")
        print(f"  TBR (reg head) R² : {results['tbr_reg_overall_r2']}")
        for slot, m in reg_results.items():
            print(f"    {slot}: MAE={m['mae']:.3f}, r={m['pearson_r']}, R²={m['r2']}, n={m['n']}")
    else:
        print(f"  TBR reg head      : no predictions saved (multitask head may be off)")
    print(f"  Combined questions: {combined_q_n}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_file}")
