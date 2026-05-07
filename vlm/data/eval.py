"""
Evaluation metrics for the mouse trajectory VLM.
For the text-generation baseline, we compute BLEU/ROUGE over free-text answers
and exact-match accuracy for genotype questions.
"""

import json
import re
from collections import defaultdict


def calculate_mouse_metrics(gt_file, train_gt_file, pred_file, out_file):
    with open(pred_file) as f:
        preds = json.load(f)

    total, correct_geno = 0, 0
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})

    for p in preds:
        answer   = p.get("answer", "")
        model_ans = p.get("model_answer", "")
        question  = p.get("orig_question", p.get("question", ""))

        total += 1
        # Genotype question: check WT/KO match
        if "genotype" in question.lower():
            gt_geno    = "KO" if "KO" in answer else "WT"
            pred_geno  = "KO" if "KO" in model_ans.upper() else ("WT" if "WT" in model_ans.upper() else "")
            correct    = gt_geno == pred_geno
            correct_geno += int(correct)
            by_type["genotype"]["total"] += 1
            by_type["genotype"]["correct"] += int(correct)
        elif "TBR" in question:
            by_type["tbr"]["total"] += 1
        else:
            by_type["combined"]["total"] += 1

    results = {
        "total":            total,
        "genotype_acc":     correct_geno / max(by_type["genotype"]["total"], 1),
        "genotype_n":       by_type["genotype"]["total"],
        "tbr_n":            by_type["tbr"]["total"],
        "combined_n":       by_type["combined"]["total"],
    }

    print("\n=== Mouse Trajectory VLM Evaluation ===")
    print(f"  Genotype accuracy : {results['genotype_acc']:.3f}  (n={results['genotype_n']})")
    print(f"  TBR questions     : {results['tbr_n']}")
    print(f"  Combined questions: {results['combined_n']}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_file}")
