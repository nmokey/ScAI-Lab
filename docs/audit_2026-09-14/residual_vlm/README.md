# Residual embeddings through the VLM

**Complete:** all 12 residual-input fits and all nine run-level artifact/independent-scoring checks passed. [Full results](RESULTS.md), [independent metrics](independent_metrics.json), and [completion hashes](completion.json).

Residual VLM genotype AUROC is **0.5484 in all three seeds**, exceeding the matching baseline (0.5161/0.4839/0.4355) and current longitudinal VLM (0.4194/0.4516/0.4516) each time. Predictions differ across seeds: the identical aggregate AUROC comes from 34 correctly ordered KO-WT pairs out of 62, with different contributions across folds. This is a consistent descriptive gain over the VLM comparators, but the absolute discrimination remains weak and below the residual linear probe's 0.6774. Accuracy is 17/32, 16/32, and 17/32, below the always-WT comparator's 18/32 in every seed.

Mean subject-level proxy MAE is **5.6564**, versus baseline **5.5506**, current longitudinal **5.6631**, and the training-mean predictor **5.6106**. Thus the primary proxy result does not beat baseline; the negligible mean improvement over the current longitudinal model is inconsistent across seeds. Week 20 is a secondary positive in 20 mice: all three residual runs improve on baseline, current longitudinal, and training-mean MAE. Mean Week-20 MAE is 1.9662 versus baseline 2.0704 (approximately 5.0% lower) and current longitudinal 2.0506. All horizons are reported, and this selected secondary pattern is not a confirmatory disease-progression result.

The user explicitly authorized this follow-up after the residual forecast experiment improved direct genotype ranking but failed its joint genotype/future-fidelity advancement criterion. That earlier result remains preserved. This comparison asks whether the residual representation's genotype readability carries through the existing VLM and whether proxy predictions change.

The [frozen protocol](PROTOCOL.json) specifies three seeds and the same four subject folds. Twelve residual-input fits are compared with the six previously verified baseline/current-longitudinal runs. Model architecture, question-boundary supervision, loss weights, optimizer, 20-epoch final-checkpoint schedule, labels, splits, and inference remain unchanged. Only the forecast source and its verification route differ from the matched original longitudinal arm.

All six existing comparator runs passed artifact verification before launch. The production dataset loader reproduced the 128 residual four-token inputs exactly (32 mice in each of four fold-specific forecast sets). Every forecast fit excludes its query animals and the outer test animals. No new genotype supervision, forecast fitting, threshold adjustment, or VLM parameter search is included.

The two additional routing tests passed after aligning their import path with the script entry point. They reject changes to matched training settings, seed, token count, model source, forecast root, and population. The new runner retains the existing training/evaluation body and checkpoint-resume/completion checks. No runtime training change was needed after launch.

## Outputs and reproduction

Runs are stored separately as `/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_residual_seed{0,1,2}`. Each run retains all four final checkpoints, exact split records, effective configurations, target normalization, prediction rows, and artifact hashes. The worker runs the three seeds in order on GPU 6 and then invokes final verification automatically.

```bash
bash scripts/run_residual_vlm_worker.sh
.venv-test/bin/python scripts/report_residual_vlm_progress.py
.venv-test/bin/python scripts/evaluate_residual_vlm.py
```

The worker can resume completed folds and complete partial Trainer checkpoints without replacing earlier comparator runs. Final reporting includes every seed, fold, and proxy horizon. AUROC compares KO-WT pairs only within each fitted model; accuracy retains the fixed zero-logit threshold. Proxy MAE is averaged within mouse before averaging mice. Independent standard-library arithmetic checks all nine runs.

This is a development comparison on repeatedly inspected folds, with genotype/diet/acquisition confounding and an unvalidated PET proxy. A VLM genotype improvement would not alone validate disease progression or individualized future-scan forecasting.
