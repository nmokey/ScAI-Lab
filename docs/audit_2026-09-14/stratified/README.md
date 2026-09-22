# Fixed evaluation amendment

**Complete:** all 24 VLM fits and the six-run artifact gate passed. Independent standard-library arithmetic reproduces all six results. See [the full results](RESULTS.md) and [the diagnosis of weak/reversed genotype separation](GENOTYPE_MECHANISM.md). There is no consistent longitudinal improvement: within-fold genotype AUROC is 0.4194/0.4516/0.4516 versus baseline 0.5161/0.4839/0.4355, and longitudinal proxy MAE is higher in all three amended seed comparisons.

Authorized after the genotype follow-up: replace pooled LOSO genotype ranking with comparisons within the same fitted model. The original LOSO results remain unchanged. This is a disclosed evaluation amendment after observing those results, not an independent confirmatory study.

The frozen [protocol](PROTOCOL.json) uses four stratified subject folds, split seed 0, the same three model seeds (0/1/2), both original arms, fixed requested 20 epochs, and the existing data, architecture, loss weights and learning rates. Every test fold has eight mice: two folds contain 4 KO/4 WT and two contain 3 KO/5 WT. There are 24 VLM fits, each with 24 training mice. Forecasters use five inner cross-fits plus one outer-training fit per outer split: 24 forecaster fits total with the existing 300-epoch settings and no genotype conditioning.

Genotype AUROC is the weighted mean of the four within-fold AUROCs, with weight `n_KO * n_WT` (16, 16, 15, 15). No cross-model score comparisons are made. Accuracy retains the original zero-logit threshold and includes a training-majority comparator. The existing proxy MAE, actual training-fold mean comparator and every horizon are also retained. No score inversion, threshold search, parameter selection, model search, or held-out checkpoint selection is permitted. All prescribed folds and seeds must complete before the final comparison is reported.

Genotype/diet/acquisition dependence remains: this estimates the existing subject-level task and cannot establish independent-group generalization. The change from 31 to 24 training mice per fit means a difference from the old LOSO scores cannot be attributed only to the aggregation formula. Seeds and the 62 within-fold KO–WT pairs are not independent observations. No significance or calibration claim is planned.

Checks passed: [31 tests](validation_tests.log), including exact outer-future invariance, training-query future exclusion with a positive control, manifest/token rejection, subject coverage, actual training means, and invariance of within-fold AUROC to cross-model offsets. [Frozen source/config hashes](frozen_inputs.json) and `source_snapshot/` preserve the implementation.

The additional unequal-pair weighting check passed in the [nine-test follow-up](weighting_followup_tests.log). [Real forecast validation](forecast_validation.json) verifies all 384 tokens and 24 training memberships against the actual baseline/archive inputs. [Configuration comparison](amendment_differences.json) confirms that original training sources and all six sets of training settings are unchanged. The first completed fits confirm 180 optimizer updates and epoch 20.0 with the new 24-mouse training folds; the original 31-mouse LOSO fits had 220 updates and reported epoch approximately 19.53 under the same requested 20-epoch setting.

`scripts/check_stratified_metrics.py` independently checks all six final results using only standard-library pair counting and arithmetic. Its report will be `independent_metrics.json`.

Execution logs are `forecast_export.log`, `worker_gpu{4,5,6}.log` and `stratified_{base,long}_seed{0,1,2}.log`. Full checkpoints, predictions, memberships, effective configs and completion hashes are saved separately under `/data1/Processed_NIfTI_Test/embeddings/vlm/runs/stratified_*`. Run locks and exact fingerprints protect restart; partial outputs cannot become complete results.

After all six runs complete:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-test/bin/python scripts/aggregate_stratified_vlm.py \
  --runs /data1/Processed_NIfTI_Test/embeddings/vlm/runs \
  --truth /data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json \
  --split docs/audit_2026-09-14/stratified/PROTOCOL.json \
  --output-dir docs/audit_2026-09-14/stratified
```

Final outputs will be `RESULTS.md`, `paired_results.json` and `completion.json`. The aggregator verifies all six source/data contracts, every retained checkpoint hash, exact forecast memberships, actual input hashes, and fold/aggregate score reproduction before writing them.
