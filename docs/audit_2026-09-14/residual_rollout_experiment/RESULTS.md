# Residual full-rollout experiment

One fixed development experiment on the existing four subject folds. No genotype supervision, no parameter search, and no VLM fitting. Every training query receives inner-cross-fitted forecasts; outer test forecasts use only outer training mice. See [the frozen protocol](PROTOCOL.json).

## Direct genotype readout

| Input | Within-fold AUROC | Correct / 32 | Balanced accuracy | Mean training AUROC | Fold AUROCs |
|---|---:|---:|---:|---:|---|
| baseline | 0.6290 | 21/32 | 0.6468 | 1.0000 | 0.6875, 0.3750, 0.5333, 0.9333 |
| current_future | 0.4516 | 15/32 | 0.4643 | 0.9983 | 0.5625, 0.3125, 0.4000, 0.5333 |
| current_combined | 0.5968 | 20/32 | 0.6349 | 1.0000 | 0.5625, 0.5000, 0.5333, 0.8000 |
| residual_future | 0.6774 | 19/32 | 0.5754 | 1.0000 | 0.6875, 0.3750, 0.6667, 1.0000 |
| residual_combined | 0.6774 | 19/32 | 0.5754 | 1.0000 | 0.6875, 0.3750, 0.6667, 1.0000 |

Identical fixed standardized logistic readout for all arms. All baseline/current predictions reproduce the preceding direct-embedding probe exactly. Always-WT accuracy is 18/32 (56.25%). AUROC counts only the 62 within-model KO-WT pairs.

## Forecast fidelity

Cosine similarity to the observed future embedding; higher is better. Each horizon uses the same available held-out scans for all four predictors. Centroids use outer-training mice only.

| Horizon | Observations | Current forecast | Residual forecast | Persistence | Training-week centroid |
|---|---:|---:|---:|---:|---:|
| Week 15 | 32 | 0.9866 | 0.9839 | 0.9796 | 0.9869 |
| Week 18 | 15 | 0.9773 | 0.9715 | 0.9681 | 0.9854 |
| Week 20 | 20 | 0.9804 | 0.9794 | 0.9745 | 0.9866 |
| Mean within mouse, then across mice | 32 mice | 0.9827 | 0.9808 | 0.9759 | 0.9866 |

**Predeclared advancement criterion passed: False.** The criterion requires both residual genotype readouts to meet baseline AUROC and residual subject-mean forecast cosine to exceed all three forecast comparators.

This combines residual anchoring, full-rollout training, unit-direction inputs, and a fixed residual penalty; it does not isolate which change caused an effect. Genotype is not disease severity. Acquisition and diet remain confounded, and these folds have been inspected repeatedly. Results are descriptive development evidence; no independent-acquisition significance or confirmatory improvement is claimed.

Evidence: [raw scores, forecast comparisons, and hashes](results.json). Checkpoints and all 384 forecast tokens are retained at the output path in the protocol.
