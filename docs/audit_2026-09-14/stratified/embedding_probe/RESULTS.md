# Direct genotype readout of corrected longitudinal embeddings

Post-audit diagnostic: one fixed standardized L2 logistic classifier per representation and fold. Same 32 mice and four validated mixed-class subject folds; 24 training mice and eight test mice each. Only existing baseline scans and nested predicted future tokens are inputs. No VLM or new forecaster fitting. Scaler statistics come only from training mice. KO=1; fixed zero-logit threshold.

| Input | Within-fold AUROC | Correct / 32 | Balanced accuracy | Fold AUROCs | Mean training AUROC |
|---|---:|---:|---:|---|---:|
| baseline | 0.6290 | 21/32 | 0.6468 | 0.6875, 0.3750, 0.5333, 0.9333 | 1.0000 |
| predicted_only | 0.4516 | 15/32 | 0.4643 | 0.5625, 0.3125, 0.4000, 0.5333 | 0.9983 |
| baseline_and_predicted | 0.5968 | 20/32 | 0.6349 | 0.5625, 0.5000, 0.5333, 0.8000 | 1.0000 |

The always-WT comparator scores 18/32 (56.25%) accuracy and 0.5 balanced accuracy. AUROC aggregates 62 within-model KO-WT pairs; scores from different fitted classifiers are never ranked together.

These results describe this fixed linear readout, not the maximum recoverable information. The small cohort and genotype/diet/acquisition confounding prevent an independent-acquisition genotype claim. Training tokens use inner cross-fitted forecasters, whereas test tokens use the outer-training forecaster; that representation shift remains part of the current pipeline. No hyperparameters, thresholds, signs, or representations were selected based on test performance. All three fixed representations are reported.

The older strong longitudinal probes used genotype-conditioned forecasts and are invalid evidence. The separate all-week RAD-DINO encoder association concerns a different population and observed visits; it is not a performance estimate for these baseline-derived forecasts.

Artifacts: [protocol](PROTOCOL.json), [scores and input hashes](results.json). Each fold also saves its feature matrix and fitted scaler/classifier for independent reproduction.
