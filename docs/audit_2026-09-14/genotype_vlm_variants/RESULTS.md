# Fixed genotype VLM variants

Both prescribed variants completed all four folds and three seeds. Settings and thresholds were fixed before fitting. All new checkpoints passed exact inference reload and independent metric checks.

| Seed | Input/model | AUROC | Correct / 32 | Balanced accuracy | KO recall | WT recall |
|---|---|---:|---:|---:|---:|---:|
| 0 | base | 0.5161 | 16/32 | 0.4603 | 0.1429 | 0.7778 |
| 0 | residual | 0.5484 | 17/32 | 0.5040 | 0.2857 | 0.7222 |
| 0 | direct_visual | 0.6935 | 16/32 | 0.4762 | 0.2857 | 0.6667 |
| 0 | genotype_only | 0.6613 | 17/32 | 0.5119 | 0.3571 | 0.6667 |
| 1 | base | 0.4839 | 16/32 | 0.4603 | 0.1429 | 0.7778 |
| 1 | residual | 0.5484 | 16/32 | 0.4603 | 0.1429 | 0.7778 |
| 1 | direct_visual | 0.6774 | 16/32 | 0.4841 | 0.3571 | 0.6111 |
| 1 | genotype_only | 0.5484 | 16/32 | 0.4762 | 0.2857 | 0.6667 |
| 2 | base | 0.4355 | 16/32 | 0.4603 | 0.1429 | 0.7778 |
| 2 | residual | 0.5484 | 17/32 | 0.4960 | 0.2143 | 0.7778 |
| 2 | direct_visual | 0.6935 | 16/32 | 0.4841 | 0.3571 | 0.6111 |
| 2 | genotype_only | 0.5323 | 17/32 | 0.5040 | 0.2857 | 0.7222 |

The fixed residual-embedding logistic classifier alone scores AUROC 0.6774, accuracy 0.5938 (19/32). It is a simpler comparator, not a matched-optimizer ablation. Always-WT accuracy is 18/32 (56.25%). AUROC counts 62 within-model KO-WT pairs; different fitted models are never ranked together.

| Seed | Residual proxy MAE | Direct-visual proxy MAE | Training-mean MAE |
|---|---:|---:|---:|
| 0 | 5.6552 | 5.7423 | 5.6106 |
| 1 | 5.7959 | 5.3915 | 5.6106 |
| 2 | 5.5180 | 5.5311 | 5.6106 |

**Genotype-only proxy outputs are not supervised and are not performance estimates.** Their raw arithmetic remains in the machine-readable audit artifacts only.

## Direct-visual component diagnostics

| Seed | Combined AUROC | Language-state component AUROC | Direct-visual component AUROC |
|---|---:|---:|---:|
| 0 | 0.6935 | 0.5806 | 0.6774 |
| 1 | 0.6774 | 0.4516 | 0.6774 |
| 2 | 0.6935 | 0.6452 | 0.6935 |

Components come from the jointly trained model; removing a component at inference is diagnostic and does not replace training a standalone model. All folds, supervised proxy horizons, and raw scores are retained in [full results](paired_results.json). No model or threshold is selected for a replacement headline result.

These are repeatedly inspected development folds from 32 mice with acquisition/diet confounding. Seeds are not independent studies, and improved genotype discrimination does not establish disease-progression prediction. [Protocol](PROTOCOL.json), [independent scoring](independent_metrics.json), [actual-backbone checks](runtime_checks.json).
