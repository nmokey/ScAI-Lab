# Residual-embedding VLM comparison

Twelve new residual-input fits use the same VLM, four subject folds, three seeds, supervision, and fixed 20-epoch schedule as the six verified baseline/current comparator runs. Only the forecast tokens and their verification route change. The user authorized this follow-up after the residual embedding probe improved genotype ranking but failed the joint forecast-fidelity gate. This is a development comparison.

| Seed | Baseline AUROC | Current longitudinal AUROC | Residual AUROC | Baseline accuracy | Current accuracy | Residual accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.5161 | 0.4194 | 0.5484 | 0.5000 | 0.5000 | 0.5312 |
| 1 | 0.4839 | 0.4516 | 0.5484 | 0.5000 | 0.5312 | 0.5000 |
| 2 | 0.4355 | 0.4516 | 0.5484 | 0.5000 | 0.5625 | 0.5312 |

AUROC averages only within-model KO-WT comparisons, weighted by 62 pairs per seed. Classification uses a fixed zero-logit threshold; the training-majority comparator scores 56.25%.

| Seed | Baseline proxy MAE | Current longitudinal proxy MAE | Residual proxy MAE | Training-mean MAE |
|---|---:|---:|---:|---:|
| 0 | 5.6678 | 5.7183 | 5.6552 | 5.6106 |
| 1 | 5.6471 | 5.6867 | 5.7959 | 5.6106 |
| 2 | 5.3371 | 5.5843 | 5.5180 | 5.6106 |

## Every fold

| Seed | Fold | Baseline AUROC | Current AUROC | Residual AUROC |
|---|---|---:|---:|---:|
| 0 | fold_00 | 0.8750 | 0.4375 | 0.6875 |
| 0 | fold_01 | 0.3125 | 0.3125 | 0.3125 |
| 0 | fold_02 | 0.4667 | 0.6667 | 0.6667 |
| 0 | fold_03 | 0.4000 | 0.2667 | 0.5333 |
| 1 | fold_00 | 0.8125 | 0.5000 | 0.7500 |
| 1 | fold_01 | 0.3750 | 0.3750 | 0.3750 |
| 1 | fold_02 | 0.4000 | 0.6000 | 0.6000 |
| 1 | fold_03 | 0.3333 | 0.3333 | 0.4667 |
| 2 | fold_00 | 0.6250 | 0.4375 | 0.7500 |
| 2 | fold_01 | 0.3750 | 0.4375 | 0.3125 |
| 2 | fold_02 | 0.2667 | 0.6000 | 0.6000 |
| 2 | fold_03 | 0.4667 | 0.3333 | 0.5333 |

## Every proxy horizon

| Seed | Week | n | Baseline MAE | Current MAE | Residual MAE | Training-mean MAE |
|---|---|---:|---:|---:|---:|---:|
| 0 | 15 | 32 | 6.7964 | 6.9486 | 6.7513 | 6.7785 |
| 0 | 18 | 15 | 6.3685 | 5.9838 | 6.2922 | 5.9948 |
| 0 | 20 | 20 | 2.1715 | 2.0583 | 1.9694 | 2.0343 |
| 1 | 15 | 32 | 6.7743 | 6.7779 | 7.1903 | 6.7785 |
| 1 | 18 | 15 | 5.9996 | 6.0286 | 6.1664 | 5.9948 |
| 1 | 20 | 20 | 2.0850 | 2.0977 | 2.0108 | 2.0343 |
| 2 | 15 | 32 | 6.3762 | 6.6698 | 6.6124 | 6.7785 |
| 2 | 18 | 15 | 6.1471 | 6.0168 | 6.2569 | 5.9948 |
| 2 | 20 | 20 | 1.9546 | 1.9959 | 1.9183 | 2.0343 |

All nine runs passed artifact verification and independent arithmetic checks. Seeds and within-fold pairs are not independent studies. These 32 mice and folds have been inspected repeatedly; acquisition and diet remain confounded with genotype. The PET outcome is an unvalidated distributional proxy. No significance, calibration, independent-acquisition, or validated disease-progression claim follows automatically.

[Full results](paired_results.json), [independent scoring](independent_metrics.json), [frozen protocol](PROTOCOL.json).
