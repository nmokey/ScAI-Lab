# Amended four-fold VLM evaluation

All 24 fits passed checkpoint, input, split and prediction checks. This evaluation was specified after inspecting the original LOSO results; model settings and targets were retained.

Genotype AUROC averages within-model test-fold AUROCs, weighted by the number of KO–WT pairs in each fold. It never ranks predictions from different fitted models against each other.

| Seed | Baseline AUROC | Longitudinal AUROC | Difference | Baseline accuracy | Longitudinal accuracy | Training-majority accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.5161 | 0.4194 | -0.0968 | 0.5000 | 0.5000 | 0.5625 |
| 1 | 0.4839 | 0.4516 | -0.0323 | 0.5000 | 0.5312 | 0.5625 |
| 2 | 0.4355 | 0.4516 | +0.0161 | 0.5000 | 0.5625 | 0.5625 |

| Seed | Baseline subject MAE | Longitudinal subject MAE | Difference | Training-mean MAE |
|---|---:|---:|---:|---:|
| 0 | 5.6678 | 5.7183 | +0.0505 | 5.6106 |
| 1 | 5.6471 | 5.6867 | +0.0396 | 5.6106 |
| 2 | 5.3371 | 5.5843 | +0.2471 | 5.6106 |

## Every test fold

| Seed | Fold | KO–WT pairs | Baseline AUROC | Longitudinal AUROC |
|---|---|---:|---:|---:|
| 0 | fold_00 | 16 | 0.8750 | 0.4375 |
| 0 | fold_01 | 16 | 0.3125 | 0.3125 |
| 0 | fold_02 | 15 | 0.4667 | 0.6667 |
| 0 | fold_03 | 15 | 0.4000 | 0.2667 |
| 1 | fold_00 | 16 | 0.8125 | 0.5000 |
| 1 | fold_01 | 16 | 0.3750 | 0.3750 |
| 1 | fold_02 | 15 | 0.4000 | 0.6000 |
| 1 | fold_03 | 15 | 0.3333 | 0.3333 |
| 2 | fold_00 | 16 | 0.6250 | 0.4375 |
| 2 | fold_01 | 16 | 0.3750 | 0.4375 |
| 2 | fold_02 | 15 | 0.2667 | 0.6000 |
| 2 | fold_03 | 15 | 0.4667 | 0.3333 |

## Every proxy horizon

| Seed | Week | n | Baseline MAE | Longitudinal MAE | Training-mean MAE |
|---|---|---:|---:|---:|---:|
| 0 | 15 | 32 | 6.7964 | 6.9486 | 6.7785 |
| 0 | 18 | 15 | 6.3685 | 5.9838 | 5.9948 |
| 0 | 20 | 20 | 2.1715 | 2.0583 | 2.0343 |
| 1 | 15 | 32 | 6.7743 | 6.7779 | 6.7785 |
| 1 | 18 | 15 | 5.9996 | 6.0286 | 5.9948 |
| 1 | 20 | 20 | 2.0850 | 2.0977 | 2.0343 |
| 2 | 15 | 32 | 6.3762 | 6.6698 | 6.7785 |
| 2 | 18 | 15 | 6.1471 | 6.0168 | 5.9948 |
| 2 | 20 | 20 | 1.9546 | 1.9959 | 2.0343 |

Seeds reuse the same 32 mice and split; they are not independent studies. There are 62 within-fold KO–WT comparisons per seed, which are also not independent observations. Training uses 24 mice per fold, versus 31 in the original LOSO study; differences from the old scores cannot be attributed solely to AUROC aggregation. Genotype and diet/acquisition group remain confounded, with only one WT acquisition component. No independent-group claim, significance test, calibration claim or biological validation of the PET proxy follows from these results.

[Machine-readable results](paired_results.json) retain per-fold scores, all proxy horizons and subject errors. Original LOSO outputs remain unchanged.
