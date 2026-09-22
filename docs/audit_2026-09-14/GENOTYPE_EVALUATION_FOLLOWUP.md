# Reopened genotype evaluation finding

> This preserves the rationale for the completed evaluation amendment. Later residual and hybrid experiments are in [current status](../STATUS.md) and [all post-audit results](../POST_AUDIT_RESULTS.md).

**Follow-up completed:** the user authorized the four-fold amendment below. All 24 VLM fits and independent scoring checks passed. The identified cross-model ranking defect is addressed; results remain weak. See [amended results](stratified/RESULTS.md) and [the subsequent mechanism investigation](stratified/GENOTYPE_MECHANISM.md). This document preserves the rationale recorded before those amended fits.

The user's concern about consistently below-chance genotype scores is justified. The earlier statement that all methodology gates were closed was too broad. The saved scores are reproducible, but pooled leave-one-subject-out AUROC is not yet a reliable estimate of this model's genotype discrimination.

## What was checked

The dataset builder, training labels, BCE objective, saved labels and independent AUROC calculation agree on KO = 1 and WT = 0. All six runs' prediction labels match their subject IDs. The other question types also produce below-chance genotype AUROC, so this is not confined to routing the genotype-only question. At the fixed zero-logit threshold, longitudinal accuracy is 50.0%, 15.6% and 43.8% across seeds; the always-WT comparator is 56.25% on these 32 mice.

The read-only diagnostic in `scripts/diagnose_genotype_fold_offsets.py` strictly reloaded all 32 baseline and all 32 longitudinal seed-0 checkpoints. Every held-out genotype logit reproduced exactly, with maximum absolute error zero. It also scored each model's 31 training mice and the identical first-fold genotype input across all models. No model was fitted, no saved prediction was changed, and no threshold or score direction was selected.

| Diagnostic, seed 0 | Baseline | Longitudinal |
|---|---:|---:|
| Original held-out AUROC | 0.3532 | 0.0635 |
| Median within-model training AUROC | 0.7185 | 0.6450 |
| Range of within-model training AUROC | 0.6681–0.8361 | 0.4487–0.7521 |
| Median within-model training-logit SD | 0.3314 | 0.0573 |
| Identical-reference score AUROC against each fold's held-out label | 0.4167 | 0.2738 |

Training results are explicitly in-sample diagnostics, not performance estimates. The reference is the first fold's held-out image/token input, fixed by directory order, and is identical across models. Its AUROC measures association between model-specific responses and the held-out labels; it is not an image-classification metric. Because this is one real input, with differing training-membership status and model-specific interactions, it cannot isolate or quantify the entire source of bias.

## Why a very low pooled AUROC need not mean inverted biological signal

There are 14 KO and 18 WT mice. Holding out a KO leaves 13/31 KO in training; holding out a WT leaves 14/31. A model predicting only the training positive fraction therefore gives every held-out KO a lower score than every held-out WT: pooled AUROC is exactly zero despite no image information. That specific predictor still has 56.25% accuracy here, so the prior-only example does not by itself explain the very low accuracy of some fitted seeds.

This pooling problem is established in the primary literature: [Stratification bias in low signal microarray studies](https://pubmed.ncbi.nlm.nih.gov/17764577/) and [Leave-one-out cross-validation, penalization, and differential bias of some prediction model performance measures](https://pubmed.ncbi.nlm.nih.gov/37127679/). Within-fold ranking avoids comparisons between differently fitted scoring functions, but each present test fold has only one mouse, so its AUROC cannot be calculated.

The actual models show some training discrimination that does not carry over to held-out mice, and the longitudinal training logits vary little between mice. Their fold-specific mean training logits also shift in the expected prior direction: −0.2645 for KO-held-out models versus −0.1558 for WT-held-out models. The identical-input control itself gives reversed rankings. These observations support weak generalization plus cross-model score effects. They do **not** prove that class proportions explain all of the reversal: the longitudinal identical-reference/held-score correlation is only 0.166, and the identical-reference class-mean gap is much smaller than the held-out gap. Optimization, sensitivity to training membership and forecast cross-fitting remain possible contributors.

## Required closure, without a model search

Reopen V07 for genotype evaluation. Keep all existing raw results and label them as pooled LOSO diagnostics. Do not flip scores, choose a favorable threshold, or report a post-hoc offset correction as validated performance.

The next bounded validation should freeze a shared stratified subject-level split with both genotypes in every test fold, retain the existing architecture, targets, 20-epoch schedule and three seeds, and compare baseline/longitudinal using within-fold AUROC averaged with a declared weighting rule. A concrete economical choice is four folds with split seed 0, pair-count weighting for the within-fold AUROCs, and the existing model seeds 0/1/2: 24 VLM fits total. Rebuild the longitudinal forecasts nested within these exact splits; existing LOSO forecast files cannot be reused because their fit sets contain members of the new test folds. Compare threshold accuracy with each fold's training-majority predictor. State that this evaluation amendment follows inspection of the original results.

This would validate the existing subject-level genotype comparison. It still cannot establish independence from acquisition group or diet, nor guarantee a longitudinal improvement. The existing paired proxy-MAE arithmetic is unchanged by this finding; genotype AUROC bias is not an explanation for the absence of a primary regression benefit.

Raw diagnostic evidence: [baseline](genotype_fold_diagnostic_base0.json), [longitudinal](genotype_fold_diagnostic_long0.json). The scope of these model diagnostics is seed 0, not all three seeds.
