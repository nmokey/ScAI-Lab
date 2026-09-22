# Validated research results

For a single summary of every completed result family after repair, including the later model experiments, embedding probes, learning controls and refreshed encoders, see [the consolidated post-audit ledger](POST_AUDIT_RESULTS.md).

All six prespecified VLM runs are complete: 32 held-out mice × two arms × three seeds. Checkpoint hashes, source/input fingerprints and paired prediction coverage were verified before scoring. The endpoint is the existing PET intensity proxy, not a validated aortic measurement.

**Current genotype evaluation:** use the [completed four-fold amendment](audit_2026-09-14/stratified/RESULTS.md), which avoids ranking scores across different fitted models. The table below preserves the original LOSO study; its pooled genotype AUROCs are not the amended discrimination estimates. See also [why the remaining small separations reverse](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md).

Subsequent residual-forecast and VLM architecture/objective experiments are reported separately below, with [complete development results](audit_2026-09-14/genotype_vlm_variants/README.md). They do not replace the original protocol's primary endpoint or confer independent validation on repeatedly inspected folds.

## Baseline versus longitudinal VLM

| Seed | Baseline subject MAE | Longitudinal subject MAE | Longitudinal − baseline | Baseline genotype AUROC | Longitudinal genotype AUROC |
|---|---:|---:|---:|---:|---:|
| 0 | 5.7587 | 5.7522 | -0.0065 | 0.3532 | 0.0635 |
| 1 | 5.6489 | 5.6662 | +0.0173 | 0.1468 | 0.0913 |
| 2 | 5.4261 | 5.5297 | +0.1035 | 0.1627 | 0.2103 |

Mean paired MAE difference: **+0.0381**; initialization SD 0.0579. The direction of the primary difference varies across seeds.
The descriptive fixed-prediction subject interval for the seed-mean difference is [-0.1880, +0.2330]. It does not account for refitting or acquisition-group dependence and is not a significance test.

Primary MAE averages valid-horizon errors within each mouse and then across mice. Negative differences favor longitudinal. Seeds are repeated fits on the same 32 mice, not extra animals. All 67 observed future subject/horizon targets are included (W15: 32; W18: 15; W20: 20).

## Per-horizon results

Values below are means of the three per-seed metrics; R² is ordinary held-out R². Every horizon is shown.

| Horizon | n | Baseline MAE | Longitudinal MAE | Training-mean MAE | Baseline MSE | Longitudinal MSE | Training-mean MSE | Baseline R² | Longitudinal R² | Training-mean R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W15 | 32 | 6.7939 | 6.8704 | 6.8545 | 82.7453 | 81.6330 | 81.2928 | -0.0846 | -0.0700 | -0.0656 |
| W18 | 15 | 6.4072 | 6.2274 | 6.4616 | 61.9312 | 58.9216 | 61.7872 | -0.1506 | -0.0947 | -0.1480 |
| W20 | 20 | 1.8378 | 1.8712 | 1.8836 | 5.8860 | 5.8387 | 5.8454 | -0.1157 | -0.1068 | -0.1080 |

The actual training-fold mean predictor has subject MAE 5.6765. Baseline's seed-mean MAE is 5.6112; longitudinal's is 5.6494. Longitudinal reduces seed-mean absolute error for 12/32 mice. The accompanying JSON preserves every mouse's paired error and the observation-weighted continuity metrics.

Per-horizon MAE differences by seed (longitudinal minus baseline):

| Horizon | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| W15 | +0.0095 | +0.0530 | +0.1669 |
| W18 | -0.1534 | -0.1761 | -0.2099 |
| W20 | +0.0558 | +0.0187 | +0.0259 |

Longitudinal has lower MAE at W15: 0/3 seeds, W18: 3/3 seeds, W20: 0/3 seeds. These are secondary per-horizon results (32, 15 and 20 observed mice respectively), and do not override the primary comparison.

## Validity and limits

**Genotype evaluation amendment complete:** the genotype AUROCs above pool different leave-one-out models and remain historical diagnostics of that protocol. The verified four-fold amendment compares only predictions from the same fitted model: baseline AUROC 0.5161 / 0.4839 / 0.4355 versus longitudinal 0.4194 / 0.4516 / 0.4516. Both independent scorers agree; no consistent longitudinal gain is established. See [the amended results](audit_2026-09-14/stratified/RESULTS.md) and [the investigation of weak/reversed class separation](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md). The original numerical outputs are unchanged.

Both arms use the same question-boundary heads, targets, folds, seeds and final-state training schedule. The longitudinal tokens are generated through five inner cross-fits plus an outer-training forecaster per held-out mouse. The installed Trainer executes 220 updates (reported epoch approximately 19.53) under the fixed requested 20-epoch configuration. No setting was selected from these held-out scores.

The NaF subset has three acquisition components (KO 4, KO 10, WT 18), with one WT component. These subject-held-out results cannot establish independent-group genotype generalization. Calibration is not assessed; threshold accuracy uses a fixed zero logit. Genotype and diet are confounded. The PET proxy has not been validated against source ROI measurements or histology.

This comparison tests adding three predicted tokens to the existing baseline. It does not establish the mechanism of a gain or personalized future modeling. Actual nested rollout cosine exceeds persistence but trails the training-week centroid at all three horizons. Extra observed visits, alternative endpoints and additional VLM control arms were outside this validation task.

## Other verified results

* All 111 PET-2 source measurements reproduce exactly; corrected RAD-DINO crops and baseline features reproduce exactly.
* RAD-DINO all-week group-held-out genotype/diet AUROC remains 0.82819, exact full-refit group-label p=0.00833 (229 scans, 78 mice, ten groups). This is an association, not disease localization; encoder selection limits confirmatory wording.
* Fresh Merlin extraction includes all 229 scans. Its group-held-out genotype/diet AUROC is 0.4464, exact full-refit p=0.1167 over 120 arrangements. The old cached Merlin rows are superseded.

### Refreshed Merlin comparison rows

| Existing metric | Fresh result |
|---|---:|
| Four-class week accuracy, subject holdout | 0.77293 |
| Early/late AUROC, subject holdout | 0.95804 |
| Early/late AUROC, group holdout | 0.95804 |
| Genotype/diet accuracy, subject holdout | 0.62009 |
| Genotype/diet AUROC, subject holdout | 0.63407 |
| Genotype/diet AUROC, group holdout | 0.44637 |
| Cohort AUROC, subject holdout | 0.55085 |
| Temporal-ordering accuracy | 0.77043 |
| Subject retrieval Recall@1 | 0.03930 |
| Week retrieval mAP@5 | 0.69561 |

## Evidence and reproduction

[Paired machine-readable results](audit_2026-09-14/paired_vlm_results.json), [closure report](audit_2026-09-14/PIPELINE_CLOSURE.md), [fixed protocol](audit_2026-09-14/EXPERIMENT_PROTOCOL.json), [commands and run paths](experiments.md), [Merlin metrics](audit_2026-09-14/merlin_regenerated_metrics.json), [RAD-DINO refit evidence](audit_2026-09-14/group_refit.log). Each paired result records its source run directories. Each run retains raw predictions, split files, checkpoint files and verification manifests.

Historical VLM summaries, including those previously labeled “leak-free,” are archived and excluded from this corrected result.

## Subsequent development comparisons on the amended four folds

All VLM rows below use the same 32 mice, four subject folds, and seeds 0/1/2. Values are means across those three seeds. AUROC counts KO–WT pairs within each fitted test fold, never across fitted models. Every prescribed run completed and passed artifact and independent-scoring checks.

| Model/input | Mean AUROC | Mean accuracy | Mean supervised proxy MAE |
|---|---:|---:|---:|
| Baseline VLM | 0.4785 | 0.5000 | 5.5506 |
| Original forecast-input VLM | 0.4409 | 0.5313 | 5.6631 |
| Residual forecast-input VLM | 0.5484 | 0.5208 | 5.6564 |
| Residual + direct visual genotype head | 0.6882 | 0.5000 | 5.5550 |
| Residual + genotype-only supervision | 0.5806 | 0.5208 | Not supervised |

The fixed direct-embedding classifier scores baseline features at AUROC 0.6290, accuracy 21/32 (0.6563), and residual features at AUROC 0.6774, accuracy 19/32 (0.5938). These are simpler classifiers with different optimization, not matched VLM ablations. Always-WT accuracy is 18/32 (0.5625). Residual future-embedding fidelity remains worse than the original forecaster and training-week centroid; the improved genotype ranking does not establish better forecasting.

Direct visual access improves VLM AUROC in every seed (0.6935, 0.6774, 0.6935), but every seed has only 16/32 correct classifications. Its visual component alone retains nearly all the combined ranking performance (AUROC 0.6774, 0.6774, 0.6935). Genotype-only supervision has inconsistent ranking gains and the same per-seed accuracies as residual VLM. Neither variant reaches 70% accuracy.

Both new variants classify all eight mice as WT in folds 00 and 03 in every seed. Direct-visual fold-03 AUROC is nevertheless 1.0 in each seed, with every raw KO probability below 0.08. Fold 01 remains below chance. This identifies both score-offset/threshold problems and remaining ranking failures; it does not establish that calibration alone will fix accuracy. The new variants also make confident mistakes rather than merely borderline guesses. [Descriptive score evidence](audit_2026-09-14/genotype_vlm_variants/score_diagnostics.json) involves no new fitting or threshold selection.

Residual VLM's Week-20 proxy MAE improves over the baseline VLM in all three seeds (mean 1.9662 versus 2.0704, 20 mice), but its overall proxy MAE worsens. Direct-visual mean proxy MAE is essentially unchanged from baseline and does not improve consistently across seeds. Genotype-only proxy outputs are unsupervised and excluded from performance interpretation. All supervised horizons remain available in the linked artifacts.

These are descriptive development findings with acquisition/diet confounding. Direct visual access has not been evaluated with matched baseline-only inputs, so its improvement does not isolate a benefit from forecast tokens or language-model reasoning. [Residual forecaster results](audit_2026-09-14/residual_rollout_experiment/README.md), [residual VLM results](audit_2026-09-14/residual_vlm/README.md), [variant results and protocol](audit_2026-09-14/genotype_vlm_variants/README.md).
