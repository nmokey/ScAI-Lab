# Results collected after pipeline repair

Current interpretation and proposed controls: [STATUS.md](STATUS.md). Dated/frozen report scope: [evidence guide](audit_2026-09-14/README.md).

This ledger summarizes every major completed result family from the repaired pipeline onward. Per-mouse predictions, per-fold metrics, complete learning curves, and verification details remain in the linked source artifacts. Pre-repair headline results are excluded. No new models, thresholds, or endpoints were fitted to produce this summary.

## Scope and comparability

- VLM and baseline-only forecast experiments: 32 NaF mice (14 KO, 18 WT), 96 question records, and 67 observed future targets: Week 15 n=32, Week 18 n=15, Week 20 n=20.
- The numerical PET target is the validated computation of an existing distributional intensity proxy. It is not a validated aortic ROI measurement or histological disease label.
- The first repaired study used leave-one-mouse-out training (31 training mice). The later amendment uses four mixed-class test folds (24 training mice, eight test mice). Their scores are separate experiments.
- Later genotype AUROC counts only KO–WT pairs from the same fitted model, weighted by 62 pairs across four folds. Threshold accuracy uses a fixed zero logit. Always-WT accuracy is 18/32 = 56.25%; balanced accuracy is 50%.
- Seeds 0/1/2 reuse the same mice and folds. Later experiments are development comparisons on repeatedly inspected data. Genotype, diet, and acquisition remain confounded; the NaF cohort has only three acquisition components, one WT.

## 1. Original repaired VLM study: 192 leave-one-mouse-out fits

The primary endpoint was subject-averaged future PET-proxy MAE (lower is better). Requested 20 epochs produced 220 optimizer updates, approximately 19.53 completed epochs, in both arms.

| Seed | Baseline MAE | Original longitudinal MAE | Longitudinal minus baseline |
| --- | --- | --- | --- |
| 0 | 5.7587 | 5.7522 | -0.0065 |
| 1 | 5.6489 | 5.6662 | +0.0173 |
| 2 | 5.4261 | 5.5297 | +0.1035 |
| Mean | 5.6112 | 5.6494 | +0.0381 |

Training-fold mean predictor: subject MAE **5.6765**. Paired difference initialization SD: **0.0579**. Conditional fixed-prediction subject interval: **−0.1880 to +0.2330**; it excludes refitting/acquisition dependence and is not a significance test.

Per-horizon metrics below average the three seed-specific metrics; Pearson r is also the mean of three seed correlations, not a pooled correlation.

| Week | Predictor | n | MAE | MSE | R² | Pearson r |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | baseline | 32 | 6.7939 | 82.7453 | -0.0846 | -0.1807 |
| 15 | longitudinal | 32 | 6.8704 | 81.6330 | -0.0700 | -0.4947 |
| 15 | training mean | 32 | 6.8545 | 81.2928 | -0.0656 | -1.0000 |
| 18 | baseline | 15 | 6.4072 | 61.9312 | -0.1506 | -0.6948 |
| 18 | longitudinal | 15 | 6.2274 | 58.9216 | -0.0947 | -0.6470 |
| 18 | training mean | 15 | 6.4616 | 61.7872 | -0.1480 | -1.0000 |
| 20 | baseline | 20 | 1.8378 | 5.8860 | -0.1157 | -0.5220 |
| 20 | longitudinal | 20 | 1.8712 | 5.8387 | -0.1068 | -0.7427 |
| 20 | training mean | 20 | 1.8836 | 5.8454 | -0.1080 | -1.0000 |

**Interpretation:** no overall longitudinal gain. Week-18 MAE improves in every seed: 6.4072 → 6.2274 on average (about 2.8%, 15 mice). Week 15 and Week 20 do not improve in any seed. Longitudinal improves seed-mean subject MAE for 12/32 mice.

The original pooled LOSO genotype AUROCs below are retained only as historical diagnostics. Cross-model score offsets and class-prior differences make them unsuitable as the current discrimination estimates, even though independent arithmetic reproduces them.

| Seed | Baseline pooled AUROC — superseded | Longitudinal pooled AUROC — superseded |
| --- | --- | --- |
| 0 | 0.3532 | 0.0635 |
| 1 | 0.1468 | 0.0913 |
| 2 | 0.1627 | 0.2103 |

[Original results and per-subject outputs](audit_2026-09-14/paired_vlm_results.md); [interpretation, prediction compression, and uncertainty](audit_2026-09-14/FINAL_INTERPRETATION.md).

## 2. Comparable VLM genotype results: amended four-fold protocol

All rows use the same split and seeds, 20 completed epochs and 180 updates. Baseline uses the observed Week-12 embedding. Original/residual longitudinal inputs add three baseline-derived predicted embeddings. The direct-visual model is a hybrid: a linear classifier on those four embeddings adds its score to the VLM genotype score. Genotype-only retains the original VLM architecture and removes language/proxy training losses.

| Model | AUROC: seeds 0 / 1 / 2 | Mean AUROC | Correct out of 32: seeds 0 / 1 / 2 | Mean accuracy |
| --- | --- | --- | --- | --- |
| Baseline VLM | 0.5161 / 0.4839 / 0.4355 | 0.4785 | 16 / 16 / 16 | 50.00% |
| Original longitudinal VLM | 0.4194 / 0.4516 / 0.4516 | 0.4409 | 16 / 17 / 18 | 53.12% |
| Residual longitudinal VLM | 0.5484 / 0.5484 / 0.5484 | 0.5484 | 17 / 16 / 17 | 52.08% |
| Residual + direct visual classifier (hybrid) | 0.6935 / 0.6774 / 0.6935 | 0.6882 | 16 / 16 / 16 | 50.00% |
| Residual + genotype-only objective | 0.6613 / 0.5484 / 0.5323 | 0.5806 | 17 / 16 / 17 | 52.08% |

Residual tokens improve VLM ranking over original and baseline inputs in all three seeds. Direct visual access further improves ranking consistently, but not accuracy. Genotype-only ranking gains are inconsistent and its per-seed accuracies equal residual VLM. No VLM variant reaches 70% accuracy or exceeds always-WT mean accuracy.

The jointly trained hybrid component diagnostics are not independent standalone training experiments:

| Seed | Combined AUROC | VLM-state component AUROC | Direct-visual component AUROC |
| --- | --- | --- | --- |
| 0 | 0.6935 | 0.5806 | 0.6774 |
| 1 | 0.6774 | 0.4516 | 0.6774 |
| 2 | 0.6935 | 0.6452 | 0.6935 |

The direct branch retains almost all combined ranking performance (mean 0.6828 versus 0.6882). The VLM-state branch itself receives image tokens; it is not a text-only control. A matched baseline-only hybrid was not run, so the hybrid result does not isolate a longitudinal-token benefit.

[Amendment](audit_2026-09-14/stratified/RESULTS.md), [residual VLM](audit_2026-09-14/residual_vlm/RESULTS.md), [hybrid/genotype-only comparison, including balanced accuracy and class recall](audit_2026-09-14/genotype_vlm_variants/RESULTS.md).

## 3. Comparable PET-proxy results: amended four-fold protocol

| Model | Seed 0 MAE | Seed 1 MAE | Seed 2 MAE | Mean MAE |
| --- | --- | --- | --- | --- |
| Baseline VLM | 5.6678 | 5.6471 | 5.3371 | 5.5506 |
| Original longitudinal VLM | 5.7183 | 5.6867 | 5.5843 | 5.6631 |
| Residual longitudinal VLM | 5.6552 | 5.7959 | 5.5180 | 5.6564 |
| Residual + direct visual classifier (hybrid) | 5.7423 | 5.3915 | 5.5311 | 5.5550 |
| Training-fold mean | 5.6106 | 5.6106 | 5.6106 | 5.6106 |

Genotype-only proxy outputs have no proxy supervision and are excluded from performance estimates. Subject MAE is not a simple average of the three horizon MAEs because future attendance differs between mice.

| Week | Predictor | n | Mean MAE | Mean MSE | Mean R² | Mean Pearson r |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | Baseline VLM | 32 | 6.6490 | 81.1123 | -0.0632 | -0.0620 |
| 15 | Original longitudinal VLM | 32 | 6.7988 | 80.8716 | -0.0600 | -0.1600 |
| 15 | Residual longitudinal VLM | 32 | 6.8513 | 81.2125 | -0.0645 | -0.0232 |
| 15 | Residual + direct visual classifier (hybrid) | 32 | 6.7378 | 80.0302 | -0.0490 | -0.0840 |
| 15 | training mean | 32 | 6.7785 | 79.5288 | -0.0424 | -0.2336 |
| 18 | Baseline VLM | 15 | 6.1717 | 56.5210 | -0.0501 | -0.1063 |
| 18 | Original longitudinal VLM | 15 | 6.0098 | 55.1967 | -0.0255 | -0.1139 |
| 18 | Residual longitudinal VLM | 15 | 6.2385 | 57.0359 | -0.0597 | 0.0155 |
| 18 | Residual + direct visual classifier (hybrid) | 15 | 5.9136 | 54.9138 | -0.0203 | -0.0362 |
| 18 | training mean | 15 | 5.9948 | 54.1875 | -0.0068 | -0.0986 |
| 20 | Baseline VLM | 20 | 2.0704 | 6.7924 | -0.2875 | -0.4501 |
| 20 | Original longitudinal VLM | 20 | 2.0506 | 6.8583 | -0.3000 | -0.5531 |
| 20 | Residual longitudinal VLM | 20 | 1.9662 | 6.2108 | -0.1773 | -0.3867 |
| 20 | Residual + direct visual classifier (hybrid) | 20 | 2.0250 | 6.7127 | -0.2724 | -0.5376 |
| 20 | training mean | 20 | 2.0343 | 6.7095 | -0.2718 | -0.5612 |

**Secondary positive:** residual VLM Week-20 MAE is lower in all seeds than baseline, original longitudinal, and training mean. Mean MAE is **1.9662 versus baseline 2.0704**, about **5.0% lower**, on 20 mice. Residual overall MAE nevertheless worsens. Direct-visual overall MAE is essentially unchanged from baseline, without a consistent per-seed advantage. The original LOSO Week-18 observation and this later Week-20 observation are separate secondary results, not a replicated primary win.

The hybrid also has lower Week-18 MAE than baseline in all three seeds: **5.9136 versus 6.1717** on average (about **4.2% lower**, 15 mice). It beats the training-mean comparator at that horizon in two of three seeds. This is another secondary development observation, with all other horizons retained above; it does not establish a gain from longitudinal tokens within a matched hybrid architecture.

## 4. Raw-embedding genotype classifiers

Fixed training-standardized L2 logistic regression (C=1), same four folds, no threshold/parameter search. Each row is one four-fold classifier evaluation, not a three-seed VLM average.

| Input | Held-out AUROC | Correct | Accuracy | Balanced accuracy | Mean training AUROC |
| --- | --- | --- | --- | --- | --- |
| Baseline only | 0.6290 | 21/32 | 65.62% | 0.6468 | 1.0000 |
| Original predicted futures only | 0.4516 | 15/32 | 46.88% | 0.4643 | 0.9983 |
| Baseline + original futures | 0.5968 | 20/32 | 62.50% | 0.6349 | 1.0000 |
| Residual predicted futures only | 0.6774 | 19/32 | 59.38% | 0.5754 | 1.0000 |
| Baseline + residual futures | 0.6774 | 19/32 | 59.38% | 0.5754 | 1.0000 |

Baseline contains some readable genotype/diet association. Original forecasts weaken this fixed readout. Residual forecasts improve ranking but not threshold accuracy over baseline: their AUROC gain is three additional correctly ordered pairs out of 62. Near-perfect training AUROC and much weaker held-out scores indicate a substantial generalization gap.

[Original embedding probe](audit_2026-09-14/stratified/embedding_probe/RESULTS.md); [all five representations and saved classifiers](audit_2026-09-14/residual_rollout_experiment/RESULTS.md).

## 5. Actual future-embedding prediction quality

Cosine similarity to observed future embeddings, higher is better. Persistence copies baseline forward; the training-week centroid predicts the mean future embedding from outer-training mice.

| Horizon | Observations | Original forecast | Residual forecast | Persistence | Training-week centroid |
| --- | --- | --- | --- | --- | --- |
| Week 15 | 32 | 0.9866 | 0.9839 | 0.9796 | 0.9869 |
| Week 18 | 15 | 0.9773 | 0.9715 | 0.9681 | 0.9854 |
| Week 20 | 20 | 0.9804 | 0.9794 | 0.9745 | 0.9866 |
| Mean within mouse, then across mice | 32 mice | 0.9827 | 0.9808 | 0.9759 | 0.9866 |

Both forecasters beat persistence but trail the training-week centroid at every horizon. Residual forecasting has worse fidelity than the original at every horizon despite stronger genotype ranking. Its predeclared joint genotype/fidelity advancement criterion **failed**; the subsequent VLM evaluation was explicitly authorized after that result. The residual experiment changed several forecaster ingredients together, so it does not isolate a single causal mechanism.

## 6. Learning and failure-mechanism diagnostics

- **Training-only capacity control:** both objectives learned the same eight real mice (4 KO, 4 WT), reaching 8/8 correct. Final genotype BCE: combined **0.000120**, genotype-only **0.000001**. The declared capacity criterion was first observed at update **120** and **180**, respectively. Both completed 300 updates. This is memorization/image-dependence evidence, not held-out accuracy.
- **Image dependence and reload:** swapped image tokens made outputs follow the donor; constant image tokens removed between-mouse differences. Both control endpoints reproduced exactly with matched inference precision.
- **Original amended VLMs:** all **24/24** had the intended mean class direction in training. Median training AUROC was about **0.7386 baseline / 0.7133 original longitudinal**; held-out class-mean gaps reversed in **9/12 baseline / 8/12 longitudinal** models. All held-out genotype logits reproduced exactly.
- **Simple baseline centroid diagnostic:** within-fold AUROC **0.5161**, weaker than the fitted baseline linear classifier (**0.6290**). This did not prove the features were signal-free.
- **Numerical precision:** changing inference precision altered only **2/96** original longitudinal decisions and **0/96** baseline decisions. Alternate-precision longitudinal AUROCs **0.4355 / 0.4274 / 0.4516** remained weak; these do not replace the frozen reported scores.
- **Repeated errors:** of 62 within-fold KO–WT pairs, the original longitudinal model ranked **30 incorrectly in all three seeds**, **24 correctly in all three**, and **8 inconsistently**. These pairs overlap in mice and are not independent observations.
- **New variants:** both predicted every mouse as WT in folds 00 and 03 in every seed. The hybrid nevertheless had **AUROC 1.0** in fold 03 each time, with all scores below threshold; fold 01 remained at **0.3125–0.375**. Thus both score offsets and ranking failures remain.
- **Confidence:** original longitudinal had **16 / 20 / 17 of 32** scores in 0.4–0.6 across seeds. The hybrid had **6/32** in every seed and **19–21/32** with raw confidence ≥0.9. Mean confidence on hybrid mistakes was about **0.83**, versus **0.85–0.93** for genotype-only. Stronger confidence has not produced better accuracy.
- **Original regression compression:** target SD at Weeks 15/18/20 was **8.734 / 7.336 / 2.297**; longitudinal prediction SD across mice was only **0.482–0.745 / 0.467–0.551 / 0.132–0.177**, with ranges across seeds. The heads captured little between-mouse variation.

[Capacity control](audit_2026-09-14/real_mouse_learning_control/RESULTS.md), [mechanism investigation](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md), [current confidence diagnostics](audit_2026-09-14/genotype_vlm_variants/score_diagnostics.json).

## 7. Encoder results retained or regenerated during the audit

This is a different population from the VLM study: 229 observed scans across weeks, 78 mice, and ten groups. It is not a baseline-only forecast benchmark.

| Encoder | All-week genotype/diet AUROC, group holdout | Exact full-refit group-label p |
| --- | --- | --- |
| RAD-DINO | 0.82819 | 0.00833 |
| Regenerated Merlin | 0.44637 | 0.11667 |

| Additional regenerated Merlin metric | Value |
| --- | --- |
| Four-week accuracy | 0.77293 |
| Four-week macro F1 | 0.74517 |
| Four-week one-versus-rest AUROC | 0.91847 |
| Early/late accuracy, subject or group holdout (119 scans) | 0.89076 |
| Early/late AUROC, subject or group holdout | 0.95804 |
| Genotype accuracy, subject holdout | 0.62009 |
| Genotype AUROC, subject holdout | 0.63407 |
| Genotype accuracy, group holdout | 0.48908 |
| Cohort accuracy | 0.55022 |
| Cohort AUROC | 0.55085 |
| Temporal-ordering accuracy | 0.77043 |
| Subject retrieval MRR | 0.09304 |
| Subject retrieval Recall@1 | 0.03930 |
| Subject retrieval Recall@3 | 0.07860 |
| Week retrieval mAP@5 | 0.69561 |

RAD-DINO retains an all-week genotype/diet association under this group-label refit test. It does not establish disease localization or independent confirmation after encoder selection. Merlin time-related performance may include acquisition-date cues; its genotype association does not survive the group-holdout comparison. Older COLIPRI/M3D and pre-repair tables were not newly validated by these repairs and are not presented here as fresh post-audit evidence.

[RAD-DINO full-refit evidence](audit_2026-09-14/group_refit.log); [all regenerated Merlin metrics and population](audit_2026-09-14/merlin_regenerated_metrics.json).

## 8. Verification evidence versus scientific validity

- Original fixed-protocol CPU gate: **114 tests passed**; real-data/quantized checks were recorded separately.
- **111/111 PET-proxy source measurements** reproduced exactly; **229 encoder rows** were validated and all **229 Merlin embeddings** regenerated after cache repair.
- The amended forecaster and residual forecaster each completed **24 nested fits / 384 exported tokens**, with input, split, normalization and artifact checks.
- Completed research VLM fitting: **192 original LOSO fits + 24 amended fits + 12 residual-input fits + 24 hybrid/genotype-only fits = 252 fits**. This excludes the separate capacity/smoke diagnostics.
- The latest variants passed **six focused tests**, actual-backbone objective/gradient/answer-invariance tests, **24 exact final checkpoint reloads**, and independent scoring of **12 new/comparator runs**. Earlier comparisons also have independent scoring evidence.
- These checks support the declared computation. They do not validate unavailable ROI/histology labels, establish independent acquisition generalization, or guarantee a positive longitudinal finding.

## Evidence ledger

Numeric tables were assembled from the saved JSON reports, without training or rescoring models. [Source fingerprints](audit_2026-09-14/post_audit_summary_sources.json) identify the inputs to this consolidated ledger. The frozen source reports remain unchanged.
