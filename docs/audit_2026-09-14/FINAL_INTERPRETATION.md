# Interpretation of the completed repaired comparison

**Genotype evaluation amendment complete:** all 24 four-fold fits passed verification. Within-fold AUROC is baseline 0.5161/0.4839/0.4355 versus longitudinal 0.4194/0.4516/0.4516; the identified cross-model pooling issue is addressed, but useful genotype discrimination is not established. The earlier broad methodology-closure claim remains too strong. See [amended results](stratified/RESULTS.md) and [the direct investigation of training versus held-out score direction](stratified/GENOTYPE_MECHANISM.md). The original numerical results below are preserved as the original fixed-protocol study.

All six prescribed runs completed 192 held-out folds. The final gate passed 114 tests, and the separate real-data, encoder and quantized-model checks passed. A second implementation using only standard-library arithmetic independently reproduced primary MAE, pairwise genotype AUROC, per-horizon MAE/MSE/R² and the training-mean comparator from the raw predictions ([check](independent_metric_check.json), [command](../../scripts/check_saved_vlm_metrics.py)). No further fitting or endpoint/parameter search was introduced.

## What the corrected result supports

The primary comparison does **not** favor longitudinal tokens. Seed-mean subject MAE is 5.6112 for baseline and 5.6494 for longitudinal: difference +0.0381, with initialization SD 0.0579. The conditional fixed-prediction subject interval is −0.1880 to +0.2330; it does not incorporate refitting or acquisition-group dependence. The true training-fold mean predictor has subject MAE 5.6765, so both arms are only slightly better on average. Individual seed performance and all horizons remain in [the complete results](../results.md).

There is a small, consistent **secondary W18 result**: longitudinal MAE is lower in all three seeds, averaging 6.2274 versus baseline 6.4072 (about 2.8% lower) on 15 observed mice. Each longitudinal seed also has lower W18 MAE than the training-mean comparator, 6.4616. W15 and W20 MAE do not improve over baseline in any seed. Report the W18 result with those other horizons and the primary outcome; it does not establish an overall benefit or personalized future modeling.

Genotype AUROCs remain poor: baseline 0.3532 / 0.1468 / 0.1627 and longitudinal 0.0635 / 0.0913 / 0.2103. The independent pairwise calculation reproduces these values. They are scores for KO with KO as the positive label, consistent with training. There is no positive genotype result from these VLM runs.

## What the existing outputs reveal about poor performance

The regression heads recover little between-mouse variation. Their predictions vary much less than the observed targets:

| Horizon | Target SD | Baseline prediction SD across seeds | Longitudinal prediction SD across seeds |
|---|---:|---:|---:|
| W15 | 8.734 | 1.200–1.680 | 0.482–0.745 |
| W18 | 7.336 | 0.658–0.827 | 0.467–0.551 |
| W20 | 2.297 | 0.183–0.285 | 0.132–0.177 |

Each prediction SD is calculated across held-out mice within one fitted-seed run, after averaging its two numerical-question predictions. The ranges span the three seeds; they are not confidence intervals. These values describe strong compression toward population-level predictions, especially in the longitudinal arm ([source calculations](independent_metric_check.json)). They do not by themselves identify whether feature limitations, optimization or target noise dominates.

The forecast diagnostic agrees with that limited personalization: baseline rollouts beat persistence but trail the outer-training week centroid at every horizon. The extra tokens supply a learned transformation of the same W12 input, with no new observed follow-up measurement. A high absolute embedding cosine therefore provides little evidence that useful individual future variation was recovered. The repaired synthetic image-signal control shows that the learning/save/reload path can learn an available signal; it does not establish that these real CT features contain the relevant future PET signal.

The genotype result also needs careful interpretation. Mean KO logits are lower for held-out KO mice than for held-out WT mice in all six runs. The existing prior-only LOSO control demonstrates that changing training class proportions with each holdout can itself create reversed pooled rankings (AUROC 0 without image information). This is a possible explanation for part of the pattern, **not an established causal diagnosis of these VLM outputs**. A full-refit VLM null was not run, so no below-chance significance claim is made. The three NaF acquisition components and single WT component additionally prevent a defensible independent-group genotype claim.

## What code repairs cannot establish

The PET-2 values reproduce exactly, but the statistic is a whole-crop distribution proxy rather than the source study's localized measurement. Original ROI and histology labels are unavailable. Identity remains conditional on the accepted filename-position rule and explicit overrides. Genotype is confounded with diet, and week with acquisition date. These limitations remain explicit rather than being presented as repaired biological ground truth.

The earlier VLM failures were not valid evidence about the intended trained model because adapters failed to reload and the protocol had other defects. The new result replaces that evidence: the declared pipeline is now technically validated, but the broad longitudinal advantage is still unsupported. The W18 pattern and the separately verified all-week RAD-DINO genotype/diet association are the limited positive observations that can be reported with their stated scope.
