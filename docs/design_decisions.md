# Research pipeline methods and validation limits

Updated 2026-09-21. These notes describe the completed repaired protocols and separately authorized development experiments. [Current status](STATUS.md) identifies completed work and proposed, unrun controls. Targeted checks support the declared computations; they do not prove the absence of all bugs or methodological issues. The [earlier methods notes](audit_2026-09-14/legacy_documents/design_decisions.md) are archived; their anatomical and inferential claims are not current. Implementation evidence is in the [closure report](audit_2026-09-14/PIPELINE_CLOSURE.md).

## Population and endpoint

The experiment retains 32 eligible NaF mice and 96 fixed-template records. Every prediction starts at Week 12 and requests Weeks 15, 18 and 20. Missing follow-up observations mask supervision and scoring; they do not choose the inference prompt. Numerical targets preserve the original eligible visits and two-decimal rounding, and are stored as structured numbers rather than parsed from question wording.

The endpoint, historically called TBR-2, is positive-voxel P95 in the middle third of a PET crop divided by a trimmed median. Hotspot trimming affects the denominator; the numerator is the original mid-band P95. It is a PET distribution proxy, not validated aortic TBR or the source study's localized SUV measurement. All 111 current source measurements reproduce exactly. Source ROI measurements and histology labels are unavailable. Retaining this endpoint avoids selecting a replacement on the evaluation animals.

Mouse identity follows the user-accepted filename order: lower-left, lower-right, upper-left, upper-right, with explicit tube/phantom overrides. Count mismatch or incomplete override coverage raises an error. This validates consistency under the heuristic, not external identity confirmation.

## Splitting, forecasts and normalization

The original repaired LOSO study holds out one subject, leaving 31 training mice. The subsequent four-fold amendment holds out eight mice, leaving 24 training mice; residual and genotype-variant follow-ups retain that split. In each protocol, five fixed inner subject partitions cross-fit forecast tokens for outer-training mice. A separate MLP fitted on the entire outer-training set predicts the outer-test mice's tokens. No genotype conditioning or observed future test scan is used. All transforms and normalization references exclude the outer holdout; training-token fits also exclude their query subject.

The original forecaster retained its MLP architecture, cosine objective and training settings. Predictions are normalized using a training-fold reference before autoregressive feedback. This changes later directions as well as output magnitudes. Manifests record fit memberships, baseline input hashes and token hashes. Missing, malformed, nonfinite or zero-norm required tokens fail instead of being silently padded.

The residual development forecaster is a separate model. At each horizon it computes `normalize(unit_baseline + learned_change(previous_prediction, cohort, horizon))`. Every horizon retains the same observed baseline anchor; the preceding prediction is the recurrent input. The output layer starts at zero, so initial directions equal baseline. Its objective combines masked, subject-weighted cosine error over the full rollout with a 0.01 squared-change penalty. The 300-update budget, 512-wide hidden layers and learning rate 0.001 were fixed. Exports use the original forecaster's training-only scale reference; genotype and PET targets are not forecaster supervision. Residual anchoring, full-rollout training, normalized inputs and regularization changed together, so their individual effects have not been isolated.

Inner fits train on fewer mice than the outer fit. Even with correct exclusions, their forecast distributions may differ; the present checks do not eliminate that possible generalization issue.

Regression normalization uses unique eligible subject/horizon observations, population SD (`ddof=0`), singleton SD 1, and mean 0 / SD 1 for absent slots. Per-fold statistics are saved and restored with the checkpoint. Repeated question types do not duplicate the normalization observations.

## VLM training and inference

Within each baseline/original-longitudinal/residual VLM comparison, arms share the same backbone, losses, projection, heads, question boundary, training schedule, folds and initialization seeds. Baseline uses one observed image token; longitudinal uses the observed token plus three forecasts. The heads read the explicit final question-prefix position during both training and inference. Ground-truth answer tokens supervise the causal language loss, but are causally downstream of the head input. Prompt/truncation rules that would change the head boundary are rejected.

Canonical PEFT serialization retains all trained adapters. Loading verifies adapter tensors, projection, heads, numerical normalization buffers and pooling metadata. Checkpoint completion metadata is written last. Resume also requires optimizer, scheduler and RNG state. The actual 4-bit backbone preserves language logits and both head outputs exactly across the verified round trip.

Training requests the existing 20 epochs. With the installed Trainer and four-step accumulation, 93 records produce 220 updates and a reported final epoch near 19.53. The original LOSO arms use that same actual schedule and final state. The four-fold comparisons instead use 72 training records and execute 180 updates / 20 epochs. Final checkpoints are fixed; logged held-out losses do not select checkpoints. Later design choices followed examination of earlier results, so the overall development sequence is not an untouched confirmatory evaluation. Calibration has not been assessed; genotype threshold accuracy uses logit > 0.

## Separate classifier and objective variants

The direct-visual hybrid adds a bias-free linear genotype score from all four concatenated image embeddings to the original VLM genotype logit. Its visual weights start at zero; per-coordinate mean and scale use 24 unique training mice only. The original combined loss remains. This provides a route around the language model and must be described as a hybrid classifier. It introduces no additional observed image. Component scores are diagnostics of a jointly trained model, not separately retrained ablations. No matched baseline-input hybrid has been run.

Genotype-only training retains the original VLM architecture, batches, three question types, and genotype loss weight, while removing language and proxy losses. Its proxy outputs are unsupervised and excluded from performance claims. Both new variants have a distinct version-3 checkpoint contract and strict variant loader; the core model uses version 2.

Direct embedding probes are separate fixed classifiers: training-only StandardScaler and L2 logistic regression with C=1 on baseline, forecast-only, or concatenated embeddings. Their AUROC/accuracy measure genotype readability, not future-embedding fidelity or VLM reasoning.

## Scoring

TBR-only and combined-question predictions are averaged within each subject/horizon. Genotype uses the genotype-only question. Current genotype AUROC counts KO–WT pairs only within each fitted test fold and weights by the 62 available pairs; scores from different models are never ranked together. Original pooled LOSO AUROCs remain historical diagnostics, not current discrimination estimates. The primary error averages valid-horizon absolute errors within each mouse, then across mice. The primary paired difference is longitudinal minus baseline, so negative favors longitudinal. Observation-weighted MAE is retained for historical continuity. Each horizon reports MAE, MSE, ordinary R² and correlation, with an independently reconstructed training-fold mean comparator. Negative ordinary R² alone does not show inferiority to that deployable comparator.

Seeds 0, 1 and 2 are paired across arms and reported separately, with initialization SD. Seeds do not increase the number of mice. Subject bootstrap intervals describe fixed predictions and do not account for refitting or acquisition-group dependence. Fixed-score label shuffling is not a full-refit test of the VLM training procedure and supplies no publication p-value here.

Metrics use the numerical multitask heads. Generated prose is retained in raw predictions but its clinical or factual quality is not established by these head metrics.

## Interpretation limits

The VLM cohort has three acquisition components (KO 4, KO 10, WT 18), including only one WT component. Holding that entire WT component out leaves no WT training class; dropping that fold cannot yield a valid group-held-out AUROC. Subject-held-out results describe this acquisition structure and do not establish independent-group genotype generalization.

Forecast tokens are deterministic functions of baseline inputs and the training population. They add no new held-out measurements, but may help a finite-data learner through representation. A two-arm gain would support adding these predicted tokens on this dataset; it would not prove personalized future modeling or explain whether token count contributes. Repeated-baseline and training-derived mean-future VLM controls were subsequently recommended, but have not been implemented, launched, or authorized. The hybrid also lacks a matched baseline-input comparison. These gaps limit attribution of gains.

The same 32 mice and folds have been repeatedly examined during development. More seeds do not create independent confirmation. Genotype and diet are confounded. Whole-body features do not localize aortic disease, and week coincides with acquisition date. Human-scale COLIPRI (2 mm) and Merlin (1.5 × 1.5 × 3 mm) resampling plus padding differ from RAD-DINO/M3D resizing; this comparison does not isolate model architecture. CT clipping at 240 HU further limits claims about source-study calcification thresholds above 350 HU.

Merlin now uses content- and preprocessing-dependent cache keys, with a fresh 229-scan extraction. TBR-3 now maps through physical affines and rejects unsupported geometry. Neither repair proves anatomical ROI validity or alignment between separate acquisitions. TBR-3 is not used by this VLM protocol.

Installed architecture verification: Merlin uses a 3D-inflated ResNet-152, not a 3D ViT; COLIPRI uses its Primus backbone with projection and attention pooling. [Merlin checkpoint/source fingerprints](audit_2026-09-14/merlin_model_provenance.json) supplement the regenerated-embedding provenance.
