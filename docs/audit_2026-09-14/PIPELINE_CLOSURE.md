# Existing-pipeline repair and validation record

**Genotype evaluation amendment complete:** the original pooled LOSO AUROC is vulnerable to severe pessimistic bias. All 24 amended four-fold fits passed verification, and two scorers agree on the within-fold AUROCs. The identified pooling defect is addressed; genotype discrimination remains weak. See [amended results](stratified/RESULTS.md) and [the saved-model mechanism investigation](stratified/GENOTYPE_MECHANISM.md). Original outputs and the code/artifact checks below remain preserved.

The code repairs required by the audit are implemented. The actual quantized backbone passed checkpoint and head-context checks, the real data passed validation, all 32 nested forecast exports are complete, and both arms passed a full one-fold training/reload/scoring check. The matched three-seed study is complete. **All six runs passed final artifact verification; corrected paired metrics are available.** See [current results](../results.md) and the [master checklist](../MASTER_PLAN.md).

## Implemented repairs and evidence

| Finding | Implemented behavior | Evidence |
|---|---|---|
| Lost LoRA adapters on reload | Canonical PEFT serialization, strict tensor validation, restored projection/heads/normalization and pooling metadata | [Real 4-bit round trip](quantized_checkpoint_fixed.json): exact adapter restoration and zero output differences; [repair details](CHECKPOINT_HEAD_REPAIRS.md) |
| Answer-conditioned heads | Both training and inference pool at the explicit question boundary; answers supervise only causal language prediction | Real-backbone answer perturbations change neither head; [96 current fixed-dataset prefix checks](question_prefix_validation_current.json) and image-signal learning/resume tests |
| Historical recoverable adapters | Repair retained raw adapter state without inventing absent normalization buffers | [Recovery ledger](adapter_recovery.json); four incomplete and two complete recoveries distinguished |
| MLP/VLM split leakage | Each outer holdout has five inner cross-fits for training tokens and one outer-training fit for test tokens | [Export log](nested_forecast_export.log); all 32 manifests; future-data perturbation invariance and permitted-data positive control |
| Forecast magnitude / wrong evaluation target | Train-fold normalization before recurrent feedback; actual W12-only rollout evaluated at all three fixed horizons | [Rollout validation](nested_rollout_validation.json); no observed future scan used as an inference input |
| Prompt-dependent numeric supervision | Structured targets, preserved two-decimal values and original observation masks; fixed W15/18/20 questions | [Input validation](validated_inputs.json): 96 records, 32 subjects, 229 finite unique encoder rows; ts0 source difference exactly zero |
| Crop identity and phantom exclusions | Accepted filename order plus explicit overrides; behavioral count/coverage guards and real-manifest checks | [Identity checks](validated_inputs.json); corrected RAD-DINO crops recomputed with zero differences in [encoder verification](encoder_repairs_validation.json) |
| Proxy computation | Recompute current PET-2 statistic from current crops; retain endpoint and eligible observations | All 111 source measurements reproduce exactly; [validation log](input_recomputation.log). Other historical CSV columns are not thereby validated |
| Duplicate target weighting | Fit normalization from unique subject/slot observations, population SD (`ddof=0`), with explicit singleton/absent-slot conventions | Per-fold `target_statistics.json`; analytical regression tests |
| Invalid aggregation / misleading uncertainty | Strict complete paired coverage, subject-averaged MAE, per-horizon metrics and actual training-fold mean comparator; per-seed AUROC | `validated_metrics.py`, `aggregate_seeds.py`; no rank pooling, no fixed-score permutation significance, no treating seeds as extra mice |
| Unsafe resumes / mixed versions | Atomic writes, frozen data/code/environment fingerprints, verified complete-fold hashes, separate partial aggregates | Repeated completed folds do not retrain or append duplicates; changed artifacts/configurations and cross-seed versions rejected |
| Stale Merlin preprocessing | Cache identity includes source contents and preprocessing implementation/version | Same-path replacement invalidates cache; fresh versus cached preprocessing matches exactly; all 229 embeddings regenerated and rescored with 120 full-refit genotype group-label arrangements; [results](merlin_regenerated_metrics.json) |
| TBR-3 ignores geometry | Physical CT/PET coordinate mapping, physical offsets/radius, explicit rejection of unsupported affine geometry | Planted-hotspot and translated-origin tests. Does not validate anatomy or alignment between separate acquisitions; excluded from the VLM endpoint |
| Conflicting reporting | Current publication path uses the fixed protocol and corrected result ledger; old summaries archived | [Results](../results.md), [methods](../design_decisions.md), [experiment log](../experiments.md) |

Observed, forecast and saved-projection norm distributions also passed on the first longitudinal fold across all 32 mice ([report](projected_token_validation.json)). The permanent proxy-reproduction command independently reproduced all 111 values again ([report](proxy_reproduction.json)); this was a check of the reusable validation command, not a change to frozen inputs.

## Fixed experiment

The [prespecified protocol](EXPERIMENT_PROTOCOL.json) keeps the existing 32 NaF mice, RAD-DINO baseline CT embeddings, PET-2 proxy, W12-to-W15/18/20 task, LLaMA backbone and training settings. Baseline uses one observed token; longitudinal uses that token plus three nested forecasts. Seeds are 0, 1 and 2 in both arms. No architecture, endpoint or hyperparameter search was added.

The primary comparison is longitudinal minus baseline absolute error, averaged across available horizons within each mouse and then across mice. Negative favors longitudinal. All horizons and seeds will be reported, along with the actual fold-training mean predictor. Genotype AUROC and fixed-zero-logit threshold accuracy are secondary; calibration has not been assessed.

Twenty epochs are **requested** in the unchanged Trainer configuration. The installed Trainer's gradient-accumulation schedule executes 220 updates and reports a final epoch of approximately 19.53 for 93 records. Both arms use this same schedule and the final state, with no held-out checkpoint selection. This actual schedule must be disclosed rather than described as exactly 20 complete passes.

Predictions are averaged between the TBR-only and combined questions for each subject/horizon; genotype uses its genotype-only question. Missing observations affect supervision and scoring, not the requested horizons. The retrospective eligible population remains fixed; this does not validate performance for baseline-only animals absent from that population.

Run manifests record package versions, model revision, source hashes, config and input hashes. [Frozen production source copies](run_configs/source_snapshot) preserve the implementation even though the repository contains uncommitted work. Both first folds also passed real idempotent resume when full runs reused their completed artifacts.

## Rollout diagnostic

| Horizon | Observed mice | Nested rollout cosine | Persistence cosine | Outer-training week centroid cosine |
|---|---:|---:|---:|---:|
| W15 | 32 | 0.98594 | 0.97965 | 0.98702 |
| W18 | 15 | 0.98297 | 0.96811 | 0.98531 |
| W20 | 20 | 0.98543 | 0.97451 | 0.98621 |

The forecasts exceed persistence but not the week centroid on this measure. A high absolute cosine therefore does not establish personalized prediction. Normalization before feedback changes later directions as well as magnitudes. Deterministic forecast tokens add no new held-out measurements, but could still help this finite-data learner through the representation they provide. The paired VLM comparison tests that narrower question; it cannot identify the mechanism without additional controls outside this task's scope.

## Limits closed through claim restrictions

* The PET target is middle-third positive-voxel P95 divided by a trimmed median. It is not the source study's localized SUV measurement or validated aortic TBR. Original ROI/histology labels are unavailable.
* Identity follows the user-accepted filename-position heuristic and explicit overrides. Computational consistency does not constitute external confirmation of animal placement.
* The NaF VLM population contains three acquisition components: KO 4, KO 10 and WT 18. There is only one WT component. Subject-held-out scores cannot establish independent-group genotype generalization, and subject bootstrap intervals do not resolve that dependence.
* Genotype and diet remain confounded. Whole-body features do not localize aortic disease. Human-scale COLIPRI/Merlin resampling and RAD-DINO/M3D resizing/windowing differ; the comparison does not isolate architectural superiority.
* TBR-3 coordinate repair does not prove inter-acquisition registration or anatomical ROI validity. It is not part of the current VLM target.
* The surviving all-week RAD-DINO association is AUROC 0.82819 with exact full-refit group-label p=0.00833 on 229 scans / 78 mice / 10 groups ([evidence](group_refit.log)). Encoder selection and exploratory choices constrain confirmatory interpretation. This is not a Week-12 VLM result or proof of disease localization.

## Verification status

The final main CPU gate passed **114 tests**, with 18 probe/real-data tests deliberately excluded ([log](pipeline_closure_tests.log)). Separate focused checks passed **29 tests**, including the actual crop-manifest identity check, override coverage and cross-seed aggregation guards ([log](closure_followup_tests.log)). Separate actual-data, actual-encoder and real-quantized-backbone checks are linked above; no claim is made that every exploratory probe or raw DICOM conversion was rerun.

All six full VLM runs and final paired aggregation are complete; see [the verified package](paired_vlm_results.json). Merlin regeneration and rescoring are complete (group-held-out genotype AUROC 0.44637; exact full-refit p=0.11667). Historical VLM metrics are excluded from the corrected model comparison, whether their headings previously said “leak-free,” “post-audit,” or “citable.” Repaired legacy checkpoints retain their original protocol limitations and are provenance/debugging artifacts.

Final interpretation and error analysis: [completed results assessment](FINAL_INTERPRETATION.md). An independent standard-library scorer reproduced all six runs’ primary MAE, pairwise AUROC, per-horizon MAE/MSE/R² and training-mean comparator ([evidence](independent_metric_check.json)).
