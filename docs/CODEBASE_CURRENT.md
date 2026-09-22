# Current code navigation — 2026-09-21

Read [status](STATUS.md), [methods](design_decisions.md), and [all results](POST_AUDIT_RESULTS.md) for current claims. The dated [initial code map](audit_2026-09-14/CODEBASE_MAP.md) describes the code before repairs; its defects and active-job descriptions are historical.

| Stage | Current entry points | Contract |
|---|---|---|
| Crops and identity | `scripts/build_nifti_dataset.py`, `quadrant_overrides.yaml` | Filename-position assumption plus explicit phantom exclusions; see DATA_MANIFEST.md |
| Encoder extraction | `scripts/get_raddino_embeddings.py`, `scripts/get_merlin_embeddings.py`, `vlm/utils/image_cache.py` | Corrected crops; Merlin content/preprocessing cache identity |
| Encoder scoring | `scripts/evaluate_embeddings.py`, `scripts/eval_stats.py` | Explicit subject/group scope; optional full-refit inference |
| Structured VQA | `scripts/create_mouse_traj_dataset.py`, `vlm/utils/target_contract.py` | Fixed horizons, structured proxy/genotype targets and observation masks |
| Data/provenance | `vlm/data/vqa_dataset.py`, `vlm/utils/research_io.py` | Required tokens fail if absent/invalid; hashes and forecast-fit exclusions |
| Core VLM | `vlm/model/vision_language_model.py`, `vlm/model/viz_emb_trainer.py`, `vlm/utils/checkpoint_utils.py` | Question-boundary heads; strict version-2 checkpoints and resume |
| Original repaired LOSO | `scripts/build_nested_forecasts.py`, `vlm/run/run_mouse_vlm_loso.py`, `scripts/aggregate_seeds.py`, `scripts/finalize_validated_results.py` | 31 training mice, 32 holdouts; original proxy study; pooled genotype AUROC superseded |
| Four-fold amendment | `scripts/stratified_vlm_protocol.py`, `scripts/build_stratified_forecasts.py`, `scripts/run_stratified_vlm.py`, `scripts/aggregate_stratified_vlm.py` | Four mixed-class folds; five inner forecast fits plus outer fit; within-model AUROC |
| Independent scoring | `scripts/check_saved_vlm_metrics.py`, `scripts/check_stratified_metrics.py` | Independent arithmetic on retained predictions |
| Mechanism diagnostics | `scripts/diagnose_genotype_fold_offsets.py`, `scripts/diagnose_genotype_separation.py`, `scripts/diagnose_genotype_prompt_context.py` | Saved-model class direction, fold offsets, precision, question context |
| Capacity control | `scripts/run_real_mouse_learning_control.py`, `scripts/finalize_learning_control_precision.py`, `scripts/check_learning_control_image_dependence.py` | Eight training mice; memorization, precision and image dependence only |
| Direct embedding readout | `scripts/probe_stratified_longitudinal_genotype.py`, `scripts/check_longitudinal_embedding_probe.py` | Fixed training-standardized logistic classifiers |
| Residual forecaster | `scripts/residual_rollout_forecast.py`, `scripts/run_residual_rollout_experiment.py`, `scripts/check_residual_rollout_experiment.py` | Baseline plus learned change, full rollout, zero initialization, change penalty; no genotype supervision |
| Residual-input VLM | `scripts/residual_vlm_protocol.py`, `scripts/run_residual_vlm.py`, `scripts/evaluate_residual_vlm.py` | Same VLM/settings; different frozen forecast inputs |
| Hybrid and genotype-only VLM | `scripts/vlm_genotype_variants.py`, `scripts/run_genotype_variants.py`, `scripts/evaluate_genotype_variants.py` | Jointly trained additive logit-level late fusion, or genotype-only objective; version-3 checkpoints |
| Score descriptions | `scripts/describe_genotype_variant_scores.py` | Describes completed predictions; no fitting or threshold selection |

Run the root scripts from the repository root with the recorded research environment. Each experiment README/PROTOCOL.json gives its configuration and output directory. The historical standalone `train_longitudinal.py` global exports are not valid inputs to nested VLM evaluation by themselves.

Final model weights and large training checkpoints stay on the research machine. Versioned source, configurations, numerical reports, lightweight probe artifacts and verification logs form the Git record. Frozen source snapshots identify executed code even when later development adds new scripts.
