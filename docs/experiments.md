# Current fixed-protocol VLM experiment

The [historical experiment log](audit_2026-09-14/legacy_documents/experiments.md) is preserved for provenance. Its descriptions of “citable” or “leak-free” VLM runs are superseded by the checkpoint, head-context and nested-split audit. Those runs do not evaluate the corrected model.

## Frozen comparison

* Existing 32 NaF mice, 96 questions, W12 baseline; future horizons W15, W18 and W20 for every mouse.
* Existing RAD-DINO embeddings and existing two-decimal PET-2 targets/masks.
* Baseline: one observed token. Longitudinal: that token plus three fold-nested normalized forecast tokens.
* Both arms: question-boundary heads, canonical strict adapter reload, final training state, requested 20 epochs, seeds 0/1/2.
* Forecaster: existing MLP, hidden 512, 300 epochs, learning rate 0.001, seed 0, five inner subject folds, genotype disabled.
* Primary outcome: subject-averaged MAE difference, longitudinal minus baseline. Genotype AUROC is secondary.

The installed Trainer executes 220 updates and reports approximately 19.53 epochs for 93 training records under the requested 20-epoch setting. This fixed schedule is shared by both arms. Held-out losses are logged for debugging and never select a checkpoint or setting.

The [protocol JSON](audit_2026-09-14/EXPERIMENT_PROTOCOL.json), [six configurations](audit_2026-09-14/run_configs), [frozen production sources](audit_2026-09-14/run_configs/source_snapshot), and per-run `run_manifest.json` identify the experiment. Full runs reuse each verified completed first fold without retraining or appending duplicates.

## Reproduce

Use the existing `.venv-test/bin/python` environment from the repository root. Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 NUMBA_CACHE_DIR=/tmp/scai-numba LD_LIBRARY_PATH=/home/ryab/miniconda3/envs/vlm_env/lib`. Select an available GPU with `CUDA_VISIBLE_DEVICES` for training. Keep long jobs in tmux.

For each arm (`base`, `long`) and seed (`0`, `1`, `2`), run:

```bash
.venv-test/bin/python -u vlm/run/run_mouse_vlm_loso.py \
  --yaml docs/audit_2026-09-14/run_configs/base_seed0.yml \
  --output-dir /data1/Processed_NIfTI_Test/embeddings/vlm/runs/validated_base_seed0
```

Change both configuration and destination for each arm/seed. A repeated command verifies and reuses complete folds; it resumes incomplete folds only from compatible retained training state. Changed fingerprints require a new output directory. Logs are `docs/audit_2026-09-14/validated_<arm>_seed<seed>.log`.

All six runs completed on 2026-09-14. Recompute their verified paired scores with:

```bash
.venv-test/bin/python scripts/aggregate_seeds.py \
  --baseline /data1/Processed_NIfTI_Test/embeddings/vlm/runs/validated_base_seed{0,1,2} \
  --longitudinal /data1/Processed_NIfTI_Test/embeddings/vlm/runs/validated_long_seed{0,1,2} \
  --truth /data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914/mouse_all_vqa_traj.json \
  --output docs/audit_2026-09-14/paired_vlm_results.json
```

The scorer rejects incomplete studies, duplicate/missing predictions, changed checkpoint files, mismatched seeds/configurations and mismatched train-mean references. The JSON includes per-horizon outcomes, per-mouse errors and source prediction directories; its Markdown companion contains the primary table.

## Input verification

```bash
.venv-test/bin/python scripts/validate_research_inputs.py \
  --report docs/audit_2026-09-14/validated_inputs.json
.venv-test/bin/python scripts/validate_nested_rollout.py \
  --forecasts /data1/Processed_NIfTI_Test/embeddings/longitudinal_nested_20260914 \
  --embeddings /data1/Processed_NIfTI_Test/embeddings/raddino/raddino_embeddings.npz \
  --report docs/audit_2026-09-14/nested_rollout_validation.json
```

For an independent proxy recomputation, `scripts/recompute_existing_pet_proxy.py` accepts `--manifest`, `--source-csv` and a separate `--output-dir`; do not overwrite the frozen dataset. Only PET-2 columns are recomputed; other historical target columns are copied and remain unvalidated.

The main CPU gate is `python -m pytest -m 'not probe and not realdata' -q`. Set `SCAI_DATA_ROOT=/data1/Processed_NIfTI_Test` to run the separate current crop-identity test. Actual quantized-backbone, corrected-encoder and data evidence is linked in the [closure report](audit_2026-09-14/PIPELINE_CLOSURE.md).

## Completed result

The six prescribed runs completed all 192 folds and passed artifact verification. Primary paired MAE difference is +0.0381 (longitudinal minus baseline), so the corrected comparison does not show an overall gain. W18 MAE is lower with longitudinal tokens in all three seeds; it remains a secondary result in 15 mice. See [all results](results.md), including every horizon, seed and training-mean comparator.

Independent arithmetic verification (does not import the production scorer):

```bash
.venv-test/bin/python scripts/check_saved_vlm_metrics.py
```

See [the final interpretation](audit_2026-09-14/FINAL_INTERPRETATION.md) for the observed prediction compression, genotype limitations and the secondary W18 result.

### Follow-up: systematically low genotype scores

Read-only checks of all 64 seed-0 baseline/longitudinal checkpoints reproduce held-out genotype logits with zero error. Median training AUROC is 0.7185/0.6450, while the identical first-fold input scored across models gives AUROC 0.4167/0.2738 against the held-out labels. These are diagnostics, not additional performance estimates. No fitting, threshold changes or score inversion was performed. V07 is reopened because reproducible pooled LOSO AUROC does not establish reliable discrimination; see [diagnosis and required closure](audit_2026-09-14/GENOTYPE_EVALUATION_FOLLOWUP.md).

```bash
CUDA_VISIBLE_DEVICES=4 .venv-test/bin/python scripts/diagnose_genotype_fold_offsets.py --run /data1/Processed_NIfTI_Test/embeddings/vlm/runs/validated_long_seed0 --output docs/audit_2026-09-14/genotype_fold_diagnostic_long0.json
CUDA_VISIBLE_DEVICES=5 .venv-test/bin/python scripts/diagnose_genotype_fold_offsets.py --run /data1/Processed_NIfTI_Test/embeddings/vlm/runs/validated_base_seed0 --output docs/audit_2026-09-14/genotype_fold_diagnostic_base0.json
```

### Authorized fixed four-fold evaluation amendment

After the pooled-LOSO follow-up, the user authorized mixed-class test folds and within-fold AUROC. The split is four-fold stratified subject CV with split seed 0, paired baseline/longitudinal arms and model seeds 0/1/2. Original model settings and targets are unchanged. All 24 forecaster fits and 24 VLM fits finished; all 384 forecast tokens, fit memberships and final model/prediction artifacts passed validation. Independent standard-library arithmetic reproduced every final genotype AUROC, accuracy and proxy MAE. Outputs and the original LOSO study remain separate. See [the frozen protocol and reproduction instructions](audit_2026-09-14/stratified/README.md). This amendment was chosen after observing the original results; no parameter search was performed.

The follow-up investigation strictly reloaded all 24 amended models, reproduced every held-out genotype score exactly, and compared training versus held-out class gaps, training-centroid responses and inference precision. All models have the intended average class direction in training; many reverse it on held-out mice. A separate four-model longitudinal seed-0 check reproduced all 96 held-out head outputs across the three question types. No saved model, threshold or research prediction was changed. See [the mechanism report and diagnostic artifacts](audit_2026-09-14/stratified/GENOTYPE_MECHANISM.md).

### Real-mouse learning capacity control

The user authorized a training-only control on eight existing training mice, with matched combined versus genotype-only objectives. Both longitudinal models received the same 24 records, seed-0 initial parameters, inputs, optimizer and 300-update budget. The combined objective met the declared 8/8-correct plus BCE ≤ 0.1 criterion by the measurement at update 120; genotype-only met it by update 180. Both finished all 300 updates. The combined objective therefore does not prevent learning this selected real-data subset under an extended budget. The KO subset is one acquisition component, and these are memorization diagnostics rather than new research-performance estimates.

An initial diagnostic reload assertion compared automatic Trainer mixed precision against ordinary inference. The probe was corrected to make precision explicit; saved endpoints then reproduced exactly, and final tensors matched checkpoint 300, without further training or weight changes. Ordinary inference also retains 8/8 correct labels in both controls. Swapping image inputs makes outputs follow donor images exactly; identical images remove all between-mouse genotype-score differences. [Complete results and evidence](audit_2026-09-14/real_mouse_learning_control/RESULTS.md). Original study outputs remain unchanged.

### Direct genotype classification of corrected longitudinal embeddings

At the user's request, a fixed linear readout was evaluated on the existing corrected embeddings, independently of the VLM. The three representations were fixed before fitting: baseline alone, three predicted future tokens concatenated, and baseline plus those predictions. All use the same validated four subject folds, verified nested forecasts, a training-only StandardScaler, and L2 logistic regression with C=1; no parameter search or new forecaster/VLM training. All 12 classifiers converged. Within-fold pair-weighted AUROC is 0.6290 / 0.4516 / 0.5968 respectively; correct labels are 21/32, 15/32, and 20/32. Mean training AUROC is 1.0000 / 0.9983 / 1.0000. This shows modest baseline association but no benefit from the predicted tokens for this fixed readout. Acquisition/diet confounding, high dimensionality relative to sample size, and the cross-fitted training versus outer-fit test representation difference remain limitations. [Protocol, full results, and saved artifacts](audit_2026-09-14/stratified/embedding_probe/RESULTS.md).

### Residual full-rollout development experiment — completed 2026-09-16

The user authorized one fixed forecaster variant: a baseline residual connection, zero-initialized residual output, full-rollout training against every available future embedding, and a 0.01 residual penalty. The existing 300-epoch budget, 512-wide hidden layers, learning rate, outer folds, inner cross-fitting, and fit-role seeds were retained. No genotype supervision or PET targets were added. Settings and sources were frozen before fitting; checkpoints and exports use a separate directory.

All 24 forecasters and 384 tokens completed, with exact checkpoint reload. A fixed linear readout scores residual future tokens at AUROC 0.6774 (19/32 correct), compared with baseline 0.6290 (21/32) and current future tokens 0.4516 (15/32). Residual plus baseline also scores 0.6774 (19/32). All earlier classifier predictions reproduce exactly. The residual ranking advantage over baseline is three of 62 within-fold pairs, with one test fold still below chance; it is descriptive development evidence.

Forecast fidelity worsens: subject-mean future cosine 0.9808 versus current 0.9827 and training-week centroid 0.9866; persistence is 0.9759. The residual model is worse than current and centroid at all three horizons. The predeclared joint genotype/fidelity advancement criterion failed, so no VLM run was included in that encoder experiment. A later user-authorized VLM follow-up is recorded below. Independent calculations verified all 20 classifiers, token/source hashes, and 67 observed-future comparisons. [Full results, interpretation, protocol, and verification](audit_2026-09-14/residual_rollout_experiment/README.md).

### User-authorized residual-embedding VLM follow-up — complete

After reviewing the raw-embedding result, the user explicitly requested a VLM evaluation. The separate follow-up froze the same model, three seeds, four subject folds, 20-epoch schedule, question-boundary supervision, loss weights, and inference used in the completed amendment. Only the nested forecast source and its validation route changed. All six existing comparator runs passed preflight, and the production dataset loader exactly reproduced every residual four-token input. Two routing/configuration tests passed. All 12 new fold fits completed on GPU 6, followed by artifact validation and independent scoring of all nine runs. [Frozen protocol, results, and run details](audit_2026-09-14/residual_vlm/README.md).

Genotype AUROC is 0.5484 for each residual VLM seed, improving on both matched VLM comparators in all three seeds. The identical totals arise from different score vectors and fold contributions, each summing to 34 of 62 correctly ordered pairs. Accuracy is 17/32, 16/32, and 17/32, below the always-WT comparator. Mean primary proxy MAE remains worse than baseline: 5.6564 versus 5.5506; current longitudinal is 5.6631 and the training-mean comparator is 5.6106. Week-20 MAE is lower in all three residual runs than all three comparators (20 mice), with mean 1.9662 versus baseline 2.0704. This is a secondary development result on repeatedly inspected folds; all horizons and seeds remain reported and no independent-group significance or biological disease-progression claim is made.

### Fixed genotype VLM variants — complete

Two user-authorized variants started on GPUs 1 and 6 after a successful frozen-input preflight: direct visual access (a zero-initialized linear score from training-standardized image tokens added to the language-state genotype score, with the combined objective unchanged), and genotype-only supervision (original architecture with language/proxy losses removed). Each retains the residual VLM's four folds, seeds 0/1/2, 20 epochs/180 updates, optimizer, batching, and zero-logit threshold. Six focused tests and actual-backbone objective, gradient, answer-invariance, and exact-reload checks passed. Variant sources/checkpoints are isolated from all earlier runs.

All 24 fits and exact final-checkpoint inference reloads completed. Genotype-only seed 2 was parallelized on free GPU 4 without changing its frozen configuration; the main queue subsequently verified and skipped it. Independent scoring reproduced all 12 new/comparator runs, and normalization/shared-initial-weight checks passed. Direct-visual AUROC is 0.6935/0.6774/0.6935 (mean 0.6882), but accuracy is 16/32 in every seed. Genotype-only AUROC is 0.6613/0.5484/0.5323 (mean 0.5806), with accuracy 17/32, 16/32, 17/32. Direct-visual mean proxy MAE is 5.5550 versus baseline 5.5506; genotype-only proxy outputs are unsupervised and excluded.

Descriptive inspection of the completed scores finds all-WT predictions in folds 00 and 03 for both variants in every seed, despite perfect direct-visual AUROC in fold 03. Fold 01 remains below chance. Direct-visual errors have mean raw confidence about 0.83, so stronger scores have not solved classification. No thresholds or settings were selected from these observations. Direct-image component scores are descriptive diagnostics of a jointly trained model, and largely retain the combined model's ranking performance. These experiments compare architecture/objective changes on residual inputs; they do not isolate the benefit of longitudinal tokens within each new architecture. [Protocol, all results, source snapshots, and runtime checks](audit_2026-09-14/genotype_vlm_variants/README.md).

### Documentation reconciliation — 2026-09-21

Current guides now distinguish direct embedding classification, forecast fidelity, downstream VLM metrics and the additive logit-level late-fusion hybrid. Known repairs and targeted checks are not described as proof that all methodological issues are closed. Recommended ablations remain unrun. Frozen reports and completion manifests retain their original scope. The genotype mechanism report was restored to its exact completion hash; its two later introductory paragraphs remain in a separately preserved annotated version. No research predictions, model weights, metrics, thresholds, or protocol settings changed. See [current status](STATUS.md) and [evidence scope](audit_2026-09-14/README.md).

The current pre-commit CPU suite passed 135 tests with 18 probe/real-data cases deselected in 162.24 seconds. This verifies the current source tree without rerunning research experiments; the [test guide](../tests/README.md#current-pre-commit-gate-2026-09-21) records the command and output.
