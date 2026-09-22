# Checkpoint and head-supervision repairs

> This is the checkpoint/head repair-stage record. Later completed protocols and the remaining methodological limitations are summarized in [current status](../STATUS.md).

Implemented 2026-09-14 at the user's request. This closes the implementation defects covered here; it does not validate the remaining data, forecast nesting, or statistical issues in the master plan, and it produces no new biological performance claim.

## Changes

- Final and intermediate checkpoints use PEFT's canonical adapter serialization. The raw `.default` key overwrite is removed. The loader populates the existing prepared adapter shell once and verifies every adapter tensor exactly.
- Checkpoints include the projection, both heads, TBR mean/std buffers, and versioned model/pooling metadata. Missing files, missing/extra tensors, wrong shapes, nonfinite weights, incompatible LoRA settings, changed pooling, and invalid normalization fail loading. An interrupted replacement lacks its completion metadata and fails loading.
- Trainer resume and best-checkpoint restoration use the same compact format. Resume restores optimizer, scheduler and random state; absent required state fails. A scoped NumPy allowlist fixes the installed Transformers/PyTorch random-state deserialization incompatibility.
- Final per-fold adapters are retained after successful inference. Intermediate checkpoints are removed at fold completion as before; originals from historical runs were not changed.
- Both heads now default to an explicit question-prefix boundary. The dataset supplies its position before image expansion; the model applies the image-token offset and validates padding/bounds. Inference derives the same boundary from the question-only input. Chat prompts do not depend on a first-EOS guess.
- Question tokens match inference tokenization exactly. The answer retains causal-language supervision. Answer truncation warns and preserves the head boundary; a question too long to leave any answer supervision fails. Training and inference use matching configured prompt wrappers.
- Prediction records include pooling and checkpoint-format provenance. Calibration status is `not_assessed`; matching context does not establish calibration.

`answer_eos` remains an explicit legacy reproduction option. Existing experiment YAMLs and old result directories retain their provenance. Use the new matched configurations:

- `vlm/yaml/viz_emb_params_mouse_nogeno_base_repaired.yml`
- `vlm/yaml/viz_emb_params_mouse_nogeno_long_repaired.yml`

Both use question pooling and fresh output directories. Subsequent closure work made LOSO default to the repaired nested longitudinal protocol and the generic single-run entry point default to baseline (a global multi-token split is rejected). All 32 nested exports are now available and the paired sweep is complete; see [the current closure record](PIPELINE_CLOSURE.md). The tests below describe the earlier checkpoint/head repair stage.

## Verification

- CPU checkpoint diagnostic: exact adapter, language-logit, genotype-head and regression-head equality across save/reload. Nonzero B matrices remain nonzero. See `checkpoint_roundtrip_fixed.log`.
- Behavioral tests cover one/four image tokens, different answers and lengths, earlier chat EOS tokens, padding, missing/invalid boundaries, image-count mismatches, corrupted checkpoint components, and legacy adapter recovery.
- A small synthetic image-to-target fit verifies gradients and updates through projection, both heads and both adapter matrices; predictions retain the learned signal after reload. Shuffling the known image signal changes predictions. This is a plumbing test only.
- Four-step training matches a two-step interruption plus resume exactly, including all model weights. Best-checkpoint restoration matches the saved two-step model. See `checkpoint_resume_fixed.log`.
- All 96 actual VQA records have training prefixes identical to inference, with zero standard-tokenization changes and zero truncated records (maximum 80 tokens, limit 150). See `question_prefix_validation.json`.
- Actual NF4 LLaMA-3.1-8B / NVIDIA L40S: two training updates produced finite gradients through both heads, projection and both adapter matrices. All 128 adapter tensors were restored exactly; all 64 B tensors remained nonzero. Language logits and both heads had maximum reload error **0.0**, matching repeated same-model evaluation error **0.0**. Answer perturbations changed neither head (maximum error **0.0**). Both recent recovered checkpoints also passed strict actual-model loading and finite-output checks. See `quantized_checkpoint_fixed.json` and `.log`.
- Final CPU gate: **85 passed, 1 pre-existing failure, 18 deselected**. All 26 new regression cases passed. See `tests_checkpoint_heads_fixed.log`.

The CPU suite has one pre-existing unrelated failure: `tests/pipeline/test_mouse_identity.py::test_count_mismatch_is_a_hard_error` checks stale source text. Its existing behavioral mismatch-rejection test passes. Subsequent closure work replaced the stale assertion, tested override coverage and validated the current data. The final expanded CPU gate passed 114 tests; see [the current closure record](PIPELINE_CLOSURE.md).

## Retained adapter recovery

No historical tmux sessions remained when this implementation started. Six full Trainer checkpoints survived. Their relevant tensors were extracted into a separate tree:

`/data1/Processed_NIfTI_Test/embeddings/vlm/runs/repaired_checkpoints_20260914/`

The original large checkpoints remain intact. Each recovery records source/config SHA-256 hashes and original pooling in `recovery.json`; the audit manifest is `adapter_recovery.json`.

| Original run / checkpoint | Recovery |
|---|---|
| `mouse_vlm_nogeno_long_ep20_f20`, fold 11 / step 112 | Complete component set: 128 adapter tensors, 64 nonzero B tensors; original answer pooling |
| `mouse_vlm_nogeno_long_ep20_qeos_seed0`, fold 08 / step 67 | Complete component set: 128 adapter tensors, 64 nonzero B tensors; original question pooling |
| `mouse_vlm_exp1_mtwt5`, fold 16 / step 11 | Adapter/projection/heads extracted; missing normalization buffers; incomplete |
| `mouse_vlm_exp2_freeze_llm`, fold 05 / step 22 | Incomplete; missing normalization; B tensors are zero in the source |
| `mouse_vlm_exp2_freeze_llm`, fold 21 / steps 67 and 101 | Incomplete; missing normalization; B tensors are zero in the sources |

Incomplete recoveries deliberately have no current-format model metadata and cannot silently load with invented normalization. Zero source B tensors are preserved, not described as trained deltas. Both complete component sets passed actual-model strict reload verification in `quantized_checkpoint_fixed.json`. Their original recovery manifests intentionally retain the extraction-time status; the separate validation report supplies the subsequent evidence.

These are retained intermediate checkpoints, not complete final-epoch LOSO results. Recovery cannot reconstruct deleted folds or convert answer-supervised training into question-supervised training. Repaired old checkpoints do not establish longitudinal improvement.

## Reproduction

From the repository root, with the existing `vlm_env`-derived environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  .venv-test/bin/python -m pytest tests/torch/test_vlm_checkpoint_and_heads.py -q
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  .venv-test/bin/python scripts/audit_checkpoint_roundtrip.py
```

Quantized validation (use a free GPU and a persistent session):

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  .venv-test/bin/python scripts/validate_vlm_quantized_checkpoint.py \
  --config vlm/yaml/viz_emb_params_mouse_nogeno_long_repaired.yml \
  --report /tmp/quantized_checkpoint_check.json \
  --repaired-root /data1/Processed_NIfTI_Test/embeddings/vlm/runs/repaired_checkpoints_20260914
```

The quantized test uses two training updates on current inputs and verifies reload; it is not a held-out evaluation. A standalone repair command is available as `scripts/repair_vlm_checkpoint.py --help`; it requires the original experiment YAML and refuses to overwrite an existing output directory.
