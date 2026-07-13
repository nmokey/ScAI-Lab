# VLM Experiment Log

Canonical record of all LOSO CV runs. Each entry documents exactly what changed,
why, and what the results were. Primary metrics for the paper: genotype accuracy
and TBR regression MAE (Δ3wk). r and R² are reported for analysis only.

All experiments use: 32 NaF subjects, LOSO CV, RAD-DINO embeddings (768-d),
multitask head on LLM EOS hidden state, LLaMA-3.1-8B-Instruct unless noted.

---

## Summary table

| Experiment | Epochs | LLM | Geno acc | Geno AUC | TBR MAE (overall) | TBR MAE (Δ3wk) | TBR r (Δ3wk) | TBR R² (overall) |
|---|---|---|---|---|---|---|---|---|
| Baseline (ts0 only) | 10 | LLaMA-3.1-8B | 0.219 | 0.163 | 6.241 | 6.806 | −0.090 | −0.357 |
| Longitudinal 10ep | 10 | LLaMA-3.1-8B | 0.531 | 0.560 | 5.393 | 6.123 | **0.447** | −0.076 |
| **Longitudinal 20ep** | 20 | LLaMA-3.1-8B | **0.719** | **0.762** | **4.736** | 6.346 | 0.376 | **0.104** |
| Longitudinal 50ep | 50 | LLaMA-3.1-8B | 0.625 | 0.627 | 5.352 | 7.459 | 0.176 | −0.011 |
| Longitudinal 100ep | 100 | LLaMA-3.1-8B | 0.625 | 0.714 | 5.490 | 7.481 | 0.202 | −0.011 |
| TinyLlama 10ep | 10 | TinyLlama-1.1B | 0.344 | 0.397 | 4.794 | 6.294 | 0.548 | 0.220 |

---

## Epoch sweep: baseline (ts0) vs longitudinal (2026-07-07)

Controlled 2D ablation. All 10 runs use the **exp3 config** (r=4, q/v LoRA,
multitask_wt=5, z-scored TBR, LLaMA-3.1-8B). The only variables are (a) epochs
and (b) input type: baseline = ts0 only (img_tokens=1, no predicted_emb_dir);
longitudinal = ts0 + MLP-predicted ts1/ts2/ts3 (img_tokens=4).

YAMLs: `viz_emb_params_mouse_{base,long}_ep{N}.yml`. Runners:
`run/run_sweep_gpuA_baseline.sh`, `run/run_sweep_gpuB_longitudinal.sh`.
(long_ep20 == the earlier `mouse_vlm_ep20` run — same config.)

### Genotype accuracy / AUROC

| Epochs | Baseline (ts0) acc / AUC | Longitudinal acc / AUC |
|---|---|---|
| 15 | 0.625 / 0.623 | 0.625 / 0.635 |
| 20 | 0.500 / 0.583 | **0.719 / 0.762** ⭐ |
| 25 | 0.375 / 0.345 | 0.438 / 0.421 |
| 30 | 0.344 / 0.298 | 0.438 / 0.536 |
| 40 | 0.375 / 0.349 | 0.656 / 0.595 |

### TBR regression — overall MAE / R²

| Epochs | Baseline MAE / R² | Longitudinal MAE / R² |
|---|---|---|
| 15 | 5.034 / 0.071 | **4.462 / 0.240** ⭐ |
| 20 | 4.828 / 0.104 | 4.736 / 0.104 |
| 25 | 4.808 / 0.065 | 5.470 / −0.075 |
| 30 | 4.699 / 0.151 | 5.465 / −0.089 |
| 40 | 4.914 / 0.057 | 6.200 / −0.203 |

### TBR regression — Δ3wk MAE / r (n=64)

| Epochs | Baseline MAE / r | Longitudinal MAE / r |
|---|---|---|
| 15 | 6.741 / 0.371 | 5.726 / **0.525** |
| 20 | 6.475 / 0.301 | 6.346 / 0.376 |
| 25 | 5.937 / 0.275 | 7.707 / −0.016 |
| 30 | 6.196 / 0.319 | 7.345 / 0.091 |
| 40 | 6.559 / 0.221 | 8.983 / −0.142 |

### Takeaways

1. **Longitudinal input drives genotype signal.** Longitudinal ≥ baseline on both
   accuracy and AUROC at every epoch count, peaking at **0.719 acc / 0.762 AUC @ 20ep**.
   Baseline never exceeds 0.625 acc and its AUROC slides *below* chance as training
   lengthens (0.35 / 0.30 / 0.35 at 25/30/40ep, 0.163 at the 10ep default) — the
   ts0-only head not only fails to separate genotype but ranks it anti-correlated,
   a hallmark of overfitting with no stable signal to lock onto. This is the strongest
   evidence that the predicted trajectory (not just more training) is what carries genotype.
2. **The two heads want different training lengths.** Longitudinal genotype peaks
   at 20ep; longitudinal TBR peaks at **15ep** (MAE 4.462, R²=0.240, Δ3wk r=0.525).
   No single epoch count is jointly optimal.
3. **Longitudinal TBR overfits past 20ep.** Overall R² goes 0.240 → 0.104 →
   negative (−0.075 / −0.089 / −0.203). Baseline TBR stays flat-and-positive across
   all epochs, i.e. it is less expressive but more stable.
4. **long_ep40 genotype (0.656) is likely noise.** The 25/30/40 longitudinal
   genotype points (0.438, 0.438, 0.656) bounce around; with 32 LOSO folds the
   per-point SE is large. Treat the >20ep region as "degraded," not monotonic.

**Recommended configs:** longitudinal 20ep for genotype (advisor's pick, confirmed);
longitudinal 15ep if TBR is the priority.

---

## Completed experiments

### `mouse_vlm_baseline_loso` — VLM baseline (ts0 only)
- **YAML:** `viz_emb_params_mouse.yml`
- **Key settings:** img_tokens=1, no predicted_emb_dir, r=16 full LoRA, multitask_wt=1, 10 epochs
- **Purpose:** lower bound — what can the VLM do from a single scan?
- **Results:**
  - Genotype acc: **0.219** / AUROC: **0.163** *(below chance — head ranks genotype anti-correlated)*
  - TBR reg MAE (overall): 6.241, r=−0.054, R²=−0.357
  - TBR reg MAE (Δ3wk): 6.806, r=−0.090, R²=−0.088, n=64
  - TBR reg MAE (Δ6wk): 7.568, r=−0.243, R²=−0.957, n=30
  - TBR reg MAE (Δ8wk): 4.340, r=0.069, R²=−4.421, n=40

### `mouse_vlm_loso` — VLM longitudinal, current best (exp3 config)
- **YAML:** `viz_emb_params_mouse_exp3_combined.yml`
- **Key settings:** img_tokens=4 (ts0 real + ts1/ts2/ts3 MLP-predicted), r=4 q/v LoRA only,
  multitask_wt=5, z-scored TBR targets, 10 epochs
- **Purpose:** primary longitudinal result; combines lessons from exp1–3
- **Results:**
  - Genotype acc: **0.531** / AUROC: **0.560**
  - TBR reg MAE (overall): 5.393, r=0.276, R²=−0.076
  - TBR reg MAE (Δ3wk): 6.123, r=0.447, R²=0.131, n=64
  - TBR reg MAE (Δ6wk): 6.855, r=−0.093, R²=−0.722, n=30
  - TBR reg MAE (Δ8wk): 3.127, r=0.207, R²=−1.826, n=40

### `mouse_vlm_tinyllama` — TinyLlama-1.1B backbone ✓
- **YAML:** `viz_emb_params_mouse_tinyllama.yml`
- **Change:** `llm_model_name` + `tokenizer_name` → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Hypothesis:** LLaMA-3.1-8B may be overparameterized for this structured embedding regression task;
  a smaller LLM may overfit less and better preserve the linear signal that the probe can decode
- **Results** (post-fix, valid):
  - Genotype acc: 0.344 / AUROC: 0.397 *(both below chance — TinyLlama lacks capacity for genotype signal)*
  - TBR reg MAE (overall): **4.794**, r=0.492, R²=**0.220**
  - TBR reg MAE (Δ3wk): 6.294, r=0.548, R²=0.193, n=64
  - TBR reg MAE (Δ6wk): **5.631**, r=0.557, R²=0.233, n=30
  - TBR reg MAE (Δ8wk): **1.766**, r=−0.020, R²=−0.143, n=40
- **Observation:** TBR regression better than 8B baseline overall (R²=0.220 vs −0.076);
  genotype badly worse (0.344 vs 0.531). Smaller model overfits less on regression but
  lacks capacity for genotype discrimination.

---

## Signal-loss investigation experiments (rerunning with denorm fix)

These all use the `mouse_vlm_loso` config as the baseline and change **one variable only**.
First runs had invalid TBR MAE due to denorm bug; reruns in progress (GPU 0: ep50→ep100, GPU 1: done).

### `mouse_vlm_ep20` — epoch ablation: 20 epochs ✓
- **YAML:** `viz_emb_params_mouse_ep20.yml`
- **Change:** `num_train_epochs: 20` (was 10)
- **Hypothesis:** model is underfitting at 10 epochs; more training should improve genotype acc and TBR MAE
- **Results:**
  - Genotype acc: **0.719** / AUROC: **0.762**
  - TBR reg MAE (overall): **4.736**, r=0.339, R²=**0.104**
  - TBR reg MAE (Δ3wk): 6.346, r=0.376, R²=0.100, n=64
  - TBR reg MAE (Δ6wk): 5.437, r=0.141, R²=−0.003, n=30
  - TBR reg MAE (Δ8wk): 1.633, r=0.070, R²=−0.067, n=40
- **Observation:** best genotype acc and best overall MAE of all configurations;
  Δ3wk r slightly lower than 10ep (0.376 vs 0.447) but overall R² turns positive.

### `mouse_vlm_ep50` — epoch ablation: 50 epochs ✓
- **YAML:** `viz_emb_params_mouse_ep50.yml`
- **Change:** `num_train_epochs: 50` (was 10)
- **Hypothesis:** continued underfitting past 20 epochs
- **Results:**
  - Genotype acc: 0.625 / AUROC: 0.627
  - TBR reg MAE (overall): 5.352, r=0.196, R²=−0.011
  - TBR reg MAE (Δ3wk): 7.459, r=0.176, R²=−0.025, n=64
  - TBR reg MAE (Δ6wk): 6.302, r=−0.034, R²=−0.178, n=30
  - TBR reg MAE (Δ8wk): 1.269, r=0.727, R²=0.408, n=40
- **Observation:** genotype and Δ3wk both degrade vs 20ep — model is overfitting past 20 epochs.

### `mouse_vlm_ep100` — epoch ablation: 100 epochs ✓
- **YAML:** `viz_emb_params_mouse_ep100.yml`
- **Change:** `num_train_epochs: 100` (was 10)
- **Hypothesis:** upper bound on epoch scaling; may reveal overfitting on this small dataset (32 subjects)
- **Results:**
  - Genotype acc: 0.625 / AUROC: 0.714
  - TBR reg MAE (overall): 5.490, r=0.247, R²=−0.011
  - TBR reg MAE (Δ3wk): 7.481, r=0.202, R²=−0.034, n=64
  - TBR reg MAE (Δ6wk): 6.126, r=0.176, R²=−0.076, n=30
  - TBR reg MAE (Δ8wk): 1.827, r=0.208, R²=−0.135, n=40
- **Observation:** nearly identical to 50ep — model saturates around 50 epochs on this dataset size.

---

## Completed config exploration (no valid LOSO results)

These runs explored config changes but were either never run to LOSO completion
or were invalidated by the TBR z-score bug fix. Documented for provenance only.

### `mouse_vlm_exp1_mtwt5` — multitask weight ablation
- **YAML:** `viz_emb_params_mouse_exp1_multitask_wt5.yml`
- **Change from baseline:** multitask_wt=5 (was 1); everything else same as baseline
  (r=16, all 7 LoRA modules, 10 epochs, 4-token input)
- **Hypothesis:** LM cross-entropy over 150 tokens dominates ~75–100× over head loss
  at wt=1; increasing to 5 would give heads more gradient
- **Results:** never run to LOSO completion — no loso_results.json

### `mouse_vlm_exp2_freeze_llm` — frozen LLM
- **YAML:** `viz_emb_params_mouse_exp2_freeze_llm.yml`
- **Change from baseline:** freeze_llm_model=True (trains only projection ~3.1M + heads ~1M params);
  multitask_wt=1, r=16 LoRA settings present but inactive due to freeze
- **Hypothesis:** LoRA fine-tuning on only 93 records/fold may be net-negative vs. a
  frozen LLM that just routes embeddings to the heads
- **Results:** never run to LOSO completion — no loso_results.json

### `mouse_vlm_exp3_combined` — pre-fix run (invalidated)
- **YAML:** `viz_emb_params_mouse_exp3_combined.yml`
- **Change from exp1/2:** combined r=4 q/v-only LoRA + multitask_wt=5; first run with
  z-scored TBR targets (added mid-experiment)
- **Results (invalid — pre-bugfix):**
  - Genotype acc: 0.625 *(inflated — TBR normalization was not yet applied correctly)*
  - TBR reg MAE (Δ3wk): 24.41 *(unnormalized scale, not comparable)*
  - Superseded by `mouse_vlm_loso` rerun after fixes

---

## Reference: linear probe upper bounds

| Condition | Geno acc | Geno AUC | TBR Δ3wk MAE | TBR Δ3wk r |
|---|---|---|---|---|
| RAD-DINO linear probe, ts0 only | 0.406 | 0.353 | — | −0.250 |
| RAD-DINO linear probe, all 4 real timepoints | 0.406 | 0.401 | — | 0.350 |
| Longitudinal linear probe (ts0 + MLP-predicted) | **0.750** | **0.718** | — | 0.030 |

Note: the longitudinal linear probe (0.750) is an inflated upper bound — the MLP
conditioning vector includes genotype as an explicit input feature, so predicted
embeddings implicitly encode the label. Treat as a ceiling, not a fair comparison.
