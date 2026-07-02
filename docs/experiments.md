# VLM Experiment Log

Canonical record of all LOSO CV runs. Each entry documents exactly what changed,
why, and what the results were. Primary metrics for the paper: genotype accuracy
and TBR regression MAE (Δ3wk). r and R² are reported for analysis only.

All experiments use: 32 NaF subjects, LOSO CV, RAD-DINO embeddings (768-d),
multitask head on LLM EOS hidden state, LLaMA-3.1-8B-Instruct unless noted.

---

## Completed experiments

### `mouse_vlm_baseline_loso` — VLM baseline (ts0 only)
- **YAML:** `viz_emb_params_mouse.yml`
- **Key settings:** img_tokens=1, no predicted_emb_dir, r=16 full LoRA, multitask_wt=1, 10 epochs
- **Purpose:** lower bound — what can the VLM do from a single scan?
- **Results:**
  - Genotype acc: **0.219**
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
  - Genotype acc: **0.531**
  - TBR reg MAE (overall): 5.393, r=0.276, R²=−0.076
  - TBR reg MAE (Δ3wk): 6.123, r=0.447, R²=0.131, n=64
  - TBR reg MAE (Δ6wk): 6.855, r=−0.093, R²=−0.722, n=30
  - TBR reg MAE (Δ8wk): 3.127, r=0.207, R²=−1.826, n=40

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

## Signal-loss investigation experiments (pending)

These all use the `mouse_vlm_loso` config as the baseline and change **one variable only**.

Run with:
```bash
cd ~/ScAI-Lab/vlm
CUDA_VISIBLE_DEVICES=0 bash run/run_signal_loss_experiments.sh
```

### `mouse_vlm_ep20` — epoch ablation: 20 epochs
- **YAML:** `viz_emb_params_mouse_ep20.yml`
- **Change:** `num_train_epochs: 20` (was 10)
- **Hypothesis:** model is underfitting at 10 epochs; more training should improve genotype acc and TBR MAE
- **Results:** *(pending)*

### `mouse_vlm_ep50` — epoch ablation: 50 epochs
- **YAML:** `viz_emb_params_mouse_ep50.yml`
- **Change:** `num_train_epochs: 50` (was 10)
- **Hypothesis:** continued underfitting past 20 epochs
- **Results:** *(pending)*

### `mouse_vlm_ep100` — epoch ablation: 100 epochs
- **YAML:** `viz_emb_params_mouse_ep100.yml`
- **Change:** `num_train_epochs: 100` (was 10)
- **Hypothesis:** upper bound on epoch scaling; may reveal overfitting on this small dataset (32 subjects)
- **Results:** *(pending)*

### `mouse_vlm_tinyllama` — TinyLlama-1.1B backbone
- **YAML:** `viz_emb_params_mouse_tinyllama.yml`
- **Change:** `llm_model_name` + `tokenizer_name` → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Hypothesis:** LLaMA-3.1-8B may be overparameterized for this structured embedding regression task;
  a smaller LLM may overfit less and better preserve the linear signal that the probe can decode
- **Results:** *(pending)*

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
