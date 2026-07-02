# VLM Experiment Log

Canonical record of all LOSO CV runs. Each entry documents exactly what changed,
why, and what the results were. Primary metrics for the paper: genotype accuracy
and TBR regression MAE (Δ3wk). r and R² are reported for analysis only.

All experiments use: 32 NaF subjects, LOSO CV, RAD-DINO embeddings (768-d),
4-token longitudinal input (ts0 real + ts1/ts2/ts3 MLP-predicted), multitask
head on LLM EOS hidden state.

---

## Baseline experiments (completed)

### `mouse_vlm_baseline_loso` — VLM baseline (ts0 only)
- **YAML:** `viz_emb_params_mouse.yml` (img_tokens=1, no predicted_emb_dir)
- **Change from default:** single image token, no longitudinal input
- **Purpose:** lower bound — what can the VLM do from a single scan?
- **Results:**
  - Genotype acc: **0.219**
  - TBR reg MAE (overall): 6.241
  - TBR reg MAE (Δ3wk): 6.806, r=−0.090, R²=−0.088, n=64
  - TBR reg MAE (Δ6wk): 7.568, r=−0.243, R²=−0.957, n=30
  - TBR reg MAE (Δ8wk): 4.340, r=0.069, R²=−4.421, n=40

### `mouse_vlm_loso` — VLM longitudinal (exp3 config, 10 epochs)
- **YAML:** `viz_emb_params_mouse_exp3_combined.yml`
- **Change from baseline:** 4-token input (ts0+MLP-predicted ts1/ts2/ts3),
  r=4 q/v LoRA, multitask_wt=5, z-scored TBR targets
- **Purpose:** primary longitudinal result
- **Results:**
  - Genotype acc: **0.531**
  - TBR reg MAE (overall): 5.393
  - TBR reg MAE (Δ3wk): 6.123, r=0.447, R²=0.131, n=64
  - TBR reg MAE (Δ6wk): 6.855, r=−0.093, R²=−0.722, n=30
  - TBR reg MAE (Δ8wk): 3.127, r=0.207, R²=−1.826, n=40

---

## Signal-loss investigation experiments

Run with: `cd ~/ScAI-Lab/vlm && CUDA_VISIBLE_DEVICES=0 bash run/run_signal_loss_experiments.sh`

All experiments below differ from the `mouse_vlm_loso` baseline by **one variable only**.

### `mouse_vlm_ep20` — Epoch ablation: 20 epochs
- **YAML:** `viz_emb_params_mouse_ep20.yml`
- **Change:** `num_train_epochs: 20` (was 10)
- **Hypothesis:** model is underfitting; more epochs should improve both genotype acc and TBR MAE
- **Results:** *(pending)*

### `mouse_vlm_ep50` — Epoch ablation: 50 epochs
- **YAML:** `viz_emb_params_mouse_ep50.yml`
- **Change:** `num_train_epochs: 50` (was 10)
- **Hypothesis:** continued underfitting past 20 epochs
- **Results:** *(pending)*

### `mouse_vlm_ep100` — Epoch ablation: 100 epochs
- **YAML:** `viz_emb_params_mouse_ep100.yml`
- **Change:** `num_train_epochs: 100` (was 10)
- **Hypothesis:** upper bound on epoch scaling; may show overfitting on this small dataset
- **Results:** *(pending)*

### `mouse_vlm_tinyllama` — TinyLlama-1.1B backbone
- **YAML:** `viz_emb_params_mouse_tinyllama.yml`
- **Change:** `llm_model_name` and `tokenizer_name` → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Hypothesis:** LLaMA-3.1-8B may be overparameterized for a structured embedding
  regression task; a smaller LLM may overfit less and preserve linear signal better
- **Results:** *(pending)*

---

## Reference: linear probe upper bounds

| Condition | Geno acc | Geno AUC | TBR Δ3wk r |
|---|---|---|---|
| RAD-DINO linear probe, ts0 only | 0.406 | 0.353 | −0.250 |
| RAD-DINO linear probe, all 4 real timepoints | 0.406 | 0.401 | 0.350 |
| Longitudinal linear probe (ts0 + MLP-predicted) | **0.750** | **0.718** | 0.030 |

Note: the longitudinal linear probe (0.750) is an inflated upper bound — the MLP
conditioning vector includes genotype as an explicit input feature, so predicted
embeddings implicitly encode the label.
