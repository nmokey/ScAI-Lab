# VLM Experiment Log

> [!WARNING]
> **Every row in this log dated before 2026-09-09 was produced with the genotype label
> injected into ts1–3 (F10) and the checkpoint selected on the held-out subject (F2), and
> the whole epoch sweep ran with `load_best_model_at_end=True`.** Those rows are archived
> for provenance and must not be cited. The only citable VLM rows are the two marked
> "leak-free re-run" in the summary table and detailed in the section of that name.
> Register: [FINDINGS.md](FINDINGS.md).

> The epoch sweep specifically: the reported best cell (20 epochs, genotype AUROC
> 0.762) has p = 0.046 against a no-signal null once treated as the maximum over
> the 10 configurations it was selected from, before any correction for F2's
> measured +0.14 to +0.20 selection inflation. "20 epochs is the sweet spot" is
> not supported as stated.


Canonical record of all LOSO CV runs. Each entry documents exactly what changed,
why, and what the results were. Primary metrics for the paper: genotype accuracy
and TBR regression MAE (Δ3wk). r and R² are reported for analysis only.

All experiments use: 32 NaF subjects, LOSO CV, RAD-DINO embeddings (768-d),
multitask head on LLM EOS hidden state, LLaMA-3.1-8B-Instruct unless noted.

---

## Summary table

All rows use LLaMA-3.1-8B-Instruct and the **canonical config** (r=4 q/v LoRA,
multitask_wt=5, z-scored TBR) except where the Config column says otherwise. The
baseline (ts0-only) and longitudinal (4-token) rows at 15–40 epochs are the
controlled epoch sweep — same config, varying only epochs and input type; see the
[epoch-sweep section](#epoch-sweep-baseline-ts0-vs-longitudinal-2026-07-07) below for
the head-to-head breakdown.

| Experiment | Config | Epochs | Geno acc | Geno AUC | TBR MAE (overall) | TBR MAE (Δ3wk) | TBR r (Δ3wk) | TBR R² (overall) |
|---|---|---|---|---|---|---|---|---|
| Baseline (ts0) | old (r=16, mtwt=1) | 10 | 0.219 | 0.163 | 6.241 | 6.806 | −0.090 | −0.357 |
| Baseline (ts0) | canonical | 15 | 0.625 | 0.623 | 5.034 | 6.741 | 0.371 | 0.071 |
| Baseline (ts0) | canonical | 20 | 0.500 | 0.583 | 4.828 | 6.475 | 0.301 | 0.104 |
| Baseline (ts0) | canonical | 25 | 0.375 | 0.345 | 4.808 | 5.937 | 0.275 | 0.065 |
| Baseline (ts0) | canonical | 30 | 0.344 | 0.298 | 4.699 | 6.196 | 0.319 | 0.151 |
| Baseline (ts0) | canonical | 40 | 0.375 | 0.349 | 4.914 | 6.559 | 0.221 | 0.057 |
| Longitudinal | canonical | 10 | 0.531 | 0.560 | 5.393 | 6.123 | **0.447** | −0.076 |
| Longitudinal | canonical | 15 | 0.625 | 0.635 | 4.462 | 5.726 | **0.525** | **0.240** |
| **Longitudinal** ⭐ | canonical | 20 | **0.719** | **0.762** | **4.736** | 6.346 | 0.376 | 0.104 |
| Longitudinal | canonical | 25 | 0.438 | 0.421 | 5.470 | 7.707 | −0.016 | −0.075 |
| Longitudinal | canonical | 30 | 0.438 | 0.536 | 5.465 | 7.345 | 0.091 | −0.089 |
| Longitudinal | canonical | 40 | 0.656 | 0.595 | 6.200 | 8.983 | −0.142 | −0.203 |
| Longitudinal | canonical | 50 | 0.625 | 0.627 | 5.352 | 7.459 | 0.176 | −0.011 |
| Longitudinal | canonical | 100 | 0.625 | 0.714 | 5.490 | 7.481 | 0.202 | −0.011 |
| TinyLlama (ts0+long) | canonical, TinyLlama-1.1B | 10 | 0.344 | 0.397 | 4.794 | 6.294 | 0.548 | 0.220 |
| **Baseline (ts0)** — leak-free, seed 0 | canonical + `load_best_model_at_end=False` | 20 | 0.531 | 0.377 | 6.392 | 7.632 | −0.146 | −0.400 |
| **Longitudinal** — leak-free, seed 0 | canonical + `--no-geno-conditioning --rollout` + `load_best_model_at_end=False` | 20 | 0.469 | 0.464 | 7.036 | 8.006 | 0.049 | −0.669 |
| **Baseline (ts0)** — leak-free, 5-seed pooled | same, `data_seed` 0–4 | 20 | — | **0.536** [0.444, 0.627] | — | — | −0.027 ± 0.071 | −0.217 ± 0.114 |
| **Longitudinal** — leak-free, 5-seed pooled | same, `data_seed` 0–4 | 20 | — | **0.369** [0.292, 0.451]¹ | — | — | −0.011 ± 0.107 | −0.269 ± 0.236 |

¹ The pooled-CI "significantly below chance" reading was withdrawn 2026-09-14: the seed spread (0.11–0.63) gives an honest interval of roughly [0.2, 0.6], and two penalties on the longitudinal arm were found and fixed (F24 token scale, F20 tube crops). See FINDINGS F22 note and F24.

> **The 5-seed pooled rows are the citable numbers; the seed-0-only rows above them are
> kept for provenance.** Every non-leak-free row in this table was produced with the
> genotype label injected into ts1–3 (F10) and the checkpoint selected on the held-out
> subject (F2). The leak-free protocol (20 epochs, same LoRA/multitask config,
> `load_best_model_at_end=False`) was re-run 2026-09-09 at `data_seed=0` and extended to
> 5 seeds on 2026-09-13 (F22; `aggregate_seeds.py`), scored with the deduplicated
> evaluator (F14; n=32 per slot). Against the n=32 no-signal null (sd 0.105, 95%
> [0.297, 0.703]), the baseline pooled AUROC sits **inside the null** and the longitudinal
> pooled AUROC sits **significantly below it** (95% CI [0.292, 0.451] excludes 0.5); the
> old 0.762 headline was z = +2.50 against the same null. TBR R² is negative in all 10
> seed×arm runs. See the [leak-free re-run](#leak-free-re-run-2026-09-09) and
> [seed sweep](#seed-sweep-5-seedsarm-2026-09-13---f22) sections below.

The 10-epoch baseline is the legacy lower-bound run (`mouse_vlm_baseline_loso`, old
r=16 / mtwt=1 config) and is **not** part of the controlled sweep — the sweep started
at 15 epochs. It is kept for provenance, not for direct comparison against the
canonical-config rows.

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

> **Superseded (2026-09-09).** Every conclusion below was drawn from runs with the genotype
> label injected into the longitudinal tokens (F10) and checkpoints selected on the held-out
> subject (F2). The leak-free re-run reverses takeaway 1 and removes the basis for 2–4. Kept
> as the historical record of what was believed and why.

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

**Recommended configs (withdrawn):** the 20ep/15ep recommendations rested on the leaky
protocol. See the leak-free re-run section.

---

## Completed experiments

### `mouse_vlm_baseline_loso` — VLM baseline (ts0 only)
- **YAML:** `viz_emb_params_mouse.yml`
- **Key settings:** img_tokens=1, no predicted_emb_dir, r=16 full LoRA, multitask_wt=1, 10 epochs
- **Purpose:** legacy lower bound — what can the VLM do from a single scan? Uses the
  old r=16 / mtwt=1 config, so it is **not** directly comparable to the canonical-config
  baseline sweep (base_ep15–40) in the epoch-sweep section above.
- **Results:**
  - Genotype acc: **0.219** / AUROC: **0.163** *(below chance — head ranks genotype anti-correlated)*
  - TBR reg MAE (overall): 6.241, r=−0.054, R²=−0.357
  - TBR reg MAE (Δ3wk): 6.806, r=−0.090, R²=−0.088, n=64
  - TBR reg MAE (Δ6wk): 7.568, r=−0.243, R²=−0.957, n=30
  - TBR reg MAE (Δ8wk): 4.340, r=0.069, R²=−4.421, n=40

### `mouse_vlm_loso` — VLM longitudinal, 10-epoch (exp3 config)
- **YAML:** `viz_emb_params_mouse_exp3_combined.yml`
- **Key settings:** img_tokens=4 (ts0 real + ts1/ts2/ts3 MLP-predicted), r=4 q/v LoRA only,
  multitask_wt=5, z-scored TBR targets, 10 epochs
- **Purpose:** combines lessons from exp1–3. Superseded as the headline result by
  `mouse_vlm_ep20` (same config, 20 epochs) — see below; the 20-epoch config is the
  canonical default (`viz_emb_params_mouse.yml`).
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

## Epoch and architecture ablations (post denorm-fix)

These all use the `mouse_vlm_loso` config as the baseline and change **one variable only**.
(An earlier round had invalid TBR MAE from a denorm bug; all runs below are post-fix.)

### `mouse_vlm_ep20` — epoch ablation: 20 epochs ✓ (CANONICAL / headline result)
- **YAML:** `viz_emb_params_mouse_ep20.yml` — identical to the canonical default `viz_emb_params_mouse.yml`
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

---

## Leak-free re-run (2026-09-09)

**Question.** Does the longitudinal input still help once it is not carrying the genotype
label, and once checkpoints are no longer selected on the held-out subject?

**Design.** Paired baseline-vs-longitudinal at a single pre-registered epoch count (20),
not a sweep — F1 showed the old headline was the maximum over 10 configurations. One knob
changed relative to the archived runs beyond the fixes themselves: nothing. `pool_at` was
left at `answer_eos` so F3 is not confounded with F10; that is the natural next ablation.

| | YAML | predicted_emb_dir | GPU | wall |
|---|---|---|---|---|
| Baseline | `viz_emb_params_mouse_nogeno_base.yml` | none (`img_tokens=1`) | L40S | 4 h 48 m |
| Longitudinal | `viz_emb_params_mouse_nogeno_long.yml` | `longitudinal_nogeno/` (rollout, no genotype bit) | L40S | 4 h 49 m |

Longitudinal MLP re-run first (`--no-geno-conditioning --rollout`, seed 0): the genotype
probe on ts1 dropped **0.869 → 0.369** (≈ the 0.353 of the observed Week 12 scan) while
T4a cosine went 0.9807 → **0.9816** — the genotype bit was contributing nothing to
prediction quality. Rollout fills all 32 subjects (ts2/ts3 at n=32, vs 15 and 9).

**Result** (all four scored with the same current evaluator):

| | Geno AUC | Geno acc | TBR MAE | TBR r | TBR R² | Δ3wk r |
|---|---|---|---|---|---|---|
| OLD baseline (leaky) | 0.583 | 0.500 | 4.810 | 0.356 | 0.107 | 0.303 |
| OLD longitudinal (leaky) | **0.762** | 0.719 | 4.736 | 0.342 | 0.105 | 0.383 |
| **NEW baseline** | 0.377 | 0.531 | 6.392 | −0.141 | −0.400 | −0.146 |
| **NEW longitudinal** | 0.464 | 0.469 | 7.036 | −0.030 | −0.669 | 0.049 |

**Reading.**
- **The genotype claim does not survive.** Both leak-free arms are inside the n=32
  no-signal interval. The 0.762 headline was the sum of label injection (F10) and
  selection on the test subject (F2); with both removed it is 0.464.
- **"Longitudinal tokens carry the genotype signal" does not survive.** The gap is
  +0.087, inside noise, and the tokens the probe showed to be ≈week-centroids with no
  per-mouse content cannot plausibly carry it.
- **The TBR claim does not survive either, and got worse.** R² is negative in both arms
  — the head is worse than predicting the training-fold mean. The drop is larger than
  F2 alone predicts for AUROC, which is expected: MSE-based selection on one subject's
  three records is an even noisier selector than a rank metric, so F2 inflated TBR more.
  Rolled-out ts2/ts3 also no longer descend from the real Week 15/18 scans (F13), which
  may have carried genuine TBR-trend information that one-step predictions passed through.

**Caveats.** One seed per arm; F1 asks for ≥5. 20 epochs was chosen under the leaky
protocol and may not be right for the fixed one. `pool_at=answer_eos` means genotype
*accuracy* is still uncalibrated (F3) — read the AUROC column only. Session confounding
(F5) is untested but is now moot for genotype, since there is no genotype effect left to
explain.

---

## Seed sweep, 5 seeds/arm (2026-09-13) — F22

**Question.** F1 asked for ≥5 seeds before citing the leak-free re-run above; it was one
seed each. Does the negative result hold up, and is the point estimate representative?

**Design.** Same config as the leak-free re-run (`viz_emb_params_mouse_nogeno_{base,long}
[_seed{1..4}].yml`, 20 epochs, `load_best_model_at_end=False`), only `data_seed` varied
(0–4; seed 0 is the run above). 8 new 32-fold LOSO runs, tmux on GPUs 1 and 6 in
parallel, ~4.85 h each. Aggregated with `scripts/aggregate_seeds.py --rescore`.

| Arm | seed0 | seed1 | seed2 | seed3 | seed4 | mean ± SD | pooled | 95% CI | perm p |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (ts0) | 0.377 | 0.595 | 0.687 | 0.313 | 0.548 | 0.504 ± 0.155 | 0.536 | [0.444, 0.627] | 0.250 |
| Longitudinal (4-token) | 0.464 | 0.111 | 0.532 | 0.627 | 0.206 | 0.388 ± 0.220 | 0.369 | [0.292, 0.451] | **0.997** |

TBR R² across all 5 seeds: baseline −0.400 / −0.152 / −0.202 / −0.100 / −0.232 (mean
−0.217 ± 0.114); longitudinal −0.669 / −0.047 / −0.165 / −0.251 / −0.215
(mean −0.269 ± 0.236) — **negative in all 10 runs.**

**Reading.** The single-seed caveat is closed: baseline sits at chance and TBR is
robustly negative everywhere, not a fluke of seed 0. The longitudinal arm's pooled AUROC
(0.369) initially read as "significantly below chance" — **withdrawn 2026-09-14.** The
pooled CI rank-pools five seeds that do not agree (0.11 / 0.21 / 0.46 / 0.53 / 0.63); the
seed spread gives ≈[0.2, 0.6], and the refit null for pooled LOSO AUROC is ~0.42 (F2's
simulation), so nothing here is distinguishable from baseline or chance. Two concrete
penalties on this arm were then found and fixed — forecast tokens exported at 2.7× the
observed norm (F24) and five tube crops in the training data (F20) — and the arm is
re-running with and without F3's `question_eos` (see the next section). The seed SD of
≈0.2 is itself the result: at n=32 the VLM's genotype output is dominated by initialisation.
---

## Corrected-data re-runs (2026-09-13/14) — F20, F24

**What changed in the data.** (F20) Two Week-15 sessions had the bed's fluid line and a
phantom segmented as mice; the real mice were shifted or discarded. Re-cropped with an
explicit quadrant override; all four encoders, TBR, the MLP, and the VQA JSONs re-derived.
Exactly five (mouse, week) entries changed. (F24) The MLP is trained with cosine loss, so its
exported forecasts had norm ~37 against ~14 for observed embeddings; the VLM's shared
projection therefore fed the LLM three tokens at 2.7× the magnitude of the one carrying
information. Exports are now rescaled to the training-fold norm (13.97).

**Runs in flight** (seed 0, 20 ep, `load_best_model_at_end=False`, ~5 h each):

| run | tokens | `pool_at` | question it answers |
|---|---|---|---|
| `mouse_vlm_nogeno_base_ep20_f20` | none (ts0 only) | answer_eos | baseline on corrected crops |
| `mouse_vlm_nogeno_long_ep20_f22` | `longitudinal_nogeno_f22` (scaled, crop-fixed) | answer_eos | does removing the scale penalty change the longitudinal arm? |
| `mouse_vlm_nogeno_long_ep20_f22_qeos` | same | **question_eos** | does matched pooling (F3) reduce seed instability? |

Expectation, stated in advance: the scale fix should remove the longitudinal arm's
*penalty* relative to baseline but not produce a positive result, because the tokens are
functions of the Week 12 scan and carry no per-mouse content it lacks. If `f22_qeos` shows
a materially tighter or higher result than `f22`, that is the F3 finding.

