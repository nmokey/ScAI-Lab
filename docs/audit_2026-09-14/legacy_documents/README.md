# Longitudinal Vision Encoder for Mouse Atherosclerosis Imaging

A **Medical Vision-Language Model (VLM)** pipeline for longitudinal atherosclerosis monitoring in mice — benchmarking pretrained 3D/2D vision encoders and training a trajectory-prediction VLM on multi-modal PET/CT data.

**2026-09-14 implementation update:** checkpoint reload and question-only supervision for both heads are repaired and tested on the actual 4-bit backbone. The runners now default to the repaired configuration and fresh output paths. See the [repair report](docs/audit_2026-09-14/CHECKPOINT_HEAD_REPAIRS.md) for validation and recovered adapters, and the [master plan](docs/MASTER_PLAN.md) for remaining gates, including nested forecasts. Historical result tables below are not corrected-run results.

---

## Motivation

Atherosclerosis (arterial plaque build-up) is the leading cause of cardiovascular disease. Tracking its progression non-invasively with PET/CT imaging is clinically valuable, but existing vision encoders are trained on human 2D data and have never been evaluated on longitudinal 3D mouse preclinical studies.

**Research questions:**
1. Can a generic, human-centric encoder understand mouse disease progression?
2. Can a VLM predict future aortic TBR trajectory and genotype from a single baseline scan embedding?
3. Does feeding longitudinal encoder predictions as additional image tokens improve VLM performance?

## Key Takeaways

> [!NOTE]
> **Audit status (2026-09-13).** The results in this file are post-audit. Two defects
> inflated every VLM number reported before 2026-09-09: the genotype label leaked into the
> longitudinal tokens (F10; probe AUROC 0.869 vs 0.353 from the observed scan), and
> checkpoints were selected on the held-out subject (F2; +0.14–0.20 AUROC on noise). The
> leak-free protocol, aggregated over 5 seeds/arm (F22), gives baseline genotype AUROC
> 0.50 ± 0.16 (chance) and longitudinal 0.39 ± 0.22 (seed SD; not distinguishable from baseline or chance — two since-fixed penalties on this arm, F20 and F24, corrected runs in flight); TBR
> R² is negative in all 10 seed×arm runs. Separately, all 10 mouse-groups in the study are
> genotype-pure (F5): under leave-one-mouse-group-out, Week-12 genotype is chance (0.50)
> and rises to 0.82 across all weeks. Every finding is backed by a test in `tests/`;
> register and evidence in [FINDINGS.md](docs/FINDINGS.md).


1. **Partly — and the two halves of this question have different answers.** *Time* is trivially readable: all four encoders separate Week 12 from Week 20 scans near-perfectly (RAD-DINO 1.000, others 0.96–0.98), and they do so **for healthy WT mice just as well as for KO** (WT-only: RAD-DINO 1.000, 3D encoders 0.88–0.96). That separation is substantially **scanner drift** — voxels containing no mouse separate the two periods at AUC 0.76; air mean HU shifted 13 HU and its variance collapsed — with a modest growth component (bone volume +10%). Not disease. *Disease* is readable by exactly one encoder: under leave-one-mouse-group-out (all 10 mouse-groups are genotype-pure, so any looser evaluation measures group recognition), **RAD-DINO's** WT-vs-KO AUROC is chance at Week 12 (0.489), 0.76 at Week 20, and 0.83 pooled (p ≈ 0.008). That contrast is protected from date confounding — both genotypes were scanned in the same periods — and it emerges as the disease develops. The three 3D encoders fall to chance under the same control. No encoder does subject retrieval (Recall@1 < 5%). Caveat: 10 groups is a thin null.
2. **No — not from a single Week 12 scan, on this data.** The leak-free VLM (label injection and test-subject checkpoint selection removed), pooled over 5 seeds (F22), reaches genotype AUROC **0.536** (95% CI [0.44, 0.63], permutation p = 0.25 — chance) and TBR R² **< 0 in all 10 runs** — worse than predicting the mean. This is consistent with the encoder result: a Week 12 embedding carries no genotype signal that generalises across mouse groups. The earlier 0.762 was the genotype label leaking through the longitudinal MLP's conditioning vector (F10) plus checkpoints selected on the held-out subject (F2). Not excluded: a nonlinear model at larger n, a multi-observed-week input, or evaluation under mouse-group holdout, none of which has been run.
3. **No.** Pooled over 5 seeds, the longitudinal arm's genotype AUROC is 0.39 ± 0.22 (per seed: 0.11 / 0.21 / 0.46 / 0.53 / 0.63) against baseline's 0.50 ± 0.16 — not distinguishable from baseline or from chance once the seed spread is honoured (an earlier pooled CI of [0.29, 0.45] overstated precision by treating the seeds as agreeing). Two things were found to be *penalising* the longitudinal arm, and both are now fixed: the forecast tokens were exported at 2.7× the observed embedding's norm because the MLP's cosine loss never constrained magnitude (F24), and five Week-15 crops in the training data were bed tubes rather than mice (F20). Corrected runs are in flight (see `docs/experiments.md`). Even so, the leak-free tokens are functions of the Week 12 scan and carry no per-mouse content it lacks — there is nothing for them to add.

All numbers above are from `docs/experiments.md` §§ "Leak-free re-run" and "Seed sweep" and `docs/FINDINGS.md` F5/F22; every pre-2026-09-09 VLM number in this repository is invalid. See the audit register for the full account.

---

## Dataset

**Source study:** Tamboline M et al. (2025). *Preclinical evaluation of high-resolution CT, 18F-FDG, and 18F-NaF PET imaging for longitudinal monitoring of atherosclerosis.* European Journal of Nuclear Medicine and Molecular Imaging 52:4256–4267.

80 unique male mice (Apoe−/− KO on high-fat diet vs. C57BL/6 WT controls), scanned longitudinally at **weeks 12, 15, 18, and 20** across two separate cohorts:

| Cohort | Tracer | Signal | n (KO + WT) |
|--------|--------|--------|-------------|
| NaF | 18F-NaF | Vascular calcification | 20 + 20 |
| FDG | 18F-FDG | Inflammation | 20 + 20 |

Each session acquires Hi-Res CT (~2814 KB/slice) and PET for 1–4 mice simultaneously. **229 CT-Hi volumes** used for encoder evaluation after filtering Lo-Res CT sessions.

Full scan inventory, per-scan mouse mappings, data heuristics, and longitudinal attendance tables: [docs/DATA_MANIFEST.md](docs/DATA_MANIFEST.md).

---

## Methods

### Pipeline Overview

```
Raw DICOM  →  NIfTI Conversion  →  Encoder Embeddings  →  Evaluation / VLM Training
                                                      ↘  Longitudinal Encoder
```

1. **DICOM → NIfTI** (`scripts/build_nifti_dataset.py`): file-size heuristic modality classification, per-modality re-scan consolidation, quadrant-based per-mouse segmentation and cropping to RAS-oriented volumes.
2. **Encoder Embeddings**: four pretrained encoders evaluated zero-shot, each producing a standardised `.npz` (N × D embeddings + metadata).
3. **Evaluation** (`scripts/evaluate_embeddings.py`): T1 unsupervised, T2 linear-probe (LOSO CV), T3 longitudinal/retrieval tasks.
4. **Longitudinal Encoder** (`scripts/train_longitudinal.py`): MLP trained under LOSO CV to predict T_{k+1} embeddings from T_k; predicted future embeddings saved for VLM input.
5. **VLM Training** (`vlm/`): frozen RAD-DINO features (up to 4 tokens: observed Week 12 + longitudinal-predicted Week 15/18/20) + linear projection + LoRA-finetuned LLaMA-3.1-8B. Two multitask heads trained jointly on LLM hidden state: TBR regression (MSE) and genotype classification (BCE).

### Encoders Evaluated

| Encoder | Architecture | Trained on | Embedding dim |
|---------|-------------|------------|---------------|
| COLIPRI | 3D ViT | Human chest CT + reports | 768 |
| Merlin | 3D ViT | Human abdominal CT + EHR | 2048 |
| RAD-DINO | 2D ViT-B/14 (32-slice mean-pool) | 882k chest X-rays (DINOv2) | 768 |
| M3D-CLIP | 3D ViT | 120k multi-modal medical images (CLIP) | 768 |

---

## Results

> [!NOTE]
> **Audit status (2026-09-13).** The results in this file are post-audit. Two defects
> inflated every VLM number reported before 2026-09-09: the genotype label leaked into the
> longitudinal tokens (F10; probe AUROC 0.869 vs 0.353 from the observed scan), and
> checkpoints were selected on the held-out subject (F2; +0.14–0.20 AUROC on noise). The
> leak-free protocol, aggregated over 5 seeds/arm (F22), gives baseline genotype AUROC
> 0.50 ± 0.16 (chance) and longitudinal 0.39 ± 0.22 (seed SD; not distinguishable from baseline or chance — two since-fixed penalties on this arm, F20 and F24, corrected runs in flight); TBR
> R² is negative in all 10 seed×arm runs. Separately, all 10 mouse-groups in the study are
> genotype-pure (F5): under leave-one-mouse-group-out, Week-12 genotype is chance (0.50)
> and rises to 0.82 across all weeks. Every finding is backed by a test in `tests/`;
> register and evidence in [FINDINGS.md](docs/FINDINGS.md).


### Encoder Evaluation

Zero-shot evaluation on 229 CT-Hi volumes, 78 mice. Full table and per-encoder analysis: [docs/results.md](docs/results.md).

| Metric | COLIPRI | Merlin | RAD-DINO | M3D |
|--------|---------|--------|----------|-----|
| T2b AUC (early vs. late), by subject | 0.987 | 0.958 | **1.000** | 0.993 |
| T2b AUC (early vs. late), by mouse-group | 0.983 | 0.958 | **1.000** | 0.983 |
| T2c AUC (WT vs. KO), by subject² | 0.666 | 0.636 | **0.920** | 0.736 |
| T2c AUC (WT vs. KO), by mouse-group³ | 0.473 | 0.458 | **0.823** (W12 only: 0.498) | 0.541 |
| T3b Recall@1 (subject retrieval) | 0.017 | **0.044** | 0.013 | 0.022 |
| T1b ARI (unsupervised week clusters)⁴ | 0.033 | 0.011 | −0.010 | **0.114** |

**Key findings:**
- **RAD-DINO** dominates all supervised tasks — perfect early/late separation (T2b AUC = 1.000, unchanged under mouse-group holdout) and the only encoder with group-generalising genotype signal, which is absent at Week 12 and rises to 0.82 across all weeks. Selected as the frozen backbone for VLM training.

All rows re-evaluated 2026-09-12 with the audited evaluator (`embeddings/eval_audit_2026-09-12/`).
² Per-fold feature standardisation; previously 0.282 / 0.491 / 0.869 / 0.567 with the unscaled probe.
³ Leave-one-mouse-group-out — the clean genotype control. Only RAD-DINO survives it (p = 0.008); the 3D encoders are at chance (p ≥ 0.2). Their by-subject genotype signal was group recognition. (F18: 10 groups split 3 WT/7 KO gives exactly C(10,3)=120 distinct arrangements, so these are exact permutation p-values, quantised in steps of 1/120 ≈ 0.0083 — not Monte Carlo approximations.)
⁴ Unsupervised; untouched by any audit change.
- **M3D** leads on unsupervised geometry (T1b ARI = 0.114), but weak genotype signal (T2c ≈ chance).
- **Subject retrieval is universally poor** (T3b Recall@1 < 5% across all encoders) — a custom encoder with longitudinal contrastive loss is needed to close this gap.

### Longitudinal Encoder (LOSO CV, RAD-DINO embeddings, 32 NaF subjects)

MLP (768-d embedding + 7-d conditioning → 768-d output; two hidden layers with LayerNorm+GELU) trained to predict T_{k+1} from T_k across 56 consecutive pairs. Retrained on the 32 NaF subjects only to match the VLM evaluation population.

Leak-free configuration (defaults as of 2026-09-14: no genotype conditioning, rollout from Week 12, export rescaled to the observed norm; `longitudinal_nogeno_f22/`). The MLP no longer receives the subject's genotype, and exported ts2/ts3 are rolled forward from Week 12 rather than predicted from observed later scans (FINDINGS.md F10, F13).

| Metric | Value | Notes |
|--------|-------|-------|
| T4a cosine sim | 0.982 | High directional accuracy; reflects tight embedding cluster geometry. Unchanged from the leaky config (0.981) — the genotype bit was contributing nothing to prediction quality |
| T4b Recall@1 | 0.018 | Chance-level subject retrieval — predicts week cluster, not individual identity |
| T4b MRR | 0.196 | Correct subject ranks ~5th out of 32 on average |
| T4c improvement rate | 0.696 | Beats returning T_k unchanged 70% of the time |

Genotype is no longer recoverable from the predicted tokens: a LOSO probe on `ts1` gives AUROC 0.369 (was 0.869 with the genotype bit), essentially the 0.353 of the observed Week 12 scan it descends from.

### VLM — genotype & TBR trajectory prediction (multitask heads, LOSO CV, 32 NaF subjects)

All VLM metrics come from the multitask heads on the LLM's last hidden state — genotype from a sigmoid classification head, TBR from a 4-slot regression head. (Text-generated TBR is unreliable: the LLM produced parseable output in <2% of held-out records, so the regression head is the sole reported TBR metric.) "Longitudinal" means feeding the MLP-predicted Week 15/18/20 embeddings (ts1/ts2/ts3) as extra image tokens alongside the observed Week 12 embedding (ts0).

Genotype is a binary WT-vs-KO task (chance: acc 0.5, AUROC 0.5); both accuracy and AUROC come from the sigmoid classification head.

| Model (20 ep, LOSO, 5-seed pooled)² | Geno AUROC | 95% CI | perm p | TBR r | TBR R² |
|---|---|---|---|---|---|
| **Baseline (ts0 only) — leak-free** | 0.536 | [0.444, 0.627] | 0.250 | −0.027 ± 0.071³ | −0.217 ± 0.114³ |
| **Longitudinal 4-token — leak-free** | 0.369 | [0.292, 0.451] | **0.997⁴** | −0.011 ± 0.107³ | −0.269 ± 0.236³ |
| *Baseline (ts0 only) — leaky, invalid* | *0.583* | | | *0.356* | *0.107* |
| *Longitudinal 4-token — leaky, invalid* | *0.762* | | | *0.342* | *0.105* |

² 5 seeds/arm (`data_seed` 0–4), aggregated with `scripts/aggregate_seeds.py` (F22); genotype AUROC is rank-normalised within seed then subject-bootstrapped across all 5×32 rows. The n=32 no-signal AUROC interval (single seed) is [0.297, 0.703]. Genotype accuracy uses a fixed logit > 0 threshold that does not transfer from training to inference (F3); read AUROC.
³ mean ± SD over the 5 seeds (not pooled).
⁴ One-sided p = P(null AUROC ≥ observed). Note the pooled statistic treats five seeds as one sample; the seed-to-seed SD (0.16–0.22) is the more honest uncertainty, and the refit null for pooled LOSO AUROC sits near 0.42, not 0.5 (F2's simulation). Do not read the longitudinal p as evidence of below-chance signal.

**Key findings:**
- **No positive genotype signal survives, and it's no longer a one-seed result.** 5 seeds/arm (F22) close the single-seed caveat: baseline's pooled AUROC (0.536) sits at chance, matching the null. The previous 0.762 decomposed into two defects: the longitudinal MLP took the held-out subject's own genotype as an input, and a LOSO probe recovered it from the predicted `ts1` token at AUROC 0.869 vs 0.353 from the observed scan (F10); and checkpoints were selected by loss on the held-out subject, +0.14 to +0.20 AUROC on noise (F2).
- **The longitudinal arm's low pooled AUROC (0.369) was an artifact of pooling, plus two real penalties, not a below-chance signal.** Per-seed values span 0.11–0.63; the honest interval is ≈[0.2, 0.6]. The arm was also fed forecast tokens at 2.7× the observed norm (F24) and five tube crops (F20), both fixed; corrected runs are in flight. What the earlier text called unexplained — it echoes an earlier unexplained below-chance result (ts0-baseline at 10 epochs, leaky protocol, AUROC 0.163) but now shows up in the longitudinal arm, survives both leak fixes, and is consistent across 5 seeds. `pool_at: question_eos` (untested) is the leading candidate mechanism (F3). See FINDINGS.md F22.
- **TBR regression is negative in every one of 10 seed×arm runs.** R² < 0 means the head is worse than the training-fold mean, robustly — not a single-seed artifact. The old positive TBR numbers were inflated more than genotype by F2, because MSE-based selection on one subject's three records is an especially noisy selector.
- **This agrees with the encoder evidence.** A linear probe on the Week 12 RAD-DINO embedding is at chance for genotype under mouse-group holdout; the VLM's input contains nothing for a positive signal to find. That makes the negative result credible rather than a training failure — though the VLM has still not been evaluated by mouse-group rather than by subject.
- *The earlier TinyLlama and epoch-sweep observations were made under the leaky protocol and are not reported here; see `docs/experiments.md` for the archived rows.*
- Δ3wk is the strongest TBR slot; Δ6wk/Δ8wk are weaker due to smaller sample sizes. See [docs/experiments.md](docs/experiments.md) for the full run log and [docs/results.md](docs/results.md) for the per-slot breakdown.

---

## Setup

### Environment

```bash
conda env create -f environment.yml
conda activate vlm_env
```

Or with pip: `pip install -r requirements.txt`

### Configuration

All server-specific paths live in `config.yaml` (gitignored). Copy the template and fill in your paths:

```bash
cp config.yaml.example config.yaml
# Set paths.data_root  → /path/to/Dicom Data/
#     paths.output_dir → where NIfTI files should be written
```

### Reproducing the Pipeline

```bash
# 1. Convert DICOM to NIfTI (all sessions)
python scripts/build_nifti_dataset.py

# 2. Extract embeddings
python scripts/get_raddino_embeddings.py    # or colipri / merlin / m3d

# 3. Evaluate encoders
python scripts/evaluate_embeddings.py \
    --embeddings {output_dir}/embeddings/raddino/raddino_embeddings.npz \
    --output-dir {output_dir}/embeddings/raddino/

# 4. Train longitudinal encoder and export predicted embeddings
python scripts/train_longitudinal.py
# → writes longitudinal/predicted_embeddings/NaF_*_ts{1,2,3}.npy

# 5. Build VQA dataset (run once; outputs already committed to data dir)
python scripts/create_mouse_traj_dataset.py

# 6. Train VLM with LOSO CV (longitudinal 4-token input + multitask heads)
#    The default yaml is the canonical config — this reproduces the headline result.
cd vlm && CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_loso.py
# → {output_dir}/embeddings/vlm/runs/mouse_vlm_loso/loso_results.json
```

---

## Repository Structure

```
scripts/                  DICOM conversion, embedding extraction, evaluation
  train_longitudinal.py   Longitudinal MLP encoder: T_k → T_{k+1} prediction (LOSO CV)
  create_mouse_traj_dataset.py  Builds VQA JSON + per-scan .safetensors embeddings
vlm/
  data/                   Dataset and evaluation code
  model/                  VisionLanguageModel, multitask heads, trainer
  run/                    Single-run and LOSO CV entry points
  yaml/                   Training hyperparameter configs
docs/
  DATA_MANIFEST.md        Full scan inventory, mouse mappings, data heuristics
                          (mouse_manifest.csv itself is written to `paths.output_dir`,
                          not the repo -- it's machine-specific, generated per run)
  results.md              Encoder + VLM evaluation — full tables and analysis
  experiments.md          VLM experiment log (all LOSO runs, ablations, sweeps)
  design_decisions.md     Non-obvious design choices and known limitations
  QA_REPORT.md            Dataset QA verification report
manifest.csv              Session-level DICOM inventory (147 rows, tracked in git)
config.yaml.example       Path and parameter template
environment.yml           Conda environment
```

---

## Citation

```bibtex
@article{tamboline2025preclinical,
  title={Preclinical evaluation of high-resolution CT, 18F-FDG, and 18F-NaF PET imaging
         for longitudinal monitoring of atherosclerosis},
  author={Tamboline, M and others},
  journal={European Journal of Nuclear Medicine and Molecular Imaging},
  volume={52},
  pages={4256--4267},
  year={2025}
}
```
