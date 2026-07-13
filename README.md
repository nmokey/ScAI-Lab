# Longitudinal Vision Encoder for Mouse Atherosclerosis Imaging

A **Medical Vision-Language Model (VLM)** pipeline for longitudinal atherosclerosis monitoring in mice — benchmarking pretrained 3D/2D vision encoders and training a trajectory-prediction VLM on multi-modal PET/CT data.

---

## Motivation

Atherosclerosis (arterial plaque build-up) is the leading cause of cardiovascular disease. Tracking its progression non-invasively with PET/CT imaging is clinically valuable, but existing vision encoders are trained on human 2D data and have never been evaluated on longitudinal 3D mouse preclinical studies.

**Research questions:**
1. Can a generic, human-centric encoder understand mouse disease progression?
2. Can a VLM predict future aortic TBR trajectory and genotype from a single baseline scan embedding?
3. Does feeding longitudinal encoder predictions as additional image tokens improve VLM performance?

## Key Takeaways

1. **Yes — but only one encoder.** Of four zero-shot human-pretrained encoders, **RAD-DINO** (a 2D chest-X-ray ViT applied slice-wise) cleanly separates early vs. late disease (AUC = 1.000) and is the *only* encoder with real genotype signal (WT-vs-KO AUC = 0.869); the 3D CT encoders are near chance on genotype. Human-centric encoders *can* read mouse disease progression, but capability is model-specific, not guaranteed. No encoder does subject retrieval (Recall@1 < 5%).
2. **Yes for genotype, weakly for TBR.** From a single Week 12 scan, the best VLM predicts genotype at 0.719 accuracy / 0.762 AUROC and recovers a positive TBR trajectory signal (Δ3wk r ≈ 0.4). TBR is inherently noisy — the ground truth is a population-level programmatic proxy, not a per-mouse gold standard (see [design_decisions.md](docs/design_decisions.md)).
3. **Yes — this is the central result.** Feeding MLP-predicted future embeddings as extra image tokens is what carries the genotype signal: it roughly doubles accuracy (0.219 → 0.531 at matched epochs) and lifts AUROC from below-chance (0.163) to 0.560, and to 0.762 at 20 epochs. A single-scan baseline has no stable genotype signal to learn.

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

### Encoder Evaluation

Zero-shot evaluation on 229 CT-Hi volumes, 78 mice. Full table and per-encoder analysis: [docs/results.md](docs/results.md).

| Metric | COLIPRI | Merlin | RAD-DINO | M3D |
|--------|---------|--------|----------|-----|
| T2b AUC (early vs. late) | 0.683 | 0.852 | **1.000** | 0.963 |
| T2c AUC (WT vs. KO) | 0.282 | 0.491 | **0.869** | 0.567 |
| T3b Recall@1 (subject retrieval) | 0.018 | **0.044** | 0.013 | 0.022 |
| T1b ARI (unsupervised week clusters) | 0.033 | 0.011 | −0.010 | **0.114** |

**Key findings:**
- **RAD-DINO** dominates all supervised tasks — perfect early/late separation (T2b AUC = 1.000) and best genotype discrimination (T2c AUC = 0.869). Selected as the frozen backbone for VLM training.
- **M3D** leads on unsupervised geometry (T1b ARI = 0.114), but weak genotype signal (T2c ≈ chance).
- **Subject retrieval is universally poor** (T3b Recall@1 < 5% across all encoders) — a custom encoder with longitudinal contrastive loss is needed to close this gap.

### Longitudinal Encoder (LOSO CV, RAD-DINO embeddings, 32 NaF subjects)

MLP (768-d embedding + 7-d conditioning → 768-d output; two hidden layers with LayerNorm+GELU) trained to predict T_{k+1} from T_k across 56 consecutive pairs. Retrained on the 32 NaF subjects only to match the VLM evaluation population.

| Metric | Value | Notes |
|--------|-------|-------|
| T4a cosine sim | 0.981 | High directional accuracy; reflects tight embedding cluster geometry |
| T4b Recall@1 | 0.036 | Near-chance subject retrieval — predicts week cluster, not individual identity |
| T4b MRR | 0.185 | Correct subject ranks ~5th out of 32 on average |
| T4c improvement rate | 0.661 | Beats returning T_k unchanged 66% of the time |

### VLM — genotype & TBR trajectory prediction (multitask heads, LOSO CV, 32 NaF subjects)

All VLM metrics come from the multitask heads on the LLM's last hidden state — genotype from a sigmoid classification head, TBR from a 4-slot regression head. (Text-generated TBR is unreliable: the LLM produced parseable output in <2% of held-out records, so the regression head is the sole reported TBR metric.) "Longitudinal" means feeding the MLP-predicted Week 15/18/20 embeddings (ts1/ts2/ts3) as extra image tokens alongside the observed Week 12 embedding (ts0).

Genotype is a binary WT-vs-KO task (chance: acc 0.5, AUROC 0.5); both accuracy and AUROC come from the sigmoid classification head.

| Model | Geno acc | Geno AUROC | TBR MAE (overall) | TBR r (Δ3wk) | TBR R² (overall) |
|-------|----------|------------|-------------------|--------------|------------------|
| Baseline (ts0 only, 10 ep) | 0.219 | 0.163 | 6.241 | −0.090 | −0.357 |
| Longitudinal 4-token (10 ep) | 0.531 | 0.560 | 5.393 | **0.447** | −0.076 |
| **Longitudinal 4-token (20 ep)** | **0.719** | **0.762** | **4.736** | 0.376 | **0.104** |

**Key findings:**
- **Longitudinal tokens are what carry genotype signal.** At matched 10 epochs, adding the MLP-predicted future embeddings roughly doubles genotype accuracy (0.219 → 0.531) and lifts AUROC from 0.163 to 0.560. The ts0-only baseline sits *below chance* on AUROC (0.163) — its head ranks genotype anti-correlated, i.e. there is no stable single-scan genotype signal to learn. Longitudinal tokens also flip TBR Δ3wk correlation from negative to r = 0.447.
- **20 epochs is the sweet spot** — best genotype (acc 0.719, AUROC 0.762), best overall TBR MAE (4.736), and the only configuration with positive overall R² (+0.104). Training further (50/100 ep) overfits back to ~0.625 genotype on this 32-subject dataset.
- **The genotype "signal-loss gap" is mostly underfitting, not a pathway limit.** A longitudinal linear probe on the same MLP-predicted embeddings reaches AUROC 0.718 — an inflated ceiling, since the MLP conditioning vector includes genotype as an input feature. At 10 epochs the VLM (AUROC 0.560) trails that ceiling, but at 20 epochs it reaches 0.762, *matching/exceeding* the probe. So the projection + LLM pathway can recover the probe-level genotype signal given enough training; the apparent loss at 10ep is an underfitting artifact. (It should not be over-read, since condition B is itself inflated.)
- **Backbone size trades off between tasks.** Swapping LLaMA-3.1-8B for TinyLlama-1.1B improves TBR regression (overall R² 0.220 vs −0.076) but collapses genotype (acc 0.344, AUROC 0.397 — below chance) — the smaller model overfits less on the numeric task but lacks capacity for genotype discrimination.
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
  results.md              Encoder + VLM evaluation — full tables and analysis
  experiments.md          VLM experiment log (all LOSO runs, ablations, sweeps)
  design_decisions.md     Non-obvious design choices and known limitations
  QA_REPORT.md            Dataset QA verification report
manifest.csv              Session-level DICOM inventory (147 rows)
mouse_manifest.csv        Per-mouse NIfTI index (generated by build_nifti_dataset.py)
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
