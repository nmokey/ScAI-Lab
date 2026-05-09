# Longitudinal Vision Encoder for Mouse Atherosclerosis Imaging

A **Medical Vision-Language Model (VLM)** pipeline for longitudinal atherosclerosis monitoring in mice — benchmarking pretrained 3D/2D vision encoders and training a trajectory-prediction VLM on multi-modal PET/CT data.

---

## Motivation

Atherosclerosis (arterial plaque build-up) is the leading cause of cardiovascular disease. Tracking its progression non-invasively with PET/CT imaging is clinically valuable, but existing vision encoders are trained on human 2D data and have never been evaluated on longitudinal 3D mouse preclinical studies.

**Research question:** Can a generic, human-centric encoder understand mouse disease, or is a custom model required? And can a VLM answer clinical questions (*"What will the TBR be at Week 20?"*) directly from scan embeddings?

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
```

1. **DICOM → NIfTI** (`scripts/build_nifti_dataset.py`): file-size heuristic modality classification, per-modality re-scan consolidation, quadrant-based per-mouse segmentation and cropping to RAS-oriented volumes.
2. **Encoder Embeddings**: four pretrained encoders evaluated zero-shot, each producing a standardised `.npz` (N × D embeddings + metadata).
3. **Evaluation** (`scripts/evaluate_embeddings.py`): T1 unsupervised, T2 linear-probe (LOSO CV), T3 longitudinal/retrieval tasks.
4. **VLM Training** (`vlm/`): frozen RAD-DINO features + linear projection + LoRA-finetuned LLaMA-3.1-8B for trajectory TBR prediction.

### Encoders Evaluated

| Encoder | Architecture | Trained on | Embedding dim |
|---------|-------------|------------|---------------|
| COLIPRI | 3D ViT | Human chest CT + reports | 768 |
| Merlin | 3D ViT | Human abdominal CT + EHR | 2048 |
| RAD-DINO | 2D ViT-B/14 (32-slice mean-pool) | 882k chest X-rays (DINOv2) | 768 |
| M3D-CLIP | 3D ViT | 120k multi-modal medical images (CLIP) | 768 |

---

## Results

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

# 3. Evaluate
python scripts/evaluate_embeddings.py \
    --embeddings {output_dir}/embeddings/raddino/raddino_embeddings.npz \
    --output-dir {output_dir}/embeddings/raddino/

# 4. Train VLM (NaF cohort, TBR trajectory prediction)
cd vlm && CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm.py
```

---

## Repository Structure

```
scripts/                  DICOM conversion, embedding extraction, evaluation
vlm/                      VLM training pipeline (dataset builder, model, trainer)
docs/
  DATA_MANIFEST.md        Full scan inventory, mouse mappings, data heuristics
  results.md              Zero-shot encoder evaluation — full tables and analysis
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
