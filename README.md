# Longitudinal Vision Encoder for Mouse Atherosclerosis Imaging

A **Medical Vision-Language Model (VLM)** research pipeline for longitudinal mouse PET/CT — benchmarking pretrained 3D/2D vision encoders and training a trajectory-prediction VLM on multi-modal PET/CT data.

**Current status — 2026-09-21:** known critical implementation defects were repaired, and the declared computations passed targeted verification. All prescribed repair and subsequent development comparisons are complete. This is evidence of reproducibility, not proof that the pipeline is free of bugs or methodological limitations. Start with [current status and outstanding controls](docs/STATUS.md), [all post-audit results](docs/POST_AUDIT_RESULTS.md), and [methods and limitations](docs/design_decisions.md). Historical VLM numbers do not evaluate the corrected model.

---

## Motivation

This project evaluates existing human-pretrained vision encoders on longitudinal mouse PET/CT, and tests whether predicted future embedding tokens improve a VLM’s prediction from a baseline CT embedding. Its current numerical endpoint is a PET intensity proxy; disease-specific validity has not been established.

**Research questions:**
1. Can a generic, human-centric encoder understand mouse disease progression?
2. Can a VLM predict the existing PET intensity proxy and genotype from a single baseline scan embedding?
3. Does feeding longitudinal encoder predictions as additional image tokens improve VLM performance?

## Current findings

1. **The original repaired LOSO comparison does not show an overall longitudinal gain.** Across its three prescribed seeds, baseline subject MAE is 5.6112 and longitudinal is 5.6494 (difference +0.0381). W18 MAE is modestly lower with longitudinal tokens in all three seeds (6.4072 → 6.2274 on average, n=15); this is a secondary result. Later four-fold results use a different training split and are reported separately.
2. **The existing PET proxy is computationally reproducible.** All 111 measurements reproduce exactly from current crops, and both arms preserve the same subjects and target masks. Its aortic or histological meaning remains unvalidated.
3. **The nested forecast implementation passes exclusion and normalization checks.** Its actual baseline rollouts outperform persistence in embedding cosine but trail the training-week centroid at every horizon. High cosine alone is not evidence of personalized prediction.
4. **A limited encoder association survives.** RAD-DINO all-week genotype/diet AUROC is 0.82819 with exact full-refit group-label p=0.00833 on 229 scans, 78 mice and ten groups. This does not localize disease or establish Week-12 VLM performance. The NaF VLM subset has only three acquisition components and one WT component.
5. **Subsequent model development improves genotype ranking, with weak accuracy.** On the amended four-fold protocol, residual-input VLM mean AUROC is 0.5484. Adding a direct embedding classifier creates a hybrid with AUROC 0.6882; its direct branch alone retains nearly all that ranking. Accuracy is 0.5208 and 0.5000 respectively; genotype-only training reaches 0.5806 AUROC and 0.5208 accuracy. These are development results on repeatedly inspected folds, not evidence of 70% accuracy, improved language-model reasoning, or validated disease forecasting. [Complete comparison and score diagnosis](docs/audit_2026-09-14/genotype_vlm_variants/README.md).
6. **The residual encoder itself shows a small ranking benefit.** A fixed classifier on residual future embeddings scores AUROC 0.6774 versus baseline 0.6290, but accuracy falls from 21/32 to 19/32. Actual future-embedding fidelity is worse than the original forecaster and training-week centroid. Repeated-baseline, mean-future VLM, and matched baseline-input hybrid controls remain proposed and unrun.

The [current result ledger](docs/results.md) is the publication source. The [previous README](docs/audit_2026-09-14/legacy_documents/README.md) and historical experiment logs are retained for provenance. Technical descriptions below should be read with the [current methods and limitations](docs/design_decisions.md); previous tables or figures are not corrected-run evidence.

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
4. **Longitudinal Encoder** (`scripts/train_longitudinal.py`): MLP forecasts nested inside each VLM outer holdout, with inner cross-fitting for training tokens and baseline-only rollout for test tokens.
5. **VLM Training** (`vlm/`): frozen RAD-DINO features (up to 4 tokens: observed Week 12 + longitudinal-predicted Week 15/18/20) + linear projection + LoRA-finetuned LLaMA-3.1-8B. Two multitask heads trained jointly on LLM hidden state: TBR regression (MSE) and genotype classification (BCE).

### Encoders Evaluated

| Encoder | Architecture | Trained on | Embedding dim |
|---------|-------------|------------|---------------|
| COLIPRI | Primus 3D encoder + projection/attention pooling | Human chest CT + reports | 768 |
| Merlin | 3D inflated ResNet-152 | Human abdominal CT + EHR | 2048 |
| RAD-DINO | 2D ViT-B/14 (32-slice mean-pool) | 882k chest X-rays (DINOv2) | 768 |
| M3D-CLIP | 3D ViT | 120k multi-modal medical images (CLIP) | 768 |

---

## Results

[The current result ledger](docs/results.md) contains verified evidence and the status of the corrected three-seed comparison. Historical VLM scores, rank-pooled inference and stale Merlin rows have been removed from this publication summary. Their original tables remain in [the archived README](docs/audit_2026-09-14/legacy_documents/README.md).

The reported VLM quantities come from its numerical regression and genotype heads. They do not establish generated-text quality. The primary endpoint is subject-averaged PET-proxy MAE, with all three future horizons and a true training-fold mean comparator reported. Genotype AUROC is secondary. The scope and inferential limits are in [the current methods](docs/design_decisions.md).

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

### Reproducing the validated comparison

Use the [frozen protocol, configurations and commands](docs/experiments.md). The existing validated inputs and all 32 nested forecast exports are already available. Do not regenerate or overwrite frozen inputs while runs consume them.

For a new reproduction, verify the current crops and PET proxy, build structured fixed-horizon records with `create_mouse_traj_dataset.py`, then use `build_nested_forecasts.py` to export forecasts separately inside each VLM holdout. The old global exports from `train_longitudinal.py` alone do not satisfy the nested VLM protocol. Use a new versioned output directory when input or code fingerprints change.

```bash
# Run one of the six prescribed arm/seed configurations; repeat for each pair.
.venv-test/bin/python -u vlm/run/run_mouse_vlm_loso.py \
  --yaml docs/audit_2026-09-14/run_configs/base_seed0.yml \
  --output-dir /data1/Processed_NIfTI_Test/embeddings/vlm/runs/validated_base_seed0

# Verify all six complete runs and build the authoritative result package.
.venv-test/bin/python scripts/finalize_validated_results.py
```

Select an available GPU for training with `CUDA_VISIBLE_DEVICES`. The full environment settings, paired scoring command and independent validation commands are in [the experiment log](docs/experiments.md). The finalizer refuses incomplete or mismatched results; `--wait` waits for all six prescribed runs without changing them.

---

## Repository Structure

```
scripts/                  DICOM conversion, embedding extraction, evaluation
  train_longitudinal.py   Existing MLP training implementation
  build_nested_forecasts.py  Outer holdouts and inner cross-fitted baseline rollouts
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
  experiments.md          Fixed protocol, commands and links to archived runs
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
