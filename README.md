# ScAI Lab — Vision Encoder (Jan 2026 – present)

## Project Overview

Building a **Medical Vision-Language Model (VLM)** ("Medical LLaVA") — a chatbot-style system where researchers can ask longitudinal questions about a subject, such as *"How much weight will this mouse lose over the next 4 weeks?"* or *"Describe the progression of atherosclerosis."*

**Your role:** Build and validate the **Vision Encoder** — the neural network that converts 3D mouse scans into embeddings the Language Model can understand. A VLM is only as good as its eyes.

**Biological focus:** Tracking **Atherosclerosis** (hardening of arteries) in mice — specifically **vascular calcification** and **inflammation** over time.

- **Input:** Longitudinal 3D scans of mice
- **Output:** Embeddings that capture disease progression (calcification & inflammation)

**The core research question:** Can a generic, human-centric encoder (**COLIPRI**) understand mouse disease, or do we need to train a custom model from scratch on mouse data to beat it?

**The challenge:** Most current VLMs are trained on 2D natural images. Medical data is **3D** (volumetric CT/MRI), **multi-modal** (different scan types per subject), and **longitudinal** (scans over months to track treatment). The first priority is developing a reasonable vision encoder baseline — important for the vision-language project and for a future self-supervised paper.

---

## Data

### Subjects & Timeline

- **147 Mice** across 4 cohorts — each week is a **separate group** sacrificed at that timepoint (IDs do not overlap across weeks)
- **4 Timepoints:** Week 12, 15, 18, 20
  - Week 12 = Early Stage (healthy-ish); Week 20 = Late Stage (diseased)
  - Week 12: 42 subjects · Week 15: 47 · Week 18: 33 · Week 20: 25
  - Not all subjects have all modalities (see Scan Inventory below)
- **Folder structure:** `Week X` → `Tracer` → `SubjectID` (e.g., `m54253`)
  - Each subject folder contains a "bag of slices" (e.g., 691 separate `.dcm` files)

### Modalities

PET/CT data using two radiotracers:

| Modality | DICOM Tag | File Size | Radiotracer | What it Measures | Project Role |
|---|---|---|---|---|---|
| **CT (Hi-Res)** | `CT` | ~2814 KB | N/A | Anatomy / Calcification | **Baseline.** Use for RadDINO/COLIPRI validation. |
| **CT (Low-Res)** | `CT` | ~705 KB | N/A | Anatomy (low res) | Attenuation correction only. Ignore for now. |
| **PET (FDG)** | `PT` | ~115 KB | 18F-FDG | Glucose / Inflammation | **Novelty.** Target for custom MAE training. |
| **PET (NaF)** | `PT` | ~115 KB | 18F-NaF | Calcium / Bone | **Critical.** Measures calcification (atherosclerosis). |

### Formats

- **`/Dicom Data/`** — Raw DICOM format. Industry standard; use this for the vision encoder.
- **`/Amgen SUV Data/ (.xif)`** — Processed SUV maps in a proprietary format (likely Amira/PMOD/Siemens). Not readable with standard Python libraries. **Stick to DICOM.**

### Data Heuristics

1. **The `_1` Rule:** If both `m54223` and `m54223_1` exist, **use only `_1`** — it is the corrected, higher-quality scan.
2. **Modality ID:** Use **file size** as the primary heuristic to identify slice modality (per PI). See the Modalities table for target sizes; apply a ±5% tolerance window. DICOM tags (`Modality`, `SeriesInstanceUID`) may be used as supplementary validation.
3. **CT Fallback:** Not all subjects have Hi-Res CT. If no Hi-Res CT slices are found, fall back to Low-Res CT.
4. **Ignore:** Files ending in `.im3`, `.vol`, `.raw`.

### Scan Inventory

Each week is a **separate cohort** sacrificed at that timepoint — mouse IDs do not overlap across weeks. Determined by file-size heuristics against the raw DICOM data.

**No mouse receives both tracers.** Within each week, the cohort is split cleanly by ID range: lower IDs → NaF (calcification), higher IDs → FDG (inflammation). These are parallel sub-experiments, not multi-modal single-animal scans.

#### Week 12 — 42 subjects  (21 Hi-res CT · 21 Lo-res CT · 20 PET-FDG · 21 PET-NaF)

| Scan | m54215 | m54216 | m54217 | m54218 | m54219 | m54221 | m54222 | m54223 | m54223_1 | m54224 | m54225 | m54226 | m54227 | m54228 | m54229 | m54231 | m54232 | m54233 | m54234 | m54244 | m54253 | m54254 | m54255 | m54256 | m54257 | m54258 | m54259 | m54260 | m54261 | m54262 | m54263 | m54264 | m54265 | m54266 | m54267 | m54268 | m54269 | m54270 | m54271 | m54272 | m54301 | m54302 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hi-res CT** |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  | ✓ |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ |

#### Week 15 — 47 subjects  (24 Hi-res CT · 20 Lo-res CT · 20 PET-FDG · 22 PET-NaF)

| Scan | m54389 | m54390 | m54391 | m54392 | m54393 | m54394 | m54395 | m54396 | m54397 | m54398 | m54399 | m54400 | m54400_1 | m54400_2 | m54401 | m54402 | m54403 | m54403_1 | m54404 | m54404_1 | m54405 | m54406 | m54407 | m54407_1 | m54407_2 | m54408 | m54408_1 | m54498 | m54499 | m54500 | m54501 | m54502 | m54503 | m54504 | m54505 | m54506 | m54507 | m54515 | m54516 | m54517 | m54518 | m54519 | m54520 | m54521 | m54522 | m54523 | m54524 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hi-res CT** |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

#### Week 18 — 33 subjects  (10 Hi-res CT · 16 Lo-res CT · 17 PET-FDG · 16 PET-NaF)

| Scan | m54632 | m54633 | m54634 | m54635 | m54636 | m54637 | m54638 | m54639 | m54640 | m54641 | m54642 | m54643 | m54644 | m54645 | m54646 | m54647 | m54675 | m54676 | m54677 | m54678 | m54679 | m54680 | m54681 | m54682 | m54683 | m54684 | m54685 | m54686 | m54687 | m54688 | m54688_1 | m54689 | m54690 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hi-res CT** |  |  |  |  | ✓ | ✓ |  | ✓ |  |  |  |  |  |  |  | ✓ |  |  |  |  |  |  |  |  | ✓ | ✓ |  | ✓ | ✓ |  | ✓ |  | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

#### Week 20 — 25 subjects  (12 Hi-res CT · 12 Lo-res CT · 12 PET-FDG · 12 PET-NaF)

| Scan | m54762 | m54763 | m54764 | m54765 | m54766 | m54767 | m54768 | m54769 | m54770 | m54771 | m54772 | m54773 | m54781 | m54782 | m54783_1 | m54784 | m54784_1 | m54785 | m54786 | m54787 | m54788 | m54789 | m54790 | m54791 | m54792 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hi-res CT** |  |  |  | ✓ | ✓ | ✓ |  |  |  | ✓ | ✓ | ✓ |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ |  |  |  | ✓ | ✓ | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |  |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |

---

## Setup

### Environment

The project uses a conda environment (`vlm_env`, Python 3.10). Reproduce it exactly with:

```bash
conda env create -f environment.yml
conda activate vlm_env
```

pip fallback (non-conda users): `pip install -r requirements.txt`

### Configuration

All server-specific paths and tunable parameters live in `config.yaml`, which is **gitignored**. Copy the committed template and fill in your paths before running anything:

```bash
cp config.yaml.example config.yaml
# Then edit config.yaml — at minimum set:
#   paths.data_root  → path to the /Dicom Data/ directory on your server
#   paths.output_dir → where converted NIfTI files should be written
```

The file size heuristic targets (`pet_kb`, `ct_hi_res_kb`, `ct_lo_res_kb`) and tolerance are also in `config.yaml` and can be tuned without touching the code.

### Scripts

| File | Purpose | Dependencies |
|---|---|---|
| `scripts/dicom_to_nifti.py` | Converts a single DICOM subject folder to NIfTI (`.nii.gz`). Separates PET and CT by file size, sorts slices by Z-position, and writes a `manifest.csv`. Single-subject test mode; reads all settings from `config.yaml`. | `SimpleITK`, `pydicom`, `PyYAML` |
| `scripts/visualize_nifti.py` | QA tool — renders three orthogonal slices (axial, coronal, sagittal) through the center of a NIfTI volume and saves a `_qa.png` alongside the input. Run after conversion to sanity-check the output. | `SimpleITK`, `matplotlib` |
| `scripts/get_colipri_embeddings.py` | Iterates over all CT NIfTI volumes in `manifest.csv`, passes each through COLIPRI, and saves embeddings to `colipri_embeddings.npz`. | `torch`, `colipri`, `torchio`, `PyYAML` |
| `scripts/evaluate_embeddings.py` | Encoder-agnostic evaluation suite. Accepts any `.npz` conforming to the standard embedding interface and runs a battery of tasks across three tiers (unsupervised, linear probe, longitudinal). Outputs `metrics.csv`, `report.txt`, and dimensionality-reduction plots. | `scikit-learn`, `matplotlib`, `umap-learn` |
| `config.yaml.example` | Committed template for `config.yaml`. Documents all available options. | — |
| `environment.yml` | Full conda environment lockfile (Python 3.10, all packages pinned). Preferred for exact reproducibility. | — |
| `requirements.txt` | pip-only fallback with pinned versions. Mirrors `environment.yml` for the direct project dependencies. | — |

---

## Pipeline

### Step 1: DICOM → NIfTI Conversion

**Script:** `scripts/dicom_to_nifti.py` — run with `python scripts/dicom_to_nifti.py` from the repo root (configure subject in `config.yaml`).

You cannot load 691 raw slices into a model. Convert each series into a single 3D volume.

1. Read `SeriesInstanceUID` of every file in the folder
2. Group files by UID (separates CT from PET if mixed)
3. Sort by `InstanceNumber` or `ImagePositionPatient` (z-axis)
4. Stack into a `(Depth, Height, Width)` numpy array
5. Save as **NIfTI** (`.nii.gz`)

Also generate a **manifest CSV** with columns: `SubjectID`, `Week`, `Modality`, `Path`.

**QA:** After conversion, run `python scripts/visualize_nifti.py <path/to/output.nii.gz>` to generate a `_qa.png` with orthogonal slice views. Confirm anatomy looks correct before proceeding.

### Step 2: Preprocessing — 3D Patching

**Constraint:** Do not globally resize/downsample (e.g., 512 → 128). Dr. Xu is concerned this blurs the tiny calcifications that are the primary signal.

**Solution:** Cut each NIfTI volume into **3D Patches** (e.g., 96×96×96 cubes). This preserves resolution and matches COLIPRI's expected input format.

### Step 3: Baseline Validation — COLIPRI / RadDINO (Zero-Shot Transfer)

- **Model:** COLIPRI (3D VLM pre-trained on human CTs + reports) or RadDINO (image-only, 3D-friendly)
- **Input:** Hi-Res CT NIfTIs → embeddings via `scripts/get_colipri_embeddings.py`

**Core success criterion:** *Week 12 embeddings cluster apart from Week 20 with no training whatsoever* — this is zero-shot transfer. The **Tier 1** tasks in the evaluation suite test this directly and require no labels at inference time: T1a lets you see it visually (t-SNE/UMAP), while T1b (ARI/NMI) and T1c (silhouette score) put a number on it. If a frozen encoder scores near zero on all three, it cannot distinguish early from late disease and is not a viable baseline.

**Why a shared evaluation harness?** The project will compare multiple encoders — COLIPRI, RadDINO, and a custom MAE (Step 4). Rather than writing one-off analysis code for each, `scripts/evaluate_embeddings.py` is a fixed, encoder-agnostic benchmark. Every encoder produces one `metrics.csv` row using the same tasks and the same CV splits. Swapping encoders means only changing the `--embeddings` argument; everything else is held constant.

**Standard embedding interface** — all encoder scripts must produce a `.npz` with these keys so any encoder plugs directly into the evaluation suite:

| Key | Shape | Description |
|---|---|---|
| `embeddings` | `(N, D)` float32 | One vector per volume |
| `subject_ids` | `(N,)` str | Mouse ID, e.g. `m54253` |
| `weeks` | `(N,)` str | `Week 12` / `Week 15` / `Week 18` / `Week 20` |
| `modalities` | `(N,)` str | `CT_HiRes` / `CT_LowRes` / `PET_FDG` / `PET_NaF` |
| `paths` | `(N,)` str | Absolute path to source NIfTI |

**Evaluation tasks** (`scripts/evaluate_embeddings.py`):

*Tier 1 — Zero-shot (no training, no labels):*

| Task | Metric(s) | What it tests |
|---|---|---|
| **T1a** t-SNE / UMAP plots | — (visual) | Do Week 12 and Week 20 clusters visually separate? ← primary success criterion |
| **T1b** k-means cluster alignment | ARI, NMI | Do unsupervised clusters align with week labels? |
| **T1c** Silhouette by week | Silhouette (cosine) | Are same-week embeddings more cohesive than random? |
| **T1d** Within-subject consistency | Δ (intra − inter sim) | Does the encoder preserve mouse identity across timepoints? |

*Tier 2 — Linear probe (frozen embeddings + one linear layer, LOSO CV):*

| Task | Metric(s) | What it tests |
|---|---|---|
| **T2a** Week classification (4-class) | Accuracy, macro-F1 | Can a linear head predict Week 12/15/18/20? |
| **T2b** Early vs. late (binary) | Accuracy, AUC-ROC | Can a linear head separate Week 12 vs. Week 20? |

*Tier 3 — Longitudinal / relational:*

| Task | Metric(s) | What it tests |
|---|---|---|
| **T3a** Pairwise temporal ordering | Accuracy (chance=0.5) | Do embeddings encode a consistent disease-progression *direction*? |
| **T3b** Same-subject retrieval | Recall@1/3, MRR | For a query scan, are same-mouse scans the nearest neighbors? |
| **T3c** Same-week retrieval | mAP@5 | For a query scan, are same-week scans the nearest neighbors? |

All Tier 2–3 supervised tasks use **Leave-One-Subject-Out (LOSO)** CV — the same mouse never appears in both train and test sets.

**Run:**
```bash
python scripts/get_colipri_embeddings.py          # produces colipri_embeddings.npz
python scripts/evaluate_embeddings.py \
    --embeddings /data1/Processed_NIfTI/colipri_embeddings.npz \
    --output-dir /data1/Processed_NIfTI/eval/colipri/
```

**Output:** `metrics.csv` (one row of all metrics per encoder — load multiple rows to compare encoders side by side), `report.txt` (human-readable summary), `plots/` (t-SNE and UMAP scatter plots).

### Step 4: Custom Model — Train from Scratch

- **Model:** **Masked Autoencoder (MAE)** with a 3D Vision Transformer (ViT) backbone
- **Input:** PET data (FDG + NaF) + CT patches
- **Method:** Mask 75% of the volume and force the model to reconstruct it (**Masked Image Modeling / MIM**)
- **Rationale:** Existing models (e.g., RadDINO) are trained on human X-rays/CTs and likely cannot understand mouse PET scans. Training from scratch lets the model learn mouse-specific inflammation and calcification features. Cross-modal masking (using CT to predict PET, or vice versa) is also worth exploring given paired CT/PET data (see BrainMVP below).

---

## Key Papers

| Paper | Relevance |
|---|---|
| **COLIPRI** | Engineering blueprint for 3D patching. Confirms chunk-based processing preserves resolution without resizing. Primary baseline encoder candidate. |
| **RAD-DINO** | "North Star" for the baseline. Proves image-only pre-training (no text reports) works well — important since metadata is sparse (only Week # and Tracer). |
| **BrainMVP** | Validates **cross-modal masking** (using CT to predict PET and vice versa). Directly relevant for the custom MAE given paired CT/PET data. |
