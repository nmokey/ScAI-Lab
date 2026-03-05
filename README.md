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

## Source Study

**Citation:** Tamboline M et al. (2025). "Preclinical evaluation of high-resolution CT, 18F-FDG, and 18F-NaF PET imaging for longitudinal monitoring of atherosclerosis." *European Journal of Nuclear Medicine and Molecular Imaging* 52:4256–4267.

**Study design:** Male Apoe−/− mice on a high-fat diet (HFD) to induce atherosclerosis; age-matched C57BL/6 wild-type (WT) mice on regular chow as controls. The **same animals** are scanned longitudinally at weeks 12, 15, 18, and 20. After each session, ~3 mice per group are euthanized for histology.

**Two imaging cohorts (separate animals, never overlap):**
- **NaF cohort:** n=20 KO + n=20 WT — 18F-NaF tracer (calcification)
- **FDG cohort:** n=20 KO + n=20 WT — 18F-FDG tracer (inflammation)
- CT acquired on all mice from both cohorts (80 total)

**Key findings (ML context):**
- **18F-NaF** separates KO from WT across all timepoints; strong correlation with histology (r²=0.83). Best overall biomarker.
- **18F-FDG** only distinguishes groups at early stages (Weeks 12–15); indistinguishable by Week 18.
- **Hi-res CT** effective for late-stage detection (Week 15+); calcified plaques visible as hyperdense regions.
- **Implication for ML:** Expect NaF and CT embeddings to separate WT from KO consistently; FDG separation may only be detectable at Weeks 12–15. Use this as a sanity check when evaluating encoder quality.

---

## Data

**80 unique mice** across two cohorts, imaged at 4 timepoints (weeks 12, 15, 18, 20). Persistent mouse IDs: `NaF_WT_1`–`NaF_WT_20`, `NaF_KO_1`–`NaF_KO_20`, `FDG_WT_1`–`FDG_WT_20`, `FDG_KO_1`–`FDG_KO_20`. No mouse receives both tracers.

Each scanner session (`m54xxx`) captures 1–4 mice simultaneously, producing a bag of DICOM slices. Folder structure: `Week X` → `Tracer` → `ScanID`.

**Modalities:**

| Modality | File Size | What it Measures | Project Role |
|---|---|---|---|
| **CT (Hi-Res)** | ~2814 KB | Anatomy / Calcification | Baseline — use for RadDINO/COLIPRI validation |
| **CT (Low-Res)** | ~705 KB | Anatomy (attenuation correction) | Ignore |
| **PET (FDG)** | ~115 KB | Glucose / Inflammation | Novelty — target for custom MAE training |
| **PET (NaF)** | ~115 KB | Calcium / Bone | Critical — measures calcification |

**Formats:** Use `/Dicom Data/` (raw DICOM). The `/Amgen SUV Data/ (.xif)` files are a proprietary format — not needed.

For full scan-level inventory, per-scan mouse mappings, data heuristics, and longitudinal attendance tables, see [DATA_MANIFEST.md](DATA_MANIFEST.md).

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
| `scripts/build_nifti_dataset.py` | **Batch conversion pipeline (3 stages).** Reads `manifest.csv`, applies per-modality consolidation, converts sessions to NIfTIs, segments and crops per-mouse volumes. Generates `mouse_manifest.csv`. Flags: `--dry-run`, `--stage {1,2,3,all}`, `--session <id> [<id> ...]`. | `SimpleITK`, `pydicom`, `scipy`, `PyYAML` |
| `scripts/visualize_nifti.py` | QA tool — renders three orthogonal slices through the center of a NIfTI volume and saves a `_qa.png`. Run after conversion to sanity-check output. | `SimpleITK`, `matplotlib` |
| `scripts/get_colipri_embeddings.py` | Iterates over all CT NIfTI volumes in `mouse_manifest.csv`, passes each through COLIPRI, and saves embeddings to `{output_dir}/embeddings/colipri/colipri_embeddings.npz`. | `torch`, `colipri`, `torchio`, `PyYAML` |
| `scripts/evaluate_embeddings.py` | Encoder-agnostic evaluation suite. Accepts any `.npz` conforming to the standard embedding interface and runs unsupervised, linear probe, and longitudinal tasks. Outputs `metrics.csv`, `report.txt`, and dimensionality-reduction plots. | `scikit-learn`, `matplotlib`, `umap-learn` |

---

## Pipeline

### Step 1: DICOM → NIfTI Conversion

**Batch conversion:** `scripts/build_nifti_dataset.py` reads `manifest.csv` and `config.yaml`; writes to `{output_dir}/`. Use `--dry-run` to verify session selection without writing files.

```bash
python scripts/build_nifti_dataset.py --dry-run           # preview all 63 sessions
python scripts/build_nifti_dataset.py --stage 1           # DICOM → session NIfTIs only
python scripts/build_nifti_dataset.py --stage 2           # segment animals from session NIfTIs
python scripts/build_nifti_dataset.py --stage 3           # crop and write per-mouse NIfTIs
python scripts/build_nifti_dataset.py                     # all stages (default)
python scripts/build_nifti_dataset.py --session m54226    # process one session only
python scripts/build_nifti_dataset.py --session m54226 m54399 --stage 2  # filter + stage
```

The dataset inventory lives in two CSVs:
- **`manifest.csv`** — session-level DICOM inventory (147 rows, 63 valid sessions after filtering `is_valid=False`). Never modified by the pipeline.
- **`mouse_manifest.csv`** — per-mouse NIfTI index generated by the pipeline. One row per (mouse × week) pair, with paths to cropped `ct_hi.nii.gz` and `pet_*.nii.gz`. This is the ML-facing index.

**Known data gaps (not pipeline bugs):**

| Session / Mouse | Week | Issue |
|---|---|---|
| `m54232` | 12 | CT-Hi DICOMs are blank attenuation-correction scan (mean HU −1015, max 366). Marked `is_valid=False`. |
| `m54267` | 12 | Same as m54232 (mean HU −1026). Marked `is_valid=False`. |
| `m54215`, `m54222`, `m54257` | 12 | CT-Lo only sessions — no CT-Hi DICOM exists for these scans. |
| `NaF_KO_05`–`NaF_KO_08` | 12 | Affected by the m54232/m54267/m54257 gaps above — no CT-Hi available. |
| `FDG_WT_17`–`FDG_WT_20` | 12 | Affected by the m54215/m54222 gaps — no CT-Hi available. |

**Three-stage pipeline:**

**Stage 1 — DICOM → session NIfTI:**
1. Classify DICOM slices by file-size heuristic (PET ~115 KB, CT-Hi ~2814 KB, CT-Lo ~705 KB; ±5% tolerance)
2. Apply per-modality version consolidation (`_1`/`_2` re-scan suffixes — take highest valid version per modality)
3. Sort slices by `ImagePositionPatient` Z-coordinate and stack into a 3D NIfTI
4. Write combined (all-mice) volumes to `sessions/{week}/{tracer}/{scan_id}/ct_hi.nii.gz` and `pet_*.nii.gz`

**Stage 2 — Animal segmentation (quadrant-based):**
1. Downsample CT to ~1 mm/voxel isotropic and threshold at ≥ 300 HU (bone only)
2. Compute the physical X/Y coordinates of every bone voxel using the image direction cosines — vectorised, no voxel-direction assumptions
3. Use the **FOV geometric centre** (image centre voxel → physical mm) as the quadrant dividing line
4. Split bone voxels into four quadrants (lower-left, lower-right, upper-left, upper-right) in physical space
5. Each populated quadrant = one mouse; position assigned unambiguously by quadrant — no sorting required
6. Record each mouse's bone centroid (in physical mm) as a degenerate point bbox (`phys_min = phys_max = centroid`)
7. **Lower-row mice only (positions 1 & 2):** shift centroid 5 mm toward the FOV Y centre to add padding toward the scan midline (upper-row mice are unaffected)

**Stage 3 — Per-mouse crop:**
1. Map the point bbox to each image's voxel space via `TransformPhysicalPointToIndex` (handles different CT/PET spacings and origins correctly)
2. Expand symmetrically by fixed padding: **20 mm XY** (transaxial) + **70 mm Z** (longitudinal) — produces identical crop dimensions for every mouse
3. Reorient to **RAS** convention (`sitk.DICOMOrient(..., "RAS")`) for consistent axis ordering across all viewers
4. Write cropped volumes to `mice/{cohort}_{genotype}_{num:02d}/week_{N}/ct_hi.nii.gz` and `pet_*.nii.gz`

**Segmentation approach history — methods tried and abandoned:**

| Approach | Problem |
|---|---|
| **Connected components + centroid sort** (original) | Sorting by `(centroid_y, centroid_x)` broke when two same-row mice had nearly identical Y centroids (< 2 mm apart). The lower-left/lower-right assignment flipped unpredictably depending on which mouse happened to have a slightly lower centroid. |
| **X-axis inversion fix** | Physical X is inverted relative to visual left (higher physical X = visually left). Fixing the sort key to `(centroid_y, -centroid_x)` corrected upper-row mice but did not fix the same-row Y-tie problem. |
| **Half-split sort** (rank by Y, then sort each half by X) | Partially mitigated same-row ties but still produced wrong assignments when one mouse's bone mass extended slightly into the other mouse's half. |
| **Connected components + bone-centroid quadrant split** | Using the overall bone centroid as the quadrant divider is biased when < 4 mice are scanned — the centroid shifts toward occupied positions, misplacing the dividing line. |
| **n_mice-based branching** (≤2 X-only, ==3 partial, ==4 quadrant) | Required knowing the expected number of mice in advance; brittle when detection returned the wrong count. |
| **Fixed padding too large (35 mm XY)** | Produced 701-voxel (70 mm) crops at 0.1 mm/voxel spacing, including adjacent mice. Reduced to 20 mm (401-voxel / 40 mm crops). |
| **Centroid point bbox with no bias** | Lower-row mice had nearly zero padding toward the scan midline (inner edge), cutting off medial anatomy. Fixed by shifting lower-row centroids 5 mm toward the FOV Y centre. |

**Current approach rationale:** The FOV geometric centre is always the midpoint between the four scanner bed positions regardless of how many mice are present, so it never drifts toward occupied positions. Empty quadrants produce no bbox and are silently skipped. All four positions use a single unified code path — no `n_mice` branching required.

**QA:** After conversion, run `python scripts/visualize_nifti.py <path/to/output.nii.gz>` to generate a `_qa.png`. Confirm anatomy looks correct before proceeding.

### Step 2: Preprocessing — 3D Patching

**Constraint:** Do not globally resize/downsample (e.g., 512 → 128) — this blurs the tiny calcifications that are the primary signal.

**Solution:** Cut each NIfTI volume into **3D patches** (e.g., 96×96×96 cubes). This preserves resolution and matches COLIPRI's expected input format.

### Step 3: Baseline Validation — COLIPRI / RadDINO (Zero-Shot Transfer)

- **Model:** COLIPRI (3D VLM pre-trained on human CTs + reports) or RadDINO (image-only, 3D-friendly)
- **Input:** Hi-Res CT NIfTIs → embeddings via `scripts/get_colipri_embeddings.py`

**Core success criterion:** *Week 12 embeddings cluster apart from Week 20 with no training whatsoever* — zero-shot transfer. The Tier 1 tasks test this directly.

**Standard embedding interface** — all encoder scripts must produce a `.npz` with these keys:

| Key | Shape | Description |
|---|---|---|
| `embeddings` | `(N, D)` float32 | One vector per volume |
| `subject_ids` | `(N,)` str | Mouse ID, e.g. `NaF_WT_03` |
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
| **T3a** Pairwise temporal ordering | Accuracy (chance=0.5) | Do embeddings encode a consistent disease-progression direction? |
| **T3b** Same-subject retrieval | Recall@1/3, MRR | For a query scan, are same-mouse scans the nearest neighbors? |
| **T3c** Same-week retrieval | mAP@5 | For a query scan, are same-week scans the nearest neighbors? |

All Tier 2–3 supervised tasks use **Leave-One-Subject-Out (LOSO)** CV — the same mouse never appears in both train and test sets.

**Run:**
```bash
python scripts/get_colipri_embeddings.py          # produces {output_dir}/embeddings/colipri/colipri_embeddings.npz
python scripts/evaluate_embeddings.py \
    --embeddings {output_dir}/embeddings/colipri/colipri_embeddings.npz \
    --output-dir {output_dir}/eval/colipri/
```

**Output:** `metrics.csv` (one row per encoder — load multiple to compare), `report.txt`, `plots/`.

### Step 4: Custom Model — Train from Scratch

- **Model:** **Masked Autoencoder (MAE)** with a 3D Vision Transformer (ViT) backbone
- **Input:** PET data (FDG + NaF) + CT patches
- **Method:** Mask 75% of the volume and force the model to reconstruct it (Masked Image Modeling / MIM)
- **Rationale:** Existing models (e.g., RadDINO) are trained on human X-rays/CTs and likely cannot understand mouse PET scans. Training from scratch lets the model learn mouse-specific inflammation and calcification features. Cross-modal masking (using CT to predict PET, or vice versa) is also worth exploring given paired CT/PET data.

---

## Key Papers

| Paper | Relevance |
|---|---|
| **COLIPRI** | Engineering blueprint for 3D patching. Confirms chunk-based processing preserves resolution without resizing. Primary baseline encoder candidate. |
| **RAD-DINO** | "North Star" for the baseline. Proves image-only pre-training (no text reports) works well — important since metadata is sparse (only Week # and Tracer). |
| **BrainMVP** | Validates **cross-modal masking** (using CT to predict PET and vice versa). Directly relevant for the custom MAE given paired CT/PET data. |
