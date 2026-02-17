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

- **~40 Mice** (IDs like `m54253` to `m54272`)
- **4 Timepoints:** Week 12, 15, 18, 20
  - Week 12 = Early Stage (healthy-ish); Week 20 = Late Stage (diseased)
  - Not all mice have all timepoints due to attrition
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
2. **Modality ID:** Use DICOM tags (`Modality`, `SeriesInstanceUID`) to distinguish CT from PET within a folder. File-size heuristics alone are insufficient at the slice level.
3. **Ignore:** Files ending in `.im3`, `.vol`, `.raw`.

---

## Pipeline

### Step 1: DICOM → NIfTI Conversion

You cannot load 691 raw slices into a model. Convert each series into a single 3D volume.

1. Read `SeriesInstanceUID` of every file in the folder
2. Group files by UID (separates CT from PET if mixed)
3. Sort by `InstanceNumber` or `ImagePositionPatient` (z-axis)
4. Stack into a `(Depth, Height, Width)` numpy array
5. Save as **NIfTI** (`.nii.gz`)

Also generate a **manifest CSV** with columns: `SubjectID`, `Week`, `Modality`, `Path`.

### Step 2: Preprocessing — 3D Patching

**Constraint:** Do not globally resize/downsample (e.g., 512 → 128). Dr. Xu is concerned this blurs the tiny calcifications that are the primary signal.

**Solution:** Cut each NIfTI volume into **3D Patches** (e.g., 96×96×96 cubes). This preserves resolution and matches COLIPRI's expected input format.

### Step 3: Baseline — COLIPRI / RadDINO (Zero-Shot Validation)

- **Model:** COLIPRI (3D VLM pre-trained on human CTs + reports) or RadDINO (image-only, 3D-friendly)
- **Input:** Hi-Res CT NIfTIs, fed as 3D patches
- **Method:** Extract embeddings from the encoder; visualize with t-SNE/UMAP
- **Success metric:** Week 12 embeddings naturally cluster apart from Week 20 without any fine-tuning ("zero-shot transfer")

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
