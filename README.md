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

**Study design:** Male Apoe−/− mice placed on a high-fat diet (HFD) starting at 6 weeks of age to induce atherosclerosis; age-matched C57BL/6 wild-type (WT) mice on regular chow served as controls. Mice were imaged longitudinally at 12, 15, 18, and 20 weeks of age — the **same animals are scanned at every timepoint** (not separate sacrificed cohorts).

**Two imaging cohorts (separate animals, never overlap):**
- **NaF cohort:** n=20 KO (Apoe−/−, HFD) + n=20 WT (C57BL/6, regular chow) → 40 unique mice
- **FDG cohort:** n=20 KO + n=20 WT → 40 unique mice (different physical animals from NaF cohort)
- CT acquired on all mice from both cohorts combined (n=40 KO + 40 WT = 80 total)

**Imaging protocol:**
- **18F-NaF** (calcification tracer): 4.07 MBq i.v. → PET scan at **1h** (6-min, 200µm CT) and **3h** post-injection (9-min, 100µm hi-res CT)
- **18F-FDG** (inflammation tracer): 8.33 MBq i.v. → PET scan at **3h** (6-min, 200µm CT) and **5h** post-injection (9-min, 100µm hi-res CT)
- Hi-res CT accompanies the later timepoint scan in each session (3h NaF / 5h FDG)

**Post-imaging sacrifice:** After each imaging session, ~n=3 per group are euthanized for histological validation (Sudan IV staining, CD68-IHC for macrophages, Ferangi Blue for calcium). Cohort size shrinks from 20 → 18 → 15 → 12 per group per cohort across weeks 12–20.

**Key findings (ML context):**
- **18F-NaF** separates KO from WT across all timepoints; signal peaks at Week 18. Strong correlation with histological disease severity (r²=0.83). Best overall biomarker.
- **18F-FDG** only distinguishes groups at early stages (Weeks 12–15); KO and WT become indistinguishable by Week 18.
- **Hi-res CT** effective for late-stage detection (Week 15+); calcified plaques visible as hyperdense regions.
- **Implication for ML:** Expect NaF and CT embeddings to separate WT from KO consistently; FDG separation may only be detectable at Weeks 12–15. Use this as a sanity check when evaluating encoder quality.

---

## Data

### Subjects & Timeline

- **80 unique mice** (40 NaF cohort + 40 FDG cohort) imaged **longitudinally** — the **same mice** are rescanned at weeks 12, 15, 18, and 20. After each imaging session, ~3 mice per group are euthanized for histology, so cohort size decreases over time.
  - **NaF cohort:** 20 WT (Control) + 20 KO (Disease) — separate physical animals from FDG cohort
  - **FDG cohort:** 20 WT (Control) + 20 KO (Disease) — separate physical animals from NaF cohort
  - **No mouse receives both tracers.** NaF and FDG cohorts are entirely different animals.
  - **Persistent mouse IDs:** `NaF_WT_1`–`NaF_WT_20`, `NaF_KO_1`–`NaF_KO_20`, `FDG_WT_1`–`FDG_WT_20`, `FDG_KO_1`–`FDG_KO_20`
- **4 Timepoints:** Week 12, 15, 18, 20
  - Week 12 = Early Stage (healthy-ish); Week 20 = Late Stage (diseased)
  - Week 12: 42 scan sessions · Week 15: 47 sessions · Week 18: 33 sessions · Week 20: 25 sessions
  - Not all sessions have all modalities (see Scan Inventory below)
- **Folder structure:** `Week X` → `Tracer` → `ScanID` (e.g., `m54253`)
  - **Scan IDs (m54xxx) identify scanner sessions, not individual mice.** Each session captures 1–4 mice scanned simultaneously.
  - Each scan folder contains a "bag of slices" (e.g., 691 separate `.dcm` files)

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

1. **The `_1` Rule:** If both `m54223` and `m54223_1` exist, they represent sequential re-scan attempts. Use a **per-modality merge**: for each modality, take it from the latest version (highest `_N` suffix) that contains a valid copy. If the final re-scan has CT-Hi but failed to capture PET, fall back to the original session's PET. If the original is absent from the DICOM inventory entirely, use the `_1` version.
2. **Modality ID:** Use **file size** as the primary heuristic to identify slice modality (per PI). See the Modalities table for target sizes; apply a ±5% tolerance window. DICOM tags (`Modality`, `SeriesInstanceUID`) may be used as supplementary validation.
3. **CT Fallback:** Not all scans have Hi-Res CT. Disregard scans with Lo-Res CT — these are not detailed enough for rich embeddings. Exclude sessions lacking Hi-Res CT from the processed NIfTI dataset entirely.
4. **Mouse Positioning in Scanner:** Within each scan session, mice are arranged in a **2×2 grid** (X-Y plane of the combined 3D volume). Mouse numbers map to spatial positions as: 1 = lower-left, 2 = lower-right, 3 = upper-left, 4 = upper-right. Sessions with 2 mice use positions 1–2; sessions with 3 mice fill positions 1–3 (one slot empty). This is used to assign per-animal crops back to persistent mouse IDs.
5. **Ignore:** Files ending in `.im3`, `.vol`, `.raw`.

### Scan Inventory

Each scanner session (m54xxx) captures 1–4 mice imaged simultaneously. **Mouse IDs persist across weeks** — the same cohort animals are rescanned at weeks 12, 15, 18, and 20. The inventory below was determined by file-size heuristics against the raw DICOM data.

**No mouse receives both tracers** — NaF and FDG cohorts are entirely separate animals (~40 unique mice each). Within each week, lower scan ID ranges are NaF sessions and higher ranges are FDG sessions.

#### Week 12 — 42 sessions  (21 Hi-res CT · 21 Lo-res CT · 20 PET-FDG · 20 PET-NaF)

| Scan | m54215 | m54216 | m54217 | m54218 | m54219 | m54221 | m54222 | m54223 | m54223_1 | m54224 | m54225 | m54226 | m54227 | m54228 | m54229 | m54231 | m54232 | m54233 | m54234 | m54244 | m54253 | m54254 | m54255 | m54256 | m54257 | m54258 | m54259 | m54260 | m54261 | m54262 | m54263 | m54264 | m54265 | m54266 | m54267 | m54268 | m54269 | m54270 | m54271 | m54272 | m54301 | m54302 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mice** | WT 1,2 | WT 3-6 | WT 7-10 | WT 11-14 | WT 15-18 | KO 1-4 | KO 5-8 | KO 9-12 | KO 9-12 | KO 13-16 | WT 19,20+1,2 ⚠️ | WT 3-6 | WT 7-10 | WT 11-14 | WT 15-18 | KO 1-4 | KO 5-8 | KO 9-12 | KO 13-16 | WT 19-20 | WT 1-4 | WT 5-8 | WT 9-12 | WT 13-16 | WT 17-20 | KO 1-4 | KO 5-8 | KO 9-12 | KO 13-16 | KO 17-20 | WT 1-4 | WT 5-8 | WT 9-12 | WT 13-16 | WT 17-20 | KO 1-4 | KO 5-8 | KO 9-12 | KO 13-16 | KO 17-20 | KO 17-20 | KO 17-20 |
| **Group** | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Disease | Disease |
| **Hi-res CT** |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  | ✓ |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ |

#### Week 15 — 47 sessions  (24 Hi-res CT · 20 Lo-res CT · 20 PET-FDG · 22 PET-NaF)

| Scan | m54389 | m54390 | m54391 | m54392 | m54393 | m54394 | m54395 | m54396 | m54397 | m54398 | m54399 | m54400 | m54400_1 | m54400_2 | m54401 | m54402 | m54403 | m54403_1 | m54404 | m54404_1 | m54405 | m54406 | m54407 | m54407_1 | m54407_2 | m54408 | m54408_1 | m54498 | m54499 | m54500 | m54501 | m54502 | m54503 | m54504 | m54505 | m54506 | m54507 | m54515 | m54516 | m54517 | m54518 | m54519 | m54520 | m54521 | m54522 | m54523 | m54524 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mice** | WT 1,2 | WT 3-6 | WT 7-10 | WT 11-14 | WT 15-18 | KO 1-4 | KO 5-8 | KO 9-11 | KO 12-15 | KO 16-18 | WT 1,2 | WT 3-6 | WT 3-6 | WT 3-6 | WT 7-10 | WT 11-14 | WT 15-18 | WT 15-18 | KO 1-4 | KO 1-4 | KO 5-8 | KO 9-11 | KO 12-15 | KO 12-15 | KO 12-15 | KO 16-18 | KO 16-18 | KO 1-4 | KO 5-8 | KO 9-12 | KO 13-15 | KO 16-18 | KO 1-4 | KO 5-8 | KO 9-12 | KO 13-15 | KO 16-18 | WT 1,2 | WT 3-6 | WT 7-10 | WT 11-14 | WT 15-18 | WT 1,2 | WT 3-6 | WT 7-10 | WT 11-14 | WT 15-18 |
| **Group** | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Control | Control | Control | Control | Control | Control |
| **Hi-res CT** |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

#### Week 18 — 33 sessions  (10 Hi-res CT · 16 Lo-res CT · 17 PET-FDG · 16 PET-NaF)

| Scan | m54632 | m54633 | m54634 | m54635 | m54636 | m54637 | m54638 | m54639 | m54640 | m54641 | m54642 | m54643 | m54644 | m54645 | m54646 | m54647 | m54675 | m54676 | m54677 | m54678 | m54679 | m54680 | m54681 | m54682 | m54683 | m54684 | m54685 | m54686 | m54687 | m54688 | m54688_1 | m54689 | m54690 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mice** | WT 1-4 | WT 5-7 | WT 8-11 | WT 12-15 | WT 1-4 | WT 5-7 | WT 8-11 | WT 12-15 | KO 1-3 | KO 4-7 | KO 8-11 | KO 12-15 | KO 2,3 ⚠️ | KO 5-7 | KO 8-11 | KO 12-15 | WT 1-4 | WT 5-8 | WT 9-12 | WT 13-16 | KO 1-4 | KO 5-8 | KO 9-12 | KO 13-15 | WT 1-4 | WT 5-8 | WT 9-12 | WT 13-16 | KO 1-4 | KO 5-8 | KO 5-8 | KO 9-12 | KO 13-15 |
| **Group** | Control | Control | Control | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease |
| **Hi-res CT** |  |  |  |  | ✓ | ✓ |  | ✓ |  |  |  |  |  |  |  | ✓ |  |  |  |  |  |  |  |  | ✓ | ✓ |  | ✓ | ✓ |  | ✓ |  | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

#### Week 20 — 25 sessions  (12 Hi-res CT · 12 Lo-res CT · 12 PET-FDG · 12 PET-NaF)

| Scan | m54762 | m54763 | m54764 | m54765 | m54766 | m54767 | m54768 | m54769 | m54770 | m54771 | m54772 | m54773 | m54781 | m54782 | m54783_1 | m54784 | m54784_1 | m54785 | m54786 | m54787 | m54788 | m54789 | m54790 | m54791 | m54792 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Mice** | WT 1-4 | WT 5-8 | WT 9-12 | WT 1-4 | WT 5-8 | WT 9-12 | KO 1-4 | KO 5-8 | KO 9-12 | KO 1-4 | KO 5-8 | KO 9-12 | KO 1-4 | KO 5-8 | KO 9-12 | WT 1-4 | WT 1-4 | WT 5-8 | WT 9-12 | KO 1-4 | KO 5-8 | KO 9,11,12 ⚠️ | WT 1-4 | WT 5-8 | WT 9-12 |
| **Group** | Control | Control | Control | Control | Control | Control | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Disease | Control | Control | Control | Control | Disease | Disease | Disease | Control | Control | Control |
| **Hi-res CT** |  |  |  | ✓ | ✓ | ✓ |  |  |  | ✓ | ✓ | ✓ |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Lo-res CT** | ✓ | ✓ | ✓ |  |  |  | ✓ | ✓ | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |  |  |
| **PET (FDG)** |  |  |  |  |  |  |  |  |  |  |  |  | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PET (NaF)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Per-Scan Mouse Mapping (XIF Annotations)

Per-mouse metadata derived from AMIDE XIF filenames in `/data1/Amgen SUV Data/`. XIF filenames encode: scan ID, genotype (WT/KO), mouse numbers, and post-injection timepoint. **Mouse numbers are persistent longitudinal IDs** — the same number refers to the same physical animal across all four weeks.

Faulty original scans (superseded by a `_1` or `_2` re-scan) are excluded from the tables below; excluded scans are listed in the footnote of each section. **Modality abbreviations:** CT-Hi = Hi-res CT (~2814 KB); CT-Lo = Lo-res CT (~705 KB); NaF = PET(NaF); FDG = PET(FDG). Modalities detected by file-size heuristic against the DICOM data.

#### Week 12 NaF

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54215 | 1, 2 | WT | Control | 1h | CT-Lo, NaF |
| m54216 | 3, 4, 5, 6 | WT | Control | 1h | CT-Lo, NaF |
| m54217 | 7, 8, 9, 10 | WT | Control | 1h | CT-Lo, NaF |
| m54218 | 11, 12, 13, 14 | WT | Control | 1h | CT-Lo, NaF |
| m54219 | 15, 16, 17, 18 | WT | Control | 1h | CT-Lo, NaF |
| m54221 | 1, 2, 3, 4 | KO | Disease | 1h | CT-Lo, NaF |
| m54222 | 5, 6, 7, 8 | KO | Disease | 1h | CT-Lo, NaF |
| m54223_1 | 9, 10, 11, 12 | KO | Disease | 1h | CT-Lo ⚠️ |
| m54224 | 13, 14, 15, 16 | KO | Disease | 1h | CT-Lo, NaF |
| m54225 ⚠️ | WT 19,20 @1h + WT 1,2 @3h | WT | Control | 1h + 3h mixed | CT-Hi, NaF |
| m54226 | 3, 4, 5, 6 | WT | Control | 3h | CT-Hi, NaF |
| m54227 | 7, 8, 9, 10 | WT | Control | 3h | CT-Hi, NaF |
| m54228 | 11, 12, 13, 14 | WT | Control | 3h | CT-Hi, NaF |
| m54229 | 15, 16, 17, 18 | WT | Control | 3h | CT-Hi, NaF |
| m54231 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Hi, NaF |
| m54232 ⚠️ | 5, 6, 7, 8 | KO | Disease | 3h | CT-Hi, NaF |
| m54233 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Hi, NaF |
| m54234 | 13, 14, 15, 16 | KO | Disease | 3h | CT-Hi, NaF |
| m54244 | 19, 20 | WT | Control | 3h | CT-Hi, NaF |
| m54301 | 17, 18, 19, 20 | KO | Disease | 1h | CT-Lo, NaF |
| m54302 | 17, 18, 19, 20 | KO | Disease | 3h | CT-Hi, NaF |

*Excluded: m54223 (faulty original; superseded by m54223_1).* ⚠️ m54223_1: NaF PET not detected in DICOM — only CT-Lo found; may need manual investigation. ⚠️ m54225: mixed-timepoint session. ⚠️ m54232: XIF uses range notation "m54232-m54250"; DICOM folder is m54232.

#### Week 12 FDG

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54253 | 1, 2, 3, 4 | WT | Control | 3h | CT-Lo, FDG |
| m54254 | 5, 6, 7, 8 | WT | Control | 3h | CT-Lo, FDG |
| m54255 | 9, 10, 11, 12 | WT | Control | 3h | CT-Lo, FDG |
| m54256 | 13, 14, 15, 16 | WT | Control | 3h | CT-Lo, FDG |
| m54257 | 17, 18, 19, 20 | WT | Control | 3h | CT-Lo, FDG |
| m54258 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Lo, FDG |
| m54259 | 5, 6, 7, 8 | KO | Disease | 3h | CT-Lo, FDG |
| m54260 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Lo, FDG |
| m54261 | 13, 14, 15, 16 | KO | Disease | 3h | CT-Lo, FDG |
| m54262 | 17, 18, 19, 20 | KO | Disease | 3h | CT-Lo, FDG |
| m54263 | 1, 2, 3, 4 | WT | Control | 5h | CT-Hi, FDG |
| m54264 | 5, 6, 7, 8 | WT | Control | 5h | CT-Hi, FDG |
| m54265 | 9, 10, 11, 12 | WT | Control | 5h | CT-Hi, FDG |
| m54266 | 13, 14, 15, 16 | WT | Control | 5h | CT-Hi, FDG |
| m54267 | 17, 18, 19, 20 | WT | Control | 5h | CT-Hi, FDG |
| m54268 | 1, 2, 3, 4 | KO | Disease | 5h | CT-Hi, FDG |
| m54269 | 5, 6, 7, 8 | KO | Disease | 5h | CT-Hi, FDG |
| m54270 | 9, 10, 11, 12 | KO | Disease | 5h | CT-Hi, FDG |
| m54271 | 13, 14, 15, 16 | KO | Disease | 5h | CT-Hi, FDG |
| m54272 | 17, 18, 19, 20 | KO | Disease | 5h | CT-Hi, FDG |

*No re-scans in Week 12 FDG.*

#### Week 15 NaF (mice 19–20 sacrificed after Week 12)

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54389 | 1, 2 | WT | Control | 1h | CT-Lo, NaF |
| m54390 | 3, 4, 5, 6 | WT | Control | 1h | CT-Lo, NaF |
| m54391 | 7, 8, 9, 10 | WT | Control | 1h | CT-Lo, NaF |
| m54392 | 11, 12, 13, 14 | WT | Control | 1h | CT-Lo, NaF |
| m54393 | 15, 16, 17, 18 | WT | Control | 1h | CT-Lo, NaF |
| m54394 | 1, 2, 3, 4 | KO | Disease | 1h | CT-Lo, NaF |
| m54395 | 5, 6, 7, 8 | KO | Disease | 1h | CT-Lo, NaF |
| m54396 | 9, 10, 11 | KO | Disease | 1h | CT-Lo, NaF |
| m54397 | 12, 13, 14, 15 | KO | Disease | 1h | CT-Lo, NaF |
| m54398 | 16, 17, 18 | KO | Disease | 1h | CT-Lo, NaF |
| m54399 | 1, 2 | WT | Control | 3h | CT-Hi, NaF |
| m54400_2 | 3, 4, 5, 6 | WT | Control | 3h | CT-Hi, NaF |
| m54401 | 7, 8, 9, 10 | WT | Control | 3h | CT-Hi, NaF |
| m54402 | 11, 12, 13, 14 | WT | Control | 3h | CT-Hi, NaF |
| m54403_1 | 15, 16, 17, 18 | WT | Control | 3h | CT-Hi, NaF |
| m54404_1 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Hi, NaF |
| m54405 | 5, 6, 7, 8 | KO | Disease | 3h | CT-Hi, NaF |
| m54406 | 9, 10, 11 | KO | Disease | 3h | CT-Hi, NaF |
| m54407_2 | 12, 13, 14, 15 | KO | Disease | 3h | CT-Hi, NaF † |
| m54408_1 | 16, 17, 18 | KO | Disease | 3h | CT-Hi, NaF |

*Excluded: m54400 and m54400_1 (faulty; superseded by m54400_2), m54403 (faulty; superseded by m54403_1), m54404 (faulty; superseded by m54404_1), m54407_1 (intermediate re-scan; superseded by m54407_2), m54408 (faulty; superseded by m54408_1).*

† m54407_2: CT-Hi sourced from m54407_2 DICOM; NaF PET sourced from m54407 DICOM (the original scan). The original m54407 CT was faulty but its PET is intact — both modalities are retained via the per-modality merge in Heuristic #1. m54407 itself is excluded from direct use but its PET DICOM is read during NIfTI build.

#### Week 15 FDG (mice 19–20 sacrificed after Week 12)

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54498 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Lo, FDG |
| m54499 | 5, 6, 7, 8 | KO | Disease | 3h | CT-Lo, FDG |
| m54500 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Lo, FDG |
| m54501 | 13, 14, 15 | KO | Disease | 3h | CT-Lo, FDG |
| m54502 | 16, 17, 18 | KO | Disease | 3h | CT-Lo, FDG |
| m54503 | 1, 2, 3, 4 | KO | Disease | 5h | CT-Hi, FDG |
| m54504 | 5, 6, 7, 8 | KO | Disease | 5h | CT-Hi, FDG |
| m54505 | 9, 10, 11, 12 | KO | Disease | 5h | CT-Hi, FDG |
| m54506 | 13, 14, 15 | KO | Disease | 5h | CT-Hi, FDG |
| m54507 | 16, 17, 18 | KO | Disease | 5h | CT-Hi, FDG |
| m54515 | 1, 2 | WT | Control | 3h | CT-Lo, FDG |
| m54516 | 3, 4, 5, 6 | WT | Control | 3h | CT-Lo, FDG |
| m54517 | 7, 8, 9, 10 | WT | Control | 3h | CT-Lo, FDG |
| m54518 | 11, 12, 13, 14 | WT | Control | 3h | CT-Lo, FDG |
| m54519 | 15, 16, 17, 18 | WT | Control | 3h | CT-Lo, FDG |
| m54520 | 1, 2 | WT | Control | 5h | CT-Hi, FDG |
| m54521 | 3, 4, 5, 6 | WT | Control | 5h | CT-Hi, FDG |
| m54522 | 7, 8, 9, 10 | WT | Control | 5h | CT-Hi, FDG |
| m54523 | 11, 12, 13, 14 | WT | Control | 5h | CT-Hi, FDG |
| m54524 | 15, 16, 17, 18 | WT | Control | 5h | CT-Hi, FDG |

*No re-scans in Week 15 FDG.*

#### Week 18 NaF (mice 16–18 sacrificed after Week 15)

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54632 | 1, 2, 3, 4 | WT | Control | 1h | CT-Lo, NaF |
| m54633 | 5, 6, 7 | WT | Control | 1h | CT-Lo, NaF |
| m54634 | 8, 9, 10, 11 | WT | Control | 1h | CT-Lo, NaF |
| m54635 | 12, 13, 14, 15 | WT | Control | 1h | CT-Lo, NaF |
| m54636 | 1, 2, 3, 4 | WT | Control | 3h | CT-Hi, NaF |
| m54637 | 5, 6, 7 | WT | Control | 3h | CT-Hi, NaF |
| m54638 | 8, 9, 10, 11 | WT | Control | 3h | NaF ⚠️ |
| m54639 | 12, 13, 14, 15 | WT | Control | 3h | CT-Hi, NaF |
| m54640 | 1, 2, 3 | KO | Disease | 1h | CT-Lo, NaF |
| m54641 | 4, 5, 6, 7 | KO | Disease | 1h | CT-Lo, NaF |
| m54642 | 8, 9, 10, 11 | KO | Disease | 1h | CT-Lo, NaF |
| m54643 | 12, 13, 14, 15 | KO | Disease | 1h | CT-Lo, NaF |
| m54644 ⚠️ | 2, 3 | KO | Disease | 3h | NaF ⚠️ |
| m54645 | 5, 6, 7 | KO | Disease | 3h | NaF ⚠️ |
| m54646 | 8, 9, 10, 11 | KO | Disease | 3h | NaF ⚠️ |
| m54647 | 12, 13, 14, 15 | KO | Disease | 3h | CT-Hi, NaF |

*No re-scans in Week 18 NaF.* ⚠️ m54638, m54644, m54645, m54646: no CT detected (NaF PET only). ⚠️ m54644: KO mice 1 and 4 absent from all Week 18 3h scans (present at 1h); reason unknown.

#### Week 18 FDG (mice 16–18 sacrificed after Week 15)

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54675 | 1, 2, 3, 4 | WT | Control | 3h | CT-Lo, FDG |
| m54676 | 5, 6, 7, 8 | WT | Control | 3h | CT-Lo, FDG |
| m54677 | 9, 10, 11, 12 | WT | Control | 3h | CT-Lo, FDG |
| m54678 | 13, 14, 15, 16 | WT | Control | 3h | CT-Lo, FDG |
| m54679 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Lo, FDG |
| m54680 | 5, 6, 7, 8 | KO | Disease | 3h | CT-Lo, FDG |
| m54681 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Lo, FDG |
| m54682 | 13, 14, 15 | KO | Disease | 3h | CT-Lo, FDG |
| m54683 | 1, 2, 3, 4 | WT | Control | 5h | CT-Hi, FDG |
| m54684 | 5, 6, 7, 8 | WT | Control | 5h | CT-Hi, FDG |
| m54685 | 9, 10, 11, 12 | WT | Control | 5h | FDG ⚠️ |
| m54686 | 13, 14, 15, 16 | WT | Control | 5h | CT-Hi, FDG |
| m54687 | 1, 2, 3, 4 | KO | Disease | 5h | CT-Hi, FDG |
| m54688_1 | 5, 6, 7, 8 | KO | Disease | 5h | CT-Hi, FDG |
| m54689 | 9, 10, 11, 12 | KO | Disease | 5h | FDG ⚠️ |
| m54690 | 13, 14, 15 | KO | Disease | 5h | CT-Hi, FDG |

*Excluded: m54688 (faulty original; superseded by m54688_1).* ⚠️ m54685, m54689: no CT detected (FDG PET only). Note: WT cohort shows 16 mice at Week 18 (expected 15 from 3 sacrificed; may reflect 1 prior death). KO drops from 18→15 as expected.

#### Week 20 NaF (mice 13–15 sacrificed after Week 18)

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54762 | 1, 2, 3, 4 | WT | Control | 1h | CT-Lo, NaF |
| m54763 | 5, 6, 7, 8 | WT | Control | 1h | CT-Lo, NaF |
| m54764 | 9, 10, 11, 12 | WT | Control | 1h | CT-Lo, NaF |
| m54765 | 1, 2, 3, 4 | WT | Control | 3h | CT-Hi, NaF |
| m54766 | 5, 6, 7, 8 | WT | Control | 3h | CT-Hi, NaF |
| m54767 | 9, 10, 11, 12 | WT | Control | 3h | CT-Hi, NaF |
| m54768 | 1, 2, 3, 4 | KO | Disease | 1h | CT-Lo, NaF |
| m54769 | 5, 6, 7, 8 | KO | Disease | 1h | CT-Lo, NaF |
| m54770 | 9, 10, 11, 12 | KO | Disease | 1h | CT-Lo, NaF |
| m54771 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Hi, NaF |
| m54772 | 5, 6, 7, 8 | KO | Disease | 3h | CT-Hi, NaF |
| m54773 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Hi, NaF |

*No re-scans in Week 20 NaF.*

#### Week 20 FDG (mice 13–15 sacrificed after Week 18)

| Scan ID | Mice | Genotype | Group | Timepoint | Modalities |
|---------|------|----------|-------|-----------|------------|
| m54781 | 1, 2, 3, 4 | KO | Disease | 3h | CT-Lo, FDG |
| m54782 | 5, 6, 7, 8 | KO | Disease | 3h | CT-Lo, FDG |
| m54783_1 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Lo, FDG |
| m54784_1 | 1, 2, 3, 4 | WT | Control | 3h | ⚠️ none detected |
| m54785 | 5, 6, 7, 8 | WT | Control | 3h | CT-Lo, FDG |
| m54786 | 9, 10, 11, 12 | WT | Control | 3h | CT-Lo, FDG |
| m54787 | 1, 2, 3, 4 | KO | Disease | 5h | CT-Hi, FDG |
| m54788 | 5, 6, 7, 8 | KO | Disease | 5h | CT-Hi, FDG |
| m54789 ⚠️ | 9, 11, 12 | KO | Disease | 5h | CT-Hi, FDG |
| m54790 | 1, 2, 3, 4 | WT | Control | 5h | CT-Hi, FDG |
| m54791 | 5, 6, 7, 8 | WT | Control | 5h | CT-Hi, FDG |
| m54792 | 9, 10, 11, 12 | WT | Control | 5h | CT-Hi, FDG |

*Excluded: m54784 (faulty original; superseded by m54784_1).* m54783_1: original m54783 absent from DICOM inventory — m54783_1 is the only scan for these mice. ⚠️ m54784_1: no modalities detected by file-size heuristic despite being the corrected re-scan — investigate manually. ⚠️ m54789: KO mouse 10 present at 3h (m54782) but absent at 5h.

### Per-Mouse Longitudinal Attendance

Mouse numbers are persistent longitudinal IDs within each cohort. WT = Control (C57BL/6, regular chow). KO = Disease (Apoe−/−, high-fat diet). NaF and FDG cohorts are entirely different physical animals.

#### NaF Cohort

| Mouse ID(s) | Week 12 | Week 15 | Week 18 | Week 20 | Notes |
|-------------|---------|---------|---------|---------|-------|
| NaF_WT_1–12 | ✓ | ✓ | ✓ | ✓ | Full longitudinal |
| NaF_WT_13–15 | ✓ | ✓ | ✓ | ✗ | Sacrificed after Week 18 |
| NaF_WT_16–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| NaF_WT_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |
| NaF_KO_1–12 | ✓ | ✓ | ✓* | ✓ | *KO 1 and KO 4 absent at Week 18 3h scan |
| NaF_KO_13–15 | ✓ | ✓ | ✓ | ✗ | Sacrificed after Week 18 |
| NaF_KO_16–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| NaF_KO_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |

#### FDG Cohort

| Mouse ID(s) | Week 12 | Week 15 | Week 18 | Week 20 | Notes |
|-------------|---------|---------|---------|---------|-------|
| FDG_WT_1–12 | ✓ | ✓ | ✓ | ✓ | Full longitudinal |
| FDG_WT_13–16 | ✓ | ✓ | ✓ | ✗ | Sacrificed/died after Week 18 |
| FDG_WT_17–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| FDG_WT_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |
| FDG_KO_1–9, 11–12 | ✓ | ✓ | ✓ | ✓ (3h+5h) | Full longitudinal |
| FDG_KO_10 | ✓ | ✓ | ✓ | ✓ (3h only) | Present at Week 20 3h; absent at 5h |
| FDG_KO_13–15 | ✓ | ✓ | ✓ | ✗ | Sacrificed after Week 18 |
| FDG_KO_16–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| FDG_KO_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |

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
| `scripts/dicom_to_nifti.py` | **Single-subject test tool.** Converts one DICOM subject folder to NIfTI. Separates PET and CT by file size, sorts slices by Z-position. Useful for inspecting an individual scan; not used for batch conversion. | `SimpleITK`, `pydicom`, `PyYAML` |
| `scripts/build_nifti_dataset.py` | **Batch conversion pipeline (3 stages).** Reads `manifest.csv`, applies the per-modality consolidation rule, converts all valid sessions to session-level NIfTIs (`sessions/`), segments individual animals via CT thresholding, and crops per-mouse NIfTIs into `mice/`. Generates `mouse_manifest.csv`. Run with `python scripts/build_nifti_dataset.py [--dry-run] [--stage {1,2,3,all}]`. | `SimpleITK`, `pydicom`, `scipy`, `PyYAML` |
| `scripts/explore_amgen_bboxes.py` | **Read-only XIF/AMIDE inspector.** Probes the Amgen SUV data directory for per-animal bounding box or ROI metadata embedded in `.xif` files. If bounding boxes are found they can be used instead of CT segmentation for animal cropping. Run with `--summary` to scan all files, or `--scan <id> --verbose` to inspect a specific scan. | `PyYAML` |
| `scripts/visualize_nifti.py` | QA tool — renders three orthogonal slices (axial, coronal, sagittal) through the center of a NIfTI volume and saves a `_qa.png` alongside the input. Run after conversion to sanity-check the output. | `SimpleITK`, `matplotlib` |
| `scripts/get_colipri_embeddings.py` | Iterates over all CT NIfTI volumes in `manifest.csv`, passes each through COLIPRI, and saves embeddings to `colipri_embeddings.npz`. | `torch`, `colipri`, `torchio`, `PyYAML` |
| `scripts/evaluate_embeddings.py` | Encoder-agnostic evaluation suite. Accepts any `.npz` conforming to the standard embedding interface and runs a battery of tasks across three tiers (unsupervised, linear probe, longitudinal). Outputs `metrics.csv`, `report.txt`, and dimensionality-reduction plots. | `scikit-learn`, `matplotlib`, `umap-learn` |
| `config.yaml.example` | Committed template for `config.yaml`. Documents all available options. | — |
| `environment.yml` | Full conda environment lockfile (Python 3.10, all packages pinned). Preferred for exact reproducibility. | — |
| `requirements.txt` | pip-only fallback with pinned versions. Mirrors `environment.yml` for the direct project dependencies. | — |

---

## Pipeline

### Step 1: DICOM → NIfTI Conversion

**Batch conversion:** `scripts/build_nifti_dataset.py` — run with `python scripts/build_nifti_dataset.py` from the repo root. Reads `manifest.csv` and `config.yaml`; writes to `{nifti_output_dir}/`. Use `--dry-run` to verify session selection without writing files, or `--stage 1` to run only the DICOM→NIfTI stage.

The dataset inventory lives in two CSVs with distinct roles:
- **`manifest.csv`** — session-level DICOM inventory (147 rows). Never modified by the pipeline.
- **`mouse_manifest.csv`** — per-mouse NIfTI index generated by the pipeline. This is the ML-facing index: one row per (mouse × week) pair, with paths to cropped `ct_hi.nii.gz` and `pet_*.nii.gz`.

The pipeline applies the per-modality merge rule (Heuristic #1) automatically: for each base session, CT-Hi and PET are drawn independently from the highest-suffix version that has a valid copy of each modality. Sessions without CT-Hi in any version are skipped entirely.

You cannot load 691 raw slices into a model. The pipeline converts each series into a single 3D volume:

1. Classify files by size heuristic (PET / CT-Hi / CT-Lo)
2. Sort by `ImagePositionPatient` Z-coordinate
3. Stack into a 3D NIfTI volume and write to `sessions/`
4. Segment individual animals in the combined volume (CT threshold → connected components)
5. Crop per-mouse sub-volumes and write to `mice/`

**Single-subject test:** `scripts/dicom_to_nifti.py` converts one subject configured in `config.yaml`. Use this for quick inspection of an individual scan, not for batch production.

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
