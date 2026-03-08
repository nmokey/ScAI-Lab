# Data Manifest

Detailed inventory of scans, per-mouse mappings, and data heuristics for the atherosclerosis mouse dataset. For a high-level overview, see [README.md](README.md).

---

## Data Heuristics

1. **The `_1` Rule:** If both `m54223` and `m54223_1` exist, they represent sequential re-scan attempts. Use a **per-modality merge**: for each modality, take it from the latest version (highest `_N` suffix) that contains a valid copy. If the final re-scan has CT-Hi but failed to capture PET, fall back to the original session's PET. If the original is absent from the DICOM inventory entirely, use the `_1` version.
2. **Modality ID:** Use **file size** as the primary heuristic to identify slice modality (per PI). See the Modalities table for target sizes; apply a ±5% tolerance window. DICOM tags (`Modality`, `SeriesInstanceUID`) may be used as supplementary validation.
3. **CT Fallback:** Not all scans have Hi-Res CT. Disregard scans with Lo-Res CT — these are not detailed enough for rich embeddings. Exclude sessions lacking Hi-Res CT from the processed NIfTI dataset entirely.
4. **Mouse Positioning in Scanner:** Within each scan session, mice are arranged in a **2×2 grid** (X-Y plane of the combined 3D volume). Mouse numbers map to spatial positions as: 1 = lower-left, 2 = lower-right, 3 = upper-left, 4 = upper-right. Sessions with 2 mice use positions 1–2; sessions with 3 mice fill positions 1–3 (one slot empty). This is used to assign per-animal crops back to persistent mouse IDs.
5. **Ignore:** Files ending in `.im3`, `.vol`, `.raw`.

---

## Scan Inventory

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

---

## Per-Scan Mouse Mapping (XIF Annotations)

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
| m54232 ⚠️ | 5, 6, 7, 8 | KO | Disease | 3h | CT-Hi ⚠️, NaF |
| m54233 | 9, 10, 11, 12 | KO | Disease | 3h | CT-Hi, NaF |
| m54234 | 13, 14, 15, 16 | KO | Disease | 3h | CT-Hi, NaF |
| m54244 | 19, 20 | WT | Control | 3h | CT-Hi, NaF |
| m54301 | 17, 18, 19, 20 | KO | Disease | 1h | CT-Lo, NaF |
| m54302 | 17, 18, 19, 20 | KO | Disease | 3h | CT-Hi, NaF |

*Excluded: m54223 (faulty original; superseded by m54223_1), m54232 (invalid — CT-Hi DICOMs are a blank attenuation correction scan with no animals; mean HU −1015; NaF PET present but no usable CT).* ⚠️ m54223_1: NaF PET not detected in DICOM — only CT-Lo found; may need manual investigation. ⚠️ m54225: mixed-timepoint session. ⚠️ m54232: CT-Hi DICOMs confirmed blank (no anatomical content); NaF_KO_05–08 have no CT-Hi crop in the processed dataset for Week 12.

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
| m54267 ⚠️ | 17, 18, 19, 20 | WT | Control | 5h | CT-Hi ⚠️, FDG |
| m54268 | 1, 2, 3, 4 | KO | Disease | 5h | CT-Hi, FDG |
| m54269 | 5, 6, 7, 8 | KO | Disease | 5h | CT-Hi, FDG |
| m54270 | 9, 10, 11, 12 | KO | Disease | 5h | CT-Hi, FDG |
| m54271 | 13, 14, 15, 16 | KO | Disease | 5h | CT-Hi, FDG |
| m54272 | 17, 18, 19, 20 | KO | Disease | 5h | CT-Hi, FDG |

*No re-scans in Week 12 FDG. Excluded: m54267 (invalid — CT-Hi DICOMs are a blank attenuation correction scan with no animals; mean HU −1026; FDG PET present but no usable CT). FDG_WT_17–20 therefore have no CT-Hi crop in the processed dataset for Week 12.*

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

---

## Per-Mouse Longitudinal Attendance

Mouse numbers are persistent longitudinal IDs within each cohort. WT = Control (C57BL/6, regular chow). KO = Disease (Apoe−/−, high-fat diet). NaF and FDG cohorts are entirely different physical animals.

#### NaF Cohort

| Mouse ID(s) | Week 12 | Week 15 | Week 18 | Week 20 | Notes |
|-------------|---------|---------|---------|---------|-------|
| NaF_WT_1–12 | ✓ | ✓ | ✓ | ✓ | Full longitudinal |
| NaF_WT_13–15 | ✓ | ✓ | ✓ | ✗ | Sacrificed after Week 18 |
| NaF_WT_16–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| NaF_WT_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |
| NaF_KO_1–4, 9–12 | ✓ | ✓ | ✓* | ✓ | *KO 1 and KO 4 absent at Week 18 3h scan |
| NaF_KO_5–8 | ✓† | ✓ | ✓* | ✓ | †Week 12 NaF PET present but CT-Hi blank (m54232 invalid); no CT-Hi crop in processed dataset |
| NaF_KO_13–15 | ✓ | ✓ | ✓ | ✗ | Sacrificed after Week 18 |
| NaF_KO_16–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| NaF_KO_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |

#### FDG Cohort

| Mouse ID(s) | Week 12 | Week 15 | Week 18 | Week 20 | Notes |
|-------------|---------|---------|---------|---------|-------|
| FDG_WT_1–12 | ✓ | ✓ | ✓ | ✓ | Full longitudinal |
| FDG_WT_13–16 | ✓ | ✓ | ✓ | ✗ | Sacrificed/died after Week 18 |
| FDG_WT_17–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| FDG_WT_19–20 | ✓† | ✗ | ✗ | ✗ | Sacrificed after Week 12; †Week 12 FDG PET present but CT-Hi blank (m54267 invalid); no CT-Hi crop in processed dataset |
| FDG_KO_1–9, 11–12 | ✓ | ✓ | ✓ | ✓ (3h+5h) | Full longitudinal |
| FDG_KO_10 | ✓ | ✓ | ✓ | ✓ (3h only) | Present at Week 20 3h; absent at 5h |
| FDG_KO_13–15 | ✓ | ✓ | ✓ | ✗ | Sacrificed after Week 18 |
| FDG_KO_16–18 | ✓ | ✓ | ✗ | ✗ | Sacrificed after Week 15 |
| FDG_KO_19–20 | ✓ | ✗ | ✗ | ✗ | Sacrificed after Week 12 |
