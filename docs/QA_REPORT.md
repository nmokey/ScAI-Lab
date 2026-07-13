# Dataset QA Report

**Date:** 2026-03-04
**Method:** Ground-truth verification against `/data1/Dicom Data/` and `/data1/Amgen SUV Data/`
**Scope:** All 147 rows of `manifest.csv`, all DICOM scan folders, all Amgen XIF files

---

## 1. Folder Inventory (DICOM directories vs. manifest)

**Result: PASS** — Every scan folder listed in the manifest (valid and superseded) exists in the DICOM directory. No manifest entries point to missing folders, and no unexpected scan folders exist that are absent from the manifest.

Specific findings:
- All 22 Week 12 NaF folders present (`m54215`–`m54302`), including `m54223` and `m54223_1`.
- All 20 Week 12 FDG folders present (`m54253`–`m54272`).
- All 27 Week 15 NaF folders present (including all `_1`/`_2` re-scan variants).
- All 20 Week 15 FDG folders present.
- All 16 Week 18 NaF folders present.
- All 17 Week 18 FDG folders present (including `m54688` and `m54688_1`).
- All 12 Week 20 NaF folders present.
- All 13 Week 20 FDG folders present. `m54783` is absent (only `m54783_1` exists) — consistent with manifest note.
- `m54784_1` exists as a folder but has no `dicom_*` subfolder inside — confirmed DICOM-less, consistent with manifest note.

---

## 2. `dicom_rel_path` Field — CRITICAL ERROR

**Result: FAIL — All 147 rows are wrong.**

The manifest's `dicom_rel_path` field uses abbreviated tracer directory names (`NaF`, `FDG`), but the actual on-disk directory names are `18F-NaF` and `18F-FDG`.

| Manifest `dicom_rel_path` | Actual on-disk path |
|---|---|
| `Week 12/NaF/m54215` | `Week 12/18F-NaF/m54215` |
| `Week 12/FDG/m54253` | `Week 12/18F-FDG/m54253` |
| `Week 15/NaF/m54389` | `Week 15/18F-NaF/m54389` |
| ... | ... |

The `build_nifti_dataset.py` pipeline's `find_dicom_dir()` function calls `os.path.join(data_root, dicom_rel_path)` directly, and will therefore fail to find **every single scan folder**. The pipeline will silently skip all 147 sessions and produce an empty dataset.

**Fix:** Replace all `dicom_rel_path` values — change `/NaF/` → `/18F-NaF/` and `/FDG/` → `/18F-FDG/` throughout `manifest.csv`.

---

## 3. Modality Detection (File-Size Heuristic)

**Result: PASS with one discrepancy (m54223) and expected entries for superseded scans.**

The file-size heuristic (PET ≈115 KB ±5%, CT-Hi ≈2814 KB ±5%, CT-Lo ≈705 KB ±5%) was applied to all `dicom_*` and `DICOM/` subfolders across all 147 scan directories. Results match the manifest `modalities` field with one exception:

### 3a. DISCREPANCY: m54223 — NaF PET present but manifest says absent

> **RESOLVED** — investigated and fixed; see [§7 Fix 2](#fix-2-m54223-naf-pet-status-corrected--done). DICOM tag inspection confirmed the NaF PET is genuine; the manifest and per-modality merge were corrected. The discussion below is the original finding, kept for provenance.

| Field | Manifest | Actual (heuristic) |
|---|---|---|
| `has_pet` | `False` | **True** (191 PET-sized files in `dicom_m54223`) |
| `modalities` | `CT-Lo` | **CT-Lo;NaF** |
| `notes` | "no NaF PET detected; superseded by m54223_1" | — |

Meanwhile, the "valid" replacement `m54223_1` has:
- `has_pet = False` (manifest) → confirmed correct: `dicom_m54223_1` contains CT-Lo only, 0 PET files.

**What this means:** The NaF PET data for **KO mice 9–12 at Week 12, 1h** exists in `m54223` but is marked as "superseded" (invalid). The valid re-scan `m54223_1` has no PET. If the pipeline follows Heuristic #1 (per-modality merge), it should pull NaF PET from `m54223` and CT-Lo from `m54223_1` — but the `superseded_by` relationship is coded in the opposite direction from what the data shows. This needs manual investigation to determine if `m54223` was superseded for a reason other than missing PET (e.g., corrupt image data despite correct file count).

**Recommendation:** Inspect `m54223` DICOM slices directly to verify PET image quality before deciding validity.

### 3b. Superseded scans with empty modalities

The following superseded scans have empty `modalities` in the manifest while the heuristic detects `NONE` (equivalent — just a formatting inconsistency):
- `m54400_1`: NONE detected ✓
- `m54407_1`: NONE detected ✓
- `m54408`: NONE detected ✓
- `m54784_1`: No `dicom_*` subfolder (no DICOM data at all) ✓

### 3c. All other modalities: PASS

Every other scan's `has_ct_hi`, `has_ct_lo`, `has_pet`, and `modalities` fields match what the file-size heuristic detects. Key confirmations:
- Week 18 NaF: m54638, m54644, m54645, m54646 correctly show NaF-only (no CT). ✓
- Week 18 FDG: m54685, m54689 correctly show FDG-only (no CT). ✓
- Week 15 NaF: m54403 correctly shows CT-Hi only (no NaF PET). ✓
- Week 15 NaF: m54407_2 correctly shows CT-Hi only (no NaF PET), while m54407 has CT-Hi + NaF. ✓

---

## 4. DICOM Subfolder Structure — Anomalies Noted

All anomalies are benign but should be documented for the pipeline.

| Scan | Anomaly | Impact |
|---|---|---|
| `m54225` | DICOM subfolder is `DICOM/` (capital), not `dicom_m54225/`. Also has `dicom_m54225_1/` (PET-only). | `find_dicom_dir` correctly handles this (case-insensitive prefix match). Prefer `DICOM/` which has complete data (CT-Hi + NaF). |
| `m54244` | Has both `DICOM/` (PET + many other files) and `dicom_m54244/` (CT-Hi + NaF). | Pipeline prefers `dicom_m54244` (exact match) → correct. |
| `m54407` | Has `DICOM/`, `DICOM_01/`, and `dicom_m54407/` — all contain CT-Hi + NaF or subsets. | Pipeline uses `dicom_m54407` (exact match) → correct. |
| `m54272` | Has extra `dicom_m54272_1/` (191 PET files + 3072 ~1 KB auxiliary files). | Pipeline uses `dicom_m54272` (exact match) → correct. |
| `m54408_1` | Has extra `dicom_m54408_1_2/` (191 PET files + 3072 ~1 KB files). | Pipeline uses `dicom_m54408_1` (exact match) → correct. |
| `m54773` | Has empty `dicom_m54773 ct/` (0 files). | Pipeline uses `dicom_m54773` (exact match) → correct. |

---

## 5. XIF Mouse Mapping (Genotype, Mouse Numbers, Timepoints)

**Result: PASS** — All XIF-sourced metadata matches the manifest.

Every Amgen XIF filename was parsed and compared against the manifest's `genotype`, `mouse_nums`, and `timepoint_h` fields. After accounting for two XIF naming conventions (NaF uses `_Nh` suffix, FDG uses space before `Nh`), and the edge cases below, **all 108 XIF files match the manifest exactly**.

### XIF edge cases (all handled correctly in manifest):

| XIF filename | Situation | Manifest handling |
|---|---|---|
| `m54232-m54250_KO 5 6 7 8_3h.xif` | Range notation; DICOM folder is `m54232` | Manifest `scan_id=m54232`, notes explain range notation ✓ |
| `m54225_WT 19 20_1h WT 1 2_3h.xif` | Mixed timepoint session | Manifest `timepoint_h=1+3`, `mouse_nums=19;20;1;2` ✓ |
| `m54688_1_KO 5 6 7 8_5h.xif` | XIF exists for `_1` re-scan, not original | Consistent: m54688 (original, faulty) has no XIF; m54688_1 does ✓ |
| `m54783_1_KO 9 10 11 12_3h.xif` | Only `_1` scan exists (original absent) | Consistent with manifest note ✓ |
| `m54784_WT 1 2 3 4_3h.xif` | XIF for faulty original m54784 (no XIF for m54784_1) | Consistent: m54784_1 is a re-scan but has no modalities ✓ |

### Manifest entries without XIF (expected):
- `m54232`: XIF is filed under range notation `m54232-m54250` — not a true mismatch.
- `m54688`: Faulty original; replaced by m54688_1 which has the XIF — consistent.

### All re-scan variants (`_1`, `_2`):
Re-scans do not have separate XIF files. The mouse IDs from the base scan's XIF are inherited by re-scan variants in the manifest. This is correct per the study design.

---

## 6. Summary Table

| Check | Result | # Issues |
|---|---|---|
| DICOM folder inventory completeness | PASS | 0 |
| `dicom_rel_path` field correctness | FIXED | 0 (was 147) |
| Modality detection (CT-Hi/CT-Lo/PET) | FIXED | 0 (was 1) |
| XIF genotype vs. manifest | PASS | 0 |
| XIF mouse numbers vs. manifest | PASS | 0 |
| XIF timepoints vs. manifest | PASS | 0 |
| Superseded/valid designations | FIXED | 0 (was 1) |

---

## 7. Fixes Applied

### Fix 1 (CRITICAL): Updated `dicom_rel_path` in manifest.csv — DONE

All 147 rows: replaced `/NaF/` → `/18F-NaF/` and `/FDG/` → `/18F-FDG/`. All paths now resolve to actual on-disk directories.

### Fix 2: m54223 NaF PET status corrected — DONE

DICOM tag inspection (`Mod=PT`, series description `f18_static`) confirmed NaF PET is genuinely present in `m54223`. The pattern mirrors m54407: original CT was faulty, CT re-scanned as m54223_1 (CT-Lo only), original NaF PET retained via Heuristic #1.

Updated manifest:
- `m54223`: `has_pet=True`, `modalities=CT-Lo;NaF`, notes updated to "Faulty CT; CT-Lo superseded by m54223_1; NaF PET from this scan is retained and merged during NIfTI build per Heuristic #1"
- `m54223_1`: notes updated to "CT-Lo only in this DICOM folder; NaF PET sourced from m54223 during NIfTI build (per-modality merge per Heuristic #1)"

KO mice 9–12 Week 12 1h NaF PET data is now correctly accounted for.

---

## 8. Observations (No Action Required)

- **`m54407` "faulty CT":** The file-size heuristic detects 1200 CT-Hi files in `m54407` (same count as valid scans). The CT was deemed "faulty" for a reason not detectable by file count (e.g., positioning error, reconstruction artifact). The manifest's NaF PET retention via Heuristic #1 is correctly documented.
- **Week 18 NaF CT-less scans:** m54638, m54644, m54645, m54646 have NaF PET only (no CT). All confirmed by heuristic. These are already flagged in manifest.
- **Week 18 FDG CT-less scans:** m54685, m54689 have FDG PET only. Confirmed. Already flagged.
- **m54784_1 (Week 20 FDG):** No `dicom_*` subfolder exists at all in this folder. The manifest's note ("No modalities detected... investigate manually") is accurate — but the cause is the absence of the DICOM subfolder entirely, not just empty files.
