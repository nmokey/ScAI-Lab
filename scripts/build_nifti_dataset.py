"""
build_nifti_dataset.py — Convert raw DICOM sessions to a per-mouse NIfTI dataset.

Reads manifest.csv (session-level ground truth), applies inclusion rules and _1/_2 consolidation,
converts DICOM → session NIfTI, segments individual animals from the combined volume, crops per-mouse
NIfTIs, and writes mouse_manifest.csv as the ML-facing index.

Output structure:
  {nifti_output_dir}/
    sessions/week_{N}/{Tracer}/{base_scan_id}/ct_hi.nii.gz
    sessions/week_{N}/{Tracer}/{base_scan_id}/pet_{naf|fdg}.nii.gz
    mice/{cohort}_{genotype}_{num:02d}/week_{N}/ct_hi.nii.gz
    mice/{cohort}_{genotype}_{num:02d}/week_{N}/pet_{naf|fdg}.nii.gz
    mouse_manifest.csv

Usage:
  python scripts/build_nifti_dataset.py [--dry-run] [--stage {1,2,3,all}]
  python scripts/build_nifti_dataset.py --stage 1 --dry-run   # preview session conversion
  python scripts/build_nifti_dataset.py --stage all           # full pipeline

Reads paths from config.yaml. NEVER modifies original DICOM data.
"""

import os
import csv
import glob
import math
import argparse
import yaml
import numpy as np
import SimpleITK as sitk
import pydicom


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            "Copy config.yaml.example to config.yaml and fill in your paths."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_config():
    cfg = load_config()
    return {
        "data_root": cfg["paths"]["data_root"],
        "nifti_output_dir": cfg["paths"]["output_dir"],
        "pet_target": cfg["file_size_heuristics"]["pet_kb"] * 1024,
        "ct_hi_target": cfg["file_size_heuristics"]["ct_hi_res_kb"] * 1024,
        "ct_lo_target": cfg["file_size_heuristics"]["ct_lo_res_kb"] * 1024,
        "tol": cfg["file_size_heuristics"]["tolerance"],
    }


# ---------------------------------------------------------------------------
# Manifest loading & consolidation
# ---------------------------------------------------------------------------

MANIFEST_PATH = "manifest.csv"
IGNORE_EXTENSIONS = {".im3", ".vol", ".raw"}


def load_manifest(manifest_path=MANIFEST_PATH):
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["week"] = int(row["week"])
            row["n_mice"] = int(row["n_mice"])
            row["has_ct_hi"] = row["has_ct_hi"] == "True"
            row["has_ct_lo"] = row["has_ct_lo"] == "True"
            row["has_pet"] = row["has_pet"] == "True"
            row["is_valid"] = row["is_valid"] == "True"
            row["mouse_nums"] = [int(x) for x in row["mouse_nums"].split(";")]
            rows.append(row)
    return rows


def get_base_id(scan_id):
    """Strip _1 / _2 suffix: 'm54400_2' → 'm54400', 'm54223_1' → 'm54223'."""
    parts = scan_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return scan_id


def get_suffix_num(scan_id):
    """Return the numeric suffix (0 = base, 1 = _1, 2 = _2, ...)."""
    parts = scan_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 0


def consolidate_sessions(rows):
    """
    Group all versions of each base session and apply per-modality merge:
      - CT-Hi: take from the latest version (highest suffix) that has has_ct_hi=True
      - PET:   take from the latest version that has has_pet=True

    Returns a list of ConsolidatedSession dicts, one per unique base scan ID,
    only for sessions that have CT-Hi somewhere across their versions.
    """
    from collections import defaultdict

    # Group by (week, tracer, base_id)
    groups = defaultdict(list)
    for row in rows:
        base = get_base_id(row["scan_id"])
        key = (row["week"], row["tracer"], base)
        groups[key].append(row)

    consolidated = []
    for (week, tracer, base_id), versions in groups.items():
        # Sort versions by suffix number ascending (base=0, _1=1, _2=2)
        versions_sorted = sorted(versions, key=lambda r: get_suffix_num(r["scan_id"]))

        # Find best CT-Hi source (highest suffix with CT-Hi)
        ct_hi_source = None
        for v in reversed(versions_sorted):
            if v["has_ct_hi"]:
                ct_hi_source = v
                break

        if ct_hi_source is None:
            continue  # No CT-Hi in any version → exclude

        # Find best PET source (highest suffix with PET)
        pet_source = None
        for v in reversed(versions_sorted):
            if v["has_pet"]:
                pet_source = v
                break

        # Use metadata (mice, genotype, etc.) from the CT-Hi source version
        meta = ct_hi_source
        consolidated.append({
            "week": week,
            "tracer": tracer,
            "base_id": base_id,
            "genotype": meta["genotype"],
            "group": meta["group"],
            "timepoint_h": meta["timepoint_h"],
            "mouse_nums": meta["mouse_nums"],
            "n_mice": meta["n_mice"],
            "ct_hi_scan_id": ct_hi_source["scan_id"],
            "ct_hi_dicom_rel": ct_hi_source["dicom_rel_path"],
            "pet_scan_id": pet_source["scan_id"] if pet_source else None,
            "pet_dicom_rel": pet_source["dicom_rel_path"] if pet_source else None,
            "has_pet": pet_source is not None,
            "notes": meta.get("notes", ""),
        })

    return consolidated


# ---------------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------------

def is_size_match(filepath, target, tol):
    try:
        return abs(os.path.getsize(filepath) - target) <= tol * target
    except OSError:
        return False


def find_dicom_dir(data_root, dicom_rel_path, target_size=None, tol=0.05):
    """
    Locate the actual DICOM subfolder for a session.
    dicom_rel_path from manifest is e.g. 'Week 12/NaF/m54215'.
    Inside that folder, look for a 'dicom_*' subfolder (or fallback).

    If target_size is given, prefer the candidate folder that contains at least
    one file matching that size (within tol). This handles sessions like m54225
    where CT and PET are in separate subfolders with non-standard names.
    """
    session_folder = os.path.join(data_root, dicom_rel_path)
    if not os.path.isdir(session_folder):
        return None
    candidates = [
        d for d in os.listdir(session_folder)
        if d.lower().startswith("dicom") and os.path.isdir(os.path.join(session_folder, d))
    ]
    if not candidates:
        return None

    if target_size is not None:
        def has_target(folder_name):
            folder = os.path.join(session_folder, folder_name)
            for f in os.listdir(folder):
                fp = os.path.join(folder, f)
                if os.path.isfile(fp) and abs(os.path.getsize(fp) - target_size) <= tol * target_size:
                    return True
            return False
        matching = [d for d in candidates if has_target(d)]
        if matching:
            matching.sort()
            return os.path.join(session_folder, matching[0])

    # Fallback: exact match, then _1 suffix, then alphabetical
    scan_id = os.path.basename(dicom_rel_path)
    exact = f"dicom_{scan_id}"
    if exact in candidates:
        return os.path.join(session_folder, exact)
    candidates.sort(key=lambda d: (not d.endswith("_1"), d))
    return os.path.join(session_folder, candidates[0])


def collect_dicom_files(dicom_dir):
    if dicom_dir is None or not os.path.isdir(dicom_dir):
        return []
    all_files = glob.glob(os.path.join(dicom_dir, "**", "*"), recursive=True)
    return [
        f for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() not in IGNORE_EXTENSIONS
    ]


def sort_slices_by_z(file_list):
    def get_z(fp):
        try:
            dcm = pydicom.dcmread(fp, stop_before_pixels=True)
            return float(dcm.ImagePositionPatient[2])
        except Exception:
            return None

    tagged = [(get_z(f), f) for f in file_list]
    valid = [(z, f) for z, f in tagged if z is not None]
    excluded = len(tagged) - len(valid)
    if excluded:
        print(f"    [!] Excluded {excluded} slice(s) with unreadable Z-position.")
    return [f for _, f in sorted(valid, key=lambda x: x[0])]


def read_nifti_from_dicoms(file_list, label):
    if not file_list:
        return None
    print(f"    ... Reading {len(file_list)} slices for {label} ...")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_list)
    try:
        image = reader.Execute()
        spacing = image.GetSpacing()
        print(f"    [i] {label} spacing: {spacing}")
        return image
    except Exception as e:
        print(f"    [!] SITK error reading {label}: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 1 — DICOM → Session NIfTI
# ---------------------------------------------------------------------------

def stage1_convert_session(session, data_root, output_root, cfg, dry_run=False):
    """
    Convert one consolidated session to NIfTI.
    Returns a dict with paths to the written files (or None if dry_run).
    """
    week_str = f"week_{session['week']}"
    tracer = session["tracer"]
    base_id = session["base_id"]
    session_dir = os.path.join(output_root, "sessions", week_str, tracer, base_id)

    ct_path = os.path.join(session_dir, "ct_hi.nii.gz")
    pet_label = f"pet_{tracer.lower()}"
    pet_path = os.path.join(session_dir, f"{pet_label}.nii.gz") if session["has_pet"] else None

    # Skip if already converted
    if not dry_run and os.path.exists(ct_path):
        already_done = not session["has_pet"] or (pet_path and os.path.exists(pet_path))
        if already_done:
            print(f"  [skip] {base_id} — already converted.")
            return {"ct_hi": ct_path, "pet": pet_path, "session_dir": session_dir}

    print(f"\n--> Stage 1: {base_id} (wk{session['week']} {tracer})")
    print(f"    CT-Hi from: {session['ct_hi_scan_id']}")
    if session["has_pet"]:
        print(f"    PET from:   {session['pet_scan_id']}")
        if session["ct_hi_scan_id"] != session["pet_scan_id"]:
            print(f"    [!] MERGE: CT and PET from different scan versions.")

    if dry_run:
        print(f"    [dry-run] would write to {session_dir}")
        return None

    # Collect DICOM slices for CT-Hi
    ct_hi_target = cfg["ct_hi_target"]
    ct_hi_tol = cfg["tol"]
    ct_dicom_dir = find_dicom_dir(data_root, session["ct_hi_dicom_rel"], target_size=ct_hi_target, tol=ct_hi_tol)
    if ct_dicom_dir is None:
        print(f"    [!] CT DICOM dir not found: {session['ct_hi_dicom_rel']}")
        return None
    all_ct_files = collect_dicom_files(ct_dicom_dir)
    ct_hi_files = sort_slices_by_z([
        f for f in all_ct_files
        if is_size_match(f, ct_hi_target, ct_hi_tol)
    ])

    ct_img = read_nifti_from_dicoms(ct_hi_files, "CT-Hi")
    if ct_img is None:
        print(f"    [!] Failed to build CT-Hi volume for {base_id}. Skipping session.")
        return None

    # Collect DICOM slices for PET (may be a different folder)
    pet_img = None
    if session["has_pet"]:
        pet_target = cfg["pet_target"]
        pet_dicom_dir = find_dicom_dir(data_root, session["pet_dicom_rel"], target_size=pet_target, tol=cfg["tol"])
        if pet_dicom_dir is None:
            print(f"    [!] PET DICOM dir not found: {session['pet_dicom_rel']}")
        else:
            all_pet_files = collect_dicom_files(pet_dicom_dir)
            pet_files = sort_slices_by_z([
                f for f in all_pet_files
                if is_size_match(f, pet_target, cfg["tol"])
            ])
            pet_img = read_nifti_from_dicoms(pet_files, f"PET-{tracer}")

    # Write outputs
    os.makedirs(session_dir, exist_ok=True)
    sitk.WriteImage(ct_img, ct_path)
    print(f"    [+] Wrote {ct_path}")

    if pet_img is not None:
        sitk.WriteImage(pet_img, pet_path)
        print(f"    [+] Wrote {pet_path}")
    elif session["has_pet"]:
        print(f"    [!] PET not written (failed to build volume).")
        pet_path = None

    return {"ct_hi": ct_path, "pet": pet_path, "session_dir": session_dir}


# ---------------------------------------------------------------------------
# Stage 2 — Segment individual animals in the combined CT volume
# ---------------------------------------------------------------------------

def segment_animals(ct_sitk, n_mice_expected, session_id):
    """
    Quadrant-based segmentation: locate individual mice in a combined multi-animal CT.

    Strategy:
      1. Downsample to ~1 mm/voxel and threshold at ≥300 HU (bone only).
      2. Compute the physical centroid of ALL bone mass — this is the centre of the
         scanner grid, midway between the rows and columns of mice.
      3. Divide bone voxels into quadrants relative to that centre using physical
         coordinates (avoids voxel-direction ambiguity).
      4. Each populated quadrant = one mouse. Position is assigned unambiguously by
         which quadrant the mouse occupies — no sorting required.

    Position convention (matches manifest mouse_nums positional order):
        pos 1 = lower-left   pos 2 = lower-right
        pos 3 = upper-left   pos 4 = upper-right

    Scanner physical coordinate conventions (empirically verified):
        higher physical X = visually left
        lower  physical Y = visually lower

    n_mice handling:
        All cases use the full 2×2 quadrant split. Empty quadrants are skipped
        automatically, so 2- and 3-mouse sessions work without special casing.

    Returns a list of bbox dicts (one per animal, in position order) or None on failure.
    """
    ct_array = sitk.GetArrayFromImage(ct_sitk)  # (Z, Y, X)
    spacing  = ct_sitk.GetSpacing()              # (x_sp, y_sp, z_sp) mm

    # --- Downsample to ~1 mm/voxel ---
    SEG_MM = 1.0
    fx = max(1, int(round(SEG_MM / spacing[0])))
    fy = max(1, int(round(SEG_MM / spacing[1])))
    fz = max(1, int(round(SEG_MM / spacing[2])))
    bone_ds = (ct_array[::fz, ::fy, ::fx] >= 300)
    dsp = (spacing[0]*fx, spacing[1]*fy, spacing[2]*fz)
    print(f"    [i] Seg: downsampled {ct_array.shape} → {bone_ds.shape} "
          f"(factors {fz},{fy},{fx}; ~{dsp[0]:.1f} mm/vox)")

    bone_zz, bone_yy, bone_xx = np.where(bone_ds)
    if len(bone_xx) == 0:
        print(f"    [!] No bone tissue found in {session_id}")
        return None

    # --- Compute physical X and Y of every bone voxel (vectorised) ---
    # TransformIndexToPhysicalPoint([1,0,0]) - origin gives the physical-space
    # step per unit image-X voxel (including spacing).  Same for Y and Z.
    # This correctly handles any direction cosine signs without assumptions.
    o  = np.array(ct_sitk.GetOrigin())
    dp = np.array([
        ct_sitk.TransformIndexToPhysicalPoint([1, 0, 0]),
        ct_sitk.TransformIndexToPhysicalPoint([0, 1, 0]),
        ct_sitk.TransformIndexToPhysicalPoint([0, 0, 1]),
    ]) - o  # dp[i] = physical step per unit voxel on axis i

    bx = bone_xx.astype(np.float32)
    by = bone_yy.astype(np.float32)
    bz = bone_zz.astype(np.float32)

    # Downsampled voxel indices → physical (multiply by factor because ds voxel i = orig voxel i*f)
    phys_x = o[0] + dp[0,0]*fx*bx + dp[1,0]*fy*by + dp[2,0]*fz*bz
    phys_y = o[1] + dp[0,1]*fx*bx + dp[1,1]*fy*by + dp[2,1]*fz*bz

    # --- FOV geometric centre = dividing line between quadrants ---
    # Using the image centre rather than the bone centroid so that sessions with
    # fewer than 4 mice (where bone centroid is biased toward occupied positions)
    # still get the correct quadrant boundaries.  The scanner bed has fixed grid
    # positions, so the FOV centre is always between the rows and columns.
    size = ct_sitk.GetSize()  # (X, Y, Z) voxel counts
    fov_centre = ct_sitk.TransformIndexToPhysicalPoint(
        [size[0] // 2, size[1] // 2, size[2] // 2])
    cx_phys = fov_centre[0]
    cy_phys = fov_centre[1]
    print(f"    [i] FOV centre (quadrant divider): ({cx_phys:.1f}, {cy_phys:.1f}) mm  "
          f"[higher X = left, lower Y = lower]")

    # --- Build quadrant masks in physical space ---
    # higher physical X = left, lower physical Y = lower
    is_left  = phys_x > cx_phys
    is_right = phys_x < cx_phys
    is_lower = phys_y < cy_phys
    is_upper = phys_y > cy_phys

    # Full 2×2 quadrant split for all n_mice values.
    # Empty quadrants (no bone voxels) are skipped below — 2/3-mouse sessions work
    # correctly because their unoccupied positions naturally have no bone.
    quadrant_masks = [
        is_lower & is_left,   # pos 1: lower-left
        is_lower & is_right,  # pos 2: lower-right
        is_upper & is_left,   # pos 3: upper-left
        is_upper & is_right,  # pos 4: upper-right
    ]

    # --- One bbox per quadrant ---
    vox_vol_mm3 = dsp[0] * dsp[1] * dsp[2]
    min_voxels  = int(1000.0 / vox_vol_mm3)   # 1 mL minimum

    bboxes = []
    for pos_idx, mask in enumerate(quadrant_masks):
        if not mask.any():
            print(f"    [!] pos {pos_idx+1}: no bone voxels in quadrant — skipping.")
            continue
        q_xx, q_yy, q_zz = bone_xx[mask], bone_yy[mask], bone_zz[mask]
        if len(q_xx) < min_voxels:
            print(f"    [!] pos {pos_idx+1}: only {len(q_xx)} voxels (<1 mL) — skipping.")
            continue

        # Bone extent → full-res voxel indices
        x_lo, x_hi = int(q_xx.min())*fx, int(q_xx.max())*fx
        y_lo, y_hi = int(q_yy.min())*fy, int(q_yy.max())*fy
        z_lo, z_hi = int(q_zz.min())*fz, int(q_zz.max())*fz

        cx = int(round(float(q_xx.mean())*fx))
        cy = int(round(float(q_yy.mean())*fy))
        cz = int(round(float(q_zz.mean())*fz))

        # For the lower two mice (pos 1 & 2), bias the centroid slightly upward
        # toward the FOV centre so symmetric padding gives more room on the
        # inward-facing (top) side where the scanner bed centre is.
        CENTROID_BIAS_MM = 5.0
        centroid_phys_raw = ct_sitk.TransformIndexToPhysicalPoint([cx, cy, cz])
        if pos_idx < 2:  # lower row only
            dy = cy_phys - centroid_phys_raw[1]
            dist_y = abs(dy)
            if dist_y > 0:
                scale = min(CENTROID_BIAS_MM, dist_y) / dist_y
                centroid_phys = (
                    centroid_phys_raw[0],
                    centroid_phys_raw[1] + dy * scale,
                    centroid_phys_raw[2],
                )
            else:
                centroid_phys = centroid_phys_raw
        else:
            centroid_phys = centroid_phys_raw
        bone_min = ct_sitk.TransformIndexToPhysicalPoint([x_lo, y_lo, z_lo])
        bone_max = ct_sitk.TransformIndexToPhysicalPoint([x_hi, y_hi, z_hi])
        print(f"    [i] pos {pos_idx+1}: bone "
              f"{abs(bone_max[0]-bone_min[0]):.0f}×"
              f"{abs(bone_max[1]-bone_min[1]):.0f}×"
              f"{abs(bone_max[2]-bone_min[2]):.0f} mm  "
              f"centroid=({centroid_phys[0]:.1f}, {centroid_phys[1]:.1f})")

        bboxes.append({
            "vox_min":    (z_lo, y_lo, x_lo),
            "vox_max":    (z_hi, y_hi, x_hi),
            "phys_min":   centroid_phys,   # point bbox → uniform crop via padding
            "phys_max":   centroid_phys,
            "centroid_x": centroid_phys[0],
            "centroid_y": centroid_phys[1],
            "position":   pos_idx + 1,
        })

    if not bboxes:
        print(f"    [!] No valid quadrants found for {session_id}")
        return None

    n_found = len(bboxes)
    if n_found < n_mice_expected:
        print(f"    [!] {n_found} quadrants populated (expected {n_mice_expected}) — proceeding with {n_found}.")
    else:
        print(f"    [i] Quadrant segmentation: {n_found}/{n_mice_expected} mice found ✓")

    return bboxes


def crop_to_physical_bbox(img_sitk, phys_min, phys_max, padding_xy_mm=35.0, padding_z_mm=70.0):
    """
    Crop img_sitk to the physical bounding box [phys_min, phys_max] (mm, X/Y/Z order).

    Uses SimpleITK's coordinate transform so this works correctly on any image regardless
    of its voxel spacing or origin — in particular, PET and CT can have different grids
    and both are correctly cropped from the same physical bbox derived from CT segmentation.

    padding_xy_mm: padding in the transaxial (X/Y) plane — kept tight to avoid including
                   adjacent mice in a multi-animal scanner bed.
    padding_z_mm:  padding along the longitudinal (Z) axis — generous to capture skull/tail.
    """
    size = img_sitk.GetSize()      # (X, Y, Z) voxel counts
    spacing = img_sitk.GetSpacing()  # (X, Y, Z) mm/voxel

    # Map physical corners into this image's voxel index space.
    # TransformPhysicalPointToIndex handles origin + direction matrix.
    idx_min = img_sitk.TransformPhysicalPointToIndex(phys_min)  # (X, Y, Z) ints
    idx_max = img_sitk.TransformPhysicalPointToIndex(phys_max)  # (X, Y, Z) ints

    # Per-axis padding in voxels (ceil so we never under-pad)
    pad_x = int(math.ceil(padding_xy_mm / spacing[0]))
    pad_y = int(math.ceil(padding_xy_mm / spacing[1]))
    pad_z = int(math.ceil(padding_z_mm  / spacing[2]))

    # Sort lo/hi per axis: flipped coordinate axes (negative direction cosines) can
    # make idx_min > idx_max for that axis.
    x_start = max(0,       min(idx_min[0], idx_max[0]) - pad_x)
    y_start = max(0,       min(idx_min[1], idx_max[1]) - pad_y)
    z_start = max(0,       min(idx_min[2], idx_max[2]) - pad_z)
    x_end   = min(size[0], max(idx_min[0], idx_max[0]) + pad_x + 1)
    y_end   = min(size[1], max(idx_min[1], idx_max[1]) + pad_y + 1)
    z_end   = min(size[2], max(idx_min[2], idx_max[2]) + pad_z + 1)

    # RegionOfInterestImageFilter takes (index, size) in (X, Y, Z) order
    extract = sitk.RegionOfInterestImageFilter()
    extract.SetIndex([x_start, y_start, z_start])
    extract.SetSize([x_end - x_start, y_end - y_start, z_end - z_start])
    return extract.Execute(img_sitk)


# ---------------------------------------------------------------------------
# Stage 3 — Crop and write per-mouse NIfTIs
# ---------------------------------------------------------------------------

def stage3_crop_and_write(session, nifti_paths, bboxes, output_root, dry_run=False):
    """
    Given a session's session NIfTIs and per-animal bounding boxes, crop and write
    per-mouse NIfTIs. Returns a list of mouse_manifest rows.
    """
    manifest_rows = []

    ct_sitk = sitk.ReadImage(nifti_paths["ct_hi"]) if not dry_run else None
    pet_sitk = None
    if nifti_paths.get("pet") and os.path.exists(nifti_paths["pet"]):
        pet_sitk = sitk.ReadImage(nifti_paths["pet"]) if not dry_run else None

    tracer_label = session["tracer"].lower()  # "naf" or "fdg"
    week = session["week"]
    genotype = session["genotype"]
    # Determine cohort from tracer
    cohort = session["tracer"]  # "NaF" or "FDG"

    # Map bbox position → mouse_num from manifest mouse_nums
    # position 1 = bboxes[0], 2 = bboxes[1], etc.
    mouse_nums = session["mouse_nums"]

    if len(bboxes) != len(mouse_nums):
        print(f"    [!] bbox count ({len(bboxes)}) ≠ n_mice ({len(mouse_nums)}) for {session['base_id']}")
        print(f"        Proceeding with min({len(bboxes)}, {len(mouse_nums)}) crops.")

    n_crops = min(len(bboxes), len(mouse_nums))

    for i in range(n_crops):
        bbox = bboxes[i]
        mouse_num = mouse_nums[i]
        mouse_id = f"{cohort}_{genotype}_{mouse_num:02d}"
        week_dir = os.path.join(output_root, "mice", mouse_id, f"week_{week}")
        ct_out = os.path.join(week_dir, "ct_hi.nii.gz")
        pet_out = os.path.join(week_dir, f"pet_{tracer_label}.nii.gz")

        if dry_run:
            print(f"    [dry-run] {mouse_id}/week_{week}: crop pos={bbox['position']} vox={bbox['vox_min']}→{bbox['vox_max']}")
            row = _build_manifest_row(session, mouse_num, mouse_id, week, ct_out, pet_out if pet_sitk else None, bbox)
            manifest_rows.append(row)
            continue

        os.makedirs(week_dir, exist_ok=True)

        ct_crop = sitk.DICOMOrient(
            crop_to_physical_bbox(ct_sitk, bbox["phys_min"], bbox["phys_max"], padding_xy_mm=20.0, padding_z_mm=70.0),
            "RAS")
        sitk.WriteImage(ct_crop, ct_out)
        print(f"    [+] {mouse_id}/week_{week}/ct_hi.nii.gz")

        actual_pet_out = None
        if pet_sitk is not None:
            pet_crop = sitk.DICOMOrient(
                crop_to_physical_bbox(pet_sitk, bbox["phys_min"], bbox["phys_max"], padding_xy_mm=20.0, padding_z_mm=70.0),
                "RAS")
            sitk.WriteImage(pet_crop, pet_out)
            print(f"    [+] {mouse_id}/week_{week}/pet_{tracer_label}.nii.gz")
            actual_pet_out = pet_out

        row = _build_manifest_row(session, mouse_num, mouse_id, week, ct_out, actual_pet_out, bbox)
        manifest_rows.append(row)

    return manifest_rows


def _build_manifest_row(session, mouse_num, mouse_id, week, ct_out, pet_out, bbox):
    parts = mouse_id.split("_")
    cohort = parts[0]
    genotype = parts[1]
    return {
        "mouse_id": mouse_id,
        "cohort": cohort,
        "genotype": genotype,
        "group": session["group"],
        "mouse_num": mouse_num,
        "week": week,
        "session_id": session["base_id"],
        "timepoint_h": session["timepoint_h"],
        "ct_hi_nifti": ct_out,
        "pet_nifti": pet_out or "",
        "ct_source_scan": session["ct_hi_scan_id"],
        "pet_source_scan": session.get("pet_scan_id") or "",
        "crop_position": bbox["position"],
        "n_mice_in_session": session["n_mice"],
        "notes": session.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Mouse manifest writer
# ---------------------------------------------------------------------------

MOUSE_MANIFEST_FIELDS = [
    "mouse_id", "cohort", "genotype", "group", "mouse_num", "week",
    "session_id", "timepoint_h", "ct_hi_nifti", "pet_nifti",
    "ct_source_scan", "pet_source_scan", "crop_position", "n_mice_in_session", "notes",
]


def write_mouse_manifest(rows, output_root, session_filter=None):
    path = os.path.join(output_root, "mouse_manifest.csv")
    # When a session filter is active, merge: keep existing rows for sessions
    # that were NOT re-processed, then append/replace rows for filtered sessions.
    if session_filter and os.path.exists(path):
        with open(path, newline="") as f:
            existing = list(csv.DictReader(f))
        filtered_session_ids = {r["session_id"] for r in rows}
        kept = [r for r in existing if r["session_id"] not in filtered_session_ids]
        rows = kept + rows
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MOUSE_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[+] Wrote mouse_manifest.csv ({len(rows)} rows) → {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(stage_filter, dry_run, cfg, output_root, session_filter=None):
    print(f"\n{'='*60}")
    print(f"  build_nifti_dataset.py")
    print(f"  data_root:       {cfg['data_root']}")
    print(f"  nifti_output_dir: {output_root}")
    print(f"  stage:           {stage_filter}  |  dry_run: {dry_run}")
    if session_filter:
        print(f"  session filter:  {sorted(session_filter)}")
    print(f"{'='*60}\n")

    rows = load_manifest()
    sessions = consolidate_sessions(rows)
    if session_filter:
        sessions = [s for s in sessions if s["base_id"] in session_filter]
        unknown = session_filter - {s["base_id"] for s in sessions}
        if unknown:
            print(f"[!] Unknown session IDs (not in manifest): {sorted(unknown)}")
    print(f"Manifest: {len(rows)} session rows → {len(sessions)} consolidated CT-Hi sessions after filtering.\n")

    mouse_manifest_rows = []

    for session in sessions:
        print(f"\n[Session] wk{session['week']} {session['tracer']} {session['base_id']} "
              f"| {session['genotype']} mice {session['mouse_nums']}")

        # --- Stage 1 ---
        nifti_paths = None
        if stage_filter in (1, "all"):
            nifti_paths = stage1_convert_session(session, cfg["data_root"], output_root, cfg, dry_run)
            if nifti_paths is None and not dry_run:
                print(f"  [!] Stage 1 failed for {session['base_id']} — skipping stages 2+3.")
                continue

        # For stages 2+3, use existing session NIfTIs if stage 1 was skipped
        if nifti_paths is None:
            week_str = f"week_{session['week']}"
            session_dir = os.path.join(output_root, "sessions", week_str, session["tracer"], session["base_id"])
            ct_path = os.path.join(session_dir, "ct_hi.nii.gz")
            pet_label = f"pet_{session['tracer'].lower()}"
            pet_path = os.path.join(session_dir, f"{pet_label}.nii.gz")
            nifti_paths = {
                "ct_hi": ct_path,
                "pet": pet_path if os.path.exists(pet_path) else None,
                "session_dir": session_dir,
            }

        # --- Stage 2 ---
        # Always run segmentation when stage 3 is needed: bboxes are in-memory only.
        bboxes = None
        if stage_filter in (2, 3, "all"):
            if dry_run:
                print(f"  [dry-run] Stage 2: would segment {session['n_mice']} animals from {session['base_id']}")
                # Create dummy bboxes for dry-run manifest preview
                bboxes = [{"vox_min": (0,0,0), "vox_max": (10,10,10),
                           "phys_min": (0.0, 0.0, 0.0), "phys_max": (10.0, 10.0, 10.0),
                           "centroid_x": float(i), "centroid_y": float(i % 2),
                           "position": i+1} for i in range(session["n_mice"])]
            else:
                if not os.path.exists(nifti_paths["ct_hi"]):
                    print(f"  [!] CT NIfTI not found: {nifti_paths['ct_hi']} — skipping stages 2+3.")
                    continue
                ct_sitk = sitk.ReadImage(nifti_paths["ct_hi"])
                bboxes = segment_animals(ct_sitk, session["n_mice"], session["base_id"])
                if bboxes is None:
                    print(f"  [!] Segmentation failed for {session['base_id']} — skipping stage 3.")
                    continue

        # --- Stage 3 ---
        if stage_filter in (3, "all") and bboxes is not None:
            rows_out = stage3_crop_and_write(session, nifti_paths, bboxes, output_root, dry_run)
            mouse_manifest_rows.extend(rows_out)

    if mouse_manifest_rows and stage_filter in (3, "all"):
        if dry_run:
            print(f"\n[dry-run] Would write mouse_manifest.csv with {len(mouse_manifest_rows)} rows.")
        else:
            write_mouse_manifest(mouse_manifest_rows, output_root, session_filter=session_filter)


def main():
    parser = argparse.ArgumentParser(description="Build per-mouse NIfTI dataset from DICOM.")
    parser.add_argument(
        "--stage", default="all",
        help="Which stage(s) to run: 1 (DICOM→NIfTI), 2 (segment), 3 (crop), all (default: all)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without writing any files.")
    parser.add_argument("--session", nargs="+", metavar="SCAN_ID",
                        help="Only process these session IDs (e.g. --session m54226 m54399).")
    args = parser.parse_args()

    # Parse stage argument
    stage = args.stage
    if stage not in ("1", "2", "3", "all"):
        print(f"[!] Invalid --stage '{stage}'. Use 1, 2, 3, or all.")
        return
    stage_filter = int(stage) if stage.isdigit() else "all"

    cfg = init_config()
    output_root = cfg["nifti_output_dir"]

    if not args.dry_run:
        os.makedirs(output_root, exist_ok=True)

    session_filter = set(args.session) if args.session else None
    run_pipeline(stage_filter, args.dry_run, cfg, output_root, session_filter)
    print("\nDone.")


if __name__ == "__main__":
    main()
