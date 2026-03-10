"""
get_m3d_embeddings.py

Iterates over all per-mouse CT-Hi NIfTI crops in mouse_manifest.csv, passes
each through BAAI's M3D-CLIP 3D vision encoder, and saves the resulting
embeddings in the standard .npz format consumed by evaluate_embeddings.py.

Installation (one-time):
    pip install transformers scipy nibabel

Model:
    GoodBaiBai88/M3D-CLIP — a 3D ViT (0.2B params) pre-trained on ~120k
    medical image-text pairs across 11 3D modalities via contrastive learning.
    Loaded via AutoModel.from_pretrained(..., trust_remote_code=True).

Preprocessing:
    1. Load NIfTI with nibabel; reorient to RAS+ (axis 2 = axial / Z).
    2. Clip HU to [HU_MIN, HU_MAX] for soft-tissue contrast.
    3. Resample volume to target shape (32 × 256 × 256) using
       scipy.ndimage.zoom with trilinear (order=1) interpolation.
    4. Min-max normalize to [0, 1] — M3D-CLIP's expected input range.
    5. Add channel dim → tensor shape (1, 32, 256, 256).

Embedding:
    model.encode_image(tensor.unsqueeze(0))[:, 0]
    Returns the CLS token: a 768-d float32 vector per volume.

Output:
    <output_dir>/embeddings/m3d/m3d_embeddings.npz
        embeddings  : float32 array of shape (N, 768)
        subject_ids : string array of shape (N,) — e.g. "NaF_WT_03"
        weeks       : string array of shape (N,) — e.g. "Week 12"
        modalities  : string array of shape (N,) — always "CT_HiRes"
        paths       : string array of shape (N,) — absolute path to source NIfTI
"""

import argparse
import os
import csv
import warnings
import yaml

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from transformers import AutoModel, AutoTokenizer


# --- Config ---
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"'{config_path}' not found. "
            "Copy config.yaml.example → config.yaml and fill in your paths."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# --- Manifest ---
def load_ct_rows(manifest_path):
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ct_path = row.get("ct_hi_nifti", "").strip()
            if ct_path:
                rows.append(row)
    print(f"[i] Found {len(rows)} CT-Hi crops in mouse_manifest.csv.")
    return rows


# --- Volume preprocessing ---
def preprocess_volume(nifti_path, target_shape, hu_min, hu_max):
    """
    Load a NIfTI CT volume and return a float32 tensor ready for M3D-CLIP.

    Steps:
      1. Load with nibabel; reorient to RAS+ (array axis 2 = axial / Z).
      2. Clip HU to [hu_min, hu_max].
      3. Resample to target_shape (D, H, W) using trilinear zoom.
      4. Min-max normalize to [0, 1].
      5. Add channel dim → torch.Tensor of shape (1, D, H, W).
    """
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)          # reorient to RAS+
    vol = img.get_fdata(dtype=np.float32)        # (X, Y, Z)

    # Reorder to (Z, Y, X) = (D, H, W) for consistent spatial axes
    vol = vol.transpose(2, 1, 0)                 # (Z, Y, X) → (D, H, W)

    # HU clipping
    vol = np.clip(vol, hu_min, hu_max)

    # Resample to target shape (D, H, W)
    zoom_factors = [t / s for t, s in zip(target_shape, vol.shape)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = scipy.ndimage.zoom(vol, zoom_factors, order=1)   # trilinear

    # Min-max normalize to [0, 1]
    v_min, v_max = vol.min(), vol.max()
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol)

    # Add channel dim → (1, D, H, W)
    tensor = torch.from_numpy(vol[np.newaxis]).float()
    return tensor


# --- Volume embedding ---
def embed_volume(nifti_path, model, device, target_shape, hu_min, hu_max):
    """
    Extract a 768-d CLS embedding for a 3D CT volume using M3D-CLIP.
    """
    tensor = preprocess_volume(nifti_path, target_shape, hu_min, hu_max)
    # model.encode_image expects (B, C, D, H, W)
    tensor = tensor.unsqueeze(0).to(device)       # (1, 1, D, H, W)
    with torch.no_grad():
        emb = model.encode_image(tensor)[:, 0]    # (1, 768) → CLS token
    return emb.squeeze(0).cpu().float().numpy()   # (768,)


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Extract M3D-CLIP embeddings from mouse CT NIfTI crops."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N volumes (useful for testing)."
    )
    args = parser.parse_args()

    # --- Tunable parameters ---
    TARGET_SHAPE = (32, 256, 256)   # (D, H, W) — M3D-CLIP's native input
    HU_MIN       = -160             # soft-tissue window lower bound (HU)
    HU_MAX       =  240             # soft-tissue window upper bound (HU)
    MODEL_ID     = "GoodBaiBai88/M3D-CLIP"
    # ---------------------------

    cfg = load_config()
    output_dir    = cfg["paths"]["output_dir"]
    manifest_path = os.path.join(output_dir, "mouse_manifest.csv")
    embed_dir     = os.path.join(output_dir, "embeddings", "m3d")
    output_path   = os.path.join(embed_dir, "m3d_embeddings.npz")
    os.makedirs(embed_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print(f"[!] mouse_manifest.csv not found at {manifest_path}")
        print("    Run build_nifti_dataset.py --stage 3 first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")
    if device.type == "cpu":
        print("    [!] Warning: M3D-CLIP on CPU will be slow. A GPU is strongly recommended.")

    print(f"[i] Loading M3D-CLIP model ({MODEL_ID})...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(device)
    model.eval()
    print(f"[i] Model ready. Target shape: {TARGET_SHAPE}, "
          f"HU window [{HU_MIN}, {HU_MAX}].\n")

    rows = load_ct_rows(manifest_path)
    if not rows:
        print("[!] No CT crops found in manifest. Exiting.")
        return

    # Filter to existing files
    valid_rows = []
    missing    = []
    for row in rows:
        ct_path = row["ct_hi_nifti"]
        if not os.path.exists(ct_path):
            label = f"{row['mouse_id']} | Week {row['week']}"
            print(f"[!] File not found, skipping: {label}")
            missing.append(label)
        else:
            valid_rows.append(row)

    if not valid_rows:
        print("[!] No valid CT crops found. Exiting.")
        return

    if args.limit:
        valid_rows = valid_rows[:args.limit]
        print(f"[i] --limit {args.limit}: restricting to first {len(valid_rows)} volume(s).")

    print(f"[i] Processing {len(valid_rows)} volumes...\n")

    embeddings  = []
    subject_ids = []
    weeks       = []
    modalities  = []
    paths       = []
    failed      = []

    for i, row in enumerate(valid_rows):
        nifti_path = row["ct_hi_nifti"]
        week_label = f"Week {row['week']}"
        label      = f"{row['mouse_id']} | {week_label}"
        print(f"[{i+1}/{len(valid_rows)}] {label}")

        try:
            emb = embed_volume(nifti_path, model, device, TARGET_SHAPE, HU_MIN, HU_MAX)
            embeddings.append(emb)
            subject_ids.append(row["mouse_id"])
            weeks.append(week_label)
            modalities.append("CT_HiRes")
            paths.append(nifti_path)
            print(f"    [+] OK  (embedding shape: {emb.shape})")
        except Exception as e:
            print(f"    [!] Failed: {e}")
            failed.append(label)

    if not embeddings:
        print("[!] No embeddings were extracted. Exiting.")
        return

    embeddings_array = np.stack(embeddings)   # (N, 768)
    np.savez(
        output_path,
        embeddings  = embeddings_array,
        subject_ids = np.array(subject_ids),
        weeks       = np.array(weeks),
        modalities  = np.array(modalities),
        paths       = np.array(paths),
    )

    print(f"\n[+] Saved {len(embeddings)} embeddings → {output_path}")
    print(f"    Array shape: {embeddings_array.shape}  "
          f"(N crops × {embeddings_array.shape[1]} dimensions)")
    print(f"    Target shape: {TARGET_SHAPE},  HU window: [{HU_MIN}, {HU_MAX}]")
    print(f"\n    To evaluate, run:")
    print(f"    python scripts/evaluate_embeddings.py \\")
    print(f"        --embeddings {output_path} \\")
    print(f"        --output-dir {embed_dir}")

    all_skipped = missing + failed
    if all_skipped:
        print(f"\n[!] {len(all_skipped)} crop(s) failed or were skipped:")
        for s in all_skipped:
            print(f"    - {s}")


if __name__ == "__main__":
    main()
