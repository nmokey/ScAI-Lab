"""
get_raddino_embeddings.py

Iterates over all per-mouse CT-Hi NIfTI crops in mouse_manifest.csv, passes
each through Microsoft's RAD-DINO 2D vision encoder, and saves the resulting
embeddings in the standard .npz format consumed by evaluate_embeddings.py.

Installation (one-time):
    pip install rad-dino nibabel

RAD-DINO is a 2D model (ViT-Base/14, trained on chest X-rays). To handle 3D
volumes we:
  1. Sample N_SLICES axial slices evenly across the volume.
  2. Apply HU windowing and scale each slice to uint8 [0, 255].
  3. Convert to a 3-channel PIL RGB image (replicating the single HU channel).
  4. Pass each slice through RAD-DINO → CLS embedding (768-d).
  5. Mean-pool the per-slice CLS embeddings → one 768-d vector per volume.

Tunable parameters (top of main()):
  N_SLICES    : number of axial slices to sample per volume (default: 32)
  HU_MIN / HU_MAX : HU window applied before scaling (default: soft-tissue
                    window W=400 L=40, i.e. [-160, 240])

Output:
    <output_dir>/embeddings/raddino/raddino_embeddings.npz
        embeddings  : float32 array of shape (N, 768)
        subject_ids : string array of shape (N,)
        weeks       : string array of shape (N,)
        modalities  : string array of shape (N,) — always "CT_HiRes"
        paths       : string array of shape (N,)
"""

import argparse
import os
import csv
import warnings
import yaml

import nibabel as nib
import numpy as np
from PIL import Image
import torch

from rad_dino import RadDino


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


# --- Slice preprocessing ---
def volume_to_slices(nifti_path, n_slices, hu_min, hu_max):
    """
    Load a NIfTI CT volume and return a list of PIL RGB images, one per
    sampled axial slice.

    Steps:
      1. Load with nibabel; reorient to RAS so axis-2 is always axial (Z).
      2. Sample n_slices evenly across the Z axis.
      3. Clip to [hu_min, hu_max] and scale linearly to uint8 [0, 255].
      4. Convert to RGB PIL Image (replicate single channel × 3).
    """
    img = nib.load(nifti_path)
    # Reorient to RAS+ so that array axis 2 = axial (inferior→superior)
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)  # shape: (X, Y, Z)

    n_z = vol.shape[2]
    indices = np.linspace(0, n_z - 1, n_slices, dtype=int)

    slices = []
    for z in indices:
        sl = vol[:, :, z]
        # HU windowing
        sl = np.clip(sl, hu_min, hu_max)
        # Scale to [0, 255] uint8
        sl = ((sl - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        # PIL grayscale → RGB (RAD-DINO expects 3-channel input)
        pil = Image.fromarray(sl, mode="L").convert("RGB")
        slices.append(pil)

    return slices


# --- Volume embedding ---
def embed_volume(nifti_path, encoder, n_slices, hu_min, hu_max):
    """
    Extract a single 768-d embedding for a 3D CT volume by mean-pooling
    per-slice CLS tokens from RAD-DINO.
    """
    slices = volume_to_slices(nifti_path, n_slices, hu_min, hu_max)
    cls_list = []
    for pil in slices:
        with torch.no_grad():
            cls, _ = encoder.extract_features(pil)   # cls: (1, 768)
        cls_list.append(cls.squeeze(0).cpu().float())  # (768,)
    # Mean-pool across slices
    return torch.stack(cls_list).mean(dim=0).numpy()   # (768,)


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N volumes (useful for testing).")
    args = parser.parse_args()

    # --- Tunable parameters ---
    N_SLICES = 32    # axial slices sampled per volume
    HU_MIN   = -160  # soft-tissue window lower bound (HU)
    HU_MAX   =  240  # soft-tissue window upper bound (HU)
    # ---------------------------

    cfg = load_config()
    output_dir    = cfg["paths"]["output_dir"]
    manifest_path = os.path.join(output_dir, "mouse_manifest.csv")
    embed_dir     = os.path.join(output_dir, "embeddings", "raddino")
    output_path   = os.path.join(embed_dir, "raddino_embeddings.npz")
    os.makedirs(embed_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print(f"[!] mouse_manifest.csv not found at {manifest_path}")
        print("    Run build_nifti_dataset.py --stage 3 first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")
    if device.type == "cpu":
        print("    [!] Warning: RAD-DINO on CPU will be slow. A GPU is strongly recommended.")

    print("[i] Loading RAD-DINO model (microsoft/rad-dino)...")
    encoder = RadDino()
    encoder.to(device)
    encoder.eval()
    print(f"[i] Model ready. Sampling {N_SLICES} axial slices per volume, "
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                emb = embed_volume(nifti_path, encoder, N_SLICES, HU_MIN, HU_MAX)
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

    embeddings_array = np.stack(embeddings)  # (N, 768)
    np.savez(
        output_path,
        embeddings  = embeddings_array,
        subject_ids = np.array(subject_ids),
        weeks       = np.array(weeks),
        modalities  = np.array(modalities),
        paths       = np.array(paths),
    )

    print(f"\n[+] Saved {len(embeddings)} embeddings → {output_path}")
    print(f"    Array shape: {embeddings_array.shape}  (N crops × {embeddings_array.shape[1]} dimensions)")
    print(f"    Slices/volume: {N_SLICES},  HU window: [{HU_MIN}, {HU_MAX}]")
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
