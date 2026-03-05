"""
get_colipri_embeddings.py

Iterates over all per-mouse CT-Hi NIfTI crops in mouse_manifest.csv, passes
each through Microsoft's COLIPRI 3D vision encoder, and saves the resulting
embeddings in the standard .npz format consumed by evaluate_embeddings.py.

Installation (one-time):
    pip install colipri torchio
    # or follow the HuggingFace instructions at:
    # https://huggingface.co/microsoft/colipri

Output:
    <output_dir>/embeddings/colipri/colipri_embeddings.npz
        embeddings  : float32 array of shape (N, 768) — one 768-d vector per crop
        subject_ids : string array of shape (N,) — e.g. "NaF_WT_03"
        weeks       : string array of shape (N,) — e.g. "Week 12"
        modalities  : string array of shape (N,) — always "CT_HiRes" here
        paths       : string array of shape (N,) — absolute path to source NIfTI

    Evaluation outputs (from evaluate_embeddings.py) go alongside:
        <output_dir>/embeddings/colipri/metrics.csv
        <output_dir>/embeddings/colipri/report.txt
        <output_dir>/embeddings/colipri/plots/

Why CT only?
    COLIPRI was pre-trained on human chest CT using Hounsfield Unit (HU)
    normalization. PET data uses SUV units and would require a different
    normalization strategy. PET is reserved for the custom MAE (Step 4).
"""

import os
import csv
import yaml
import torch
import numpy as np
import torchio as tio
from colipri import get_model, get_processor


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
    """
    Read mouse_manifest.csv and return one entry per CT-Hi crop.
    Skips rows where ct_hi_nifti is missing or the file does not exist.
    """
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ct_path = row.get("ct_hi_nifti", "").strip()
            if ct_path:
                rows.append(row)
    print(f"[i] Found {len(rows)} CT-Hi crops in mouse_manifest.csv.")
    return rows


# --- Embedding Extraction ---
def extract_embedding(nifti_path, model, processor, device):
    """
    Load a NIfTI, run COLIPRI preprocessing, and return a pooled 768-d embedding.

    COLIPRI's processor handles all preprocessing internally:
      - Reorients to RAS+ coordinate system
      - Resamples to 2 mm isotropic voxel spacing
      - Resizes to 192 × 192 × 192
      - Normalises HU values to [-1, +1] (clipped at ±1000 HU)

    encode_image args:
      pool=True     → multi-head attention pooling over the 24³ patch token grid
      project=True  → project into the text-aligned embedding space (768-d)
    """
    volume = tio.ScalarImage(nifti_path)
    preprocessed = processor.process_images(volume)
    batch = processor.to_images_batch(preprocessed).to(device)

    with torch.no_grad():
        embedding = model.encode_image(batch, pool=True, project=True)

    return embedding.squeeze(0).cpu().numpy()  # (768,)


# --- Main ---
def main():
    cfg = load_config()
    output_dir    = cfg["paths"]["output_dir"]
    manifest_path = os.path.join(output_dir, "mouse_manifest.csv")
    embed_dir     = os.path.join(output_dir, "embeddings", "colipri")
    output_path   = os.path.join(embed_dir, "colipri_embeddings.npz")
    os.makedirs(embed_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print(f"[!] mouse_manifest.csv not found at {manifest_path}")
        print("    Run build_nifti_dataset.py --stage 3 first.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")
    if device.type == "cpu":
        print("    [!] Warning: COLIPRI on CPU will be very slow. A GPU is strongly recommended.")

    # Load model
    print("[i] Loading COLIPRI model (microsoft/colipri)...")
    model = get_model().to(device)
    model.eval()
    processor = get_processor()
    print("[i] Model ready.\n")

    # Load manifest
    rows = load_ct_rows(manifest_path)
    if not rows:
        print("[!] No CT crops found in manifest. Exiting.")
        return

    # Extract embeddings
    embeddings  = []
    subject_ids = []
    weeks       = []
    modalities  = []
    paths       = []
    failed      = []

    for i, row in enumerate(rows):
        nifti_path = row["ct_hi_nifti"]
        # Normalise week to "Week N" format expected by evaluate_embeddings.py
        week_label = f"Week {row['week']}"
        label = f"{row['mouse_id']} | {week_label}"
        print(f"[{i+1}/{len(rows)}] {label}")

        if not os.path.exists(nifti_path):
            print(f"    [!] File not found, skipping.")
            failed.append(label)
            continue

        try:
            emb = extract_embedding(nifti_path, model, processor, device)
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

    # Save
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
    print(f"    Array shape: {embeddings_array.shape}  (N crops × 768 dimensions)")
    print(f"\n    To evaluate, run:")
    print(f"    python scripts/evaluate_embeddings.py \\")
    print(f"        --embeddings {output_path} \\")
    print(f"        --output-dir {embed_dir}")

    if failed:
        print(f"\n[!] {len(failed)} crop(s) failed or were skipped:")
        for f in failed:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
