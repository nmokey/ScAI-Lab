"""
getColipriEmbeddings.py

Iterates over all CT NIfTI volumes in manifest.csv, passes each through
Microsoft's COLIPRI 3D vision encoder, and saves the resulting embeddings.

Installation (one-time):
    pip install colipri torchio
    # or follow the HuggingFace instructions at:
    # https://huggingface.co/microsoft/colipri

Output:
    <output_dir>/colipri_embeddings.npz
        embeddings  : float32 array of shape (N, 768) — one 768-d vector per volume
        subject_ids : string array of shape (N,)
        weeks       : string array of shape (N,)
        modalities  : string array of shape (N,)
        paths       : string array of shape (N,) — path to source NIfTI

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
CT_MODALITIES = {"CT_HiRes", "CT_LowRes"}

def load_ct_rows(manifest_path):
    """Read manifest.csv and return only CT rows (HiRes preferred; LowRes as fallback)."""
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Modality"] in CT_MODALITIES:
                rows.append(row)

    # If a subject+week has both HiRes and LowRes, keep only HiRes
    seen = {}
    for row in rows:
        key = (row["SubjectID"], row["Week"])
        if key not in seen or row["Modality"] == "CT_HiRes":
            seen[key] = row
    deduplicated = list(seen.values())

    print(f"[i] Found {len(deduplicated)} CT volumes in manifest ({len(rows)} total, {len(rows) - len(deduplicated)} LowRes shadowed by HiRes).")
    return deduplicated


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
    manifest_path = os.path.join(output_dir, "manifest.csv")
    output_path   = os.path.join(output_dir, "colipri_embeddings.npz")

    if not os.path.exists(manifest_path):
        print(f"[!] manifest.csv not found at {manifest_path}")
        print("    Run dicomToNifti.py first to generate the manifest.")
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
        print("[!] No CT volumes found in manifest. Exiting.")
        return

    # Extract embeddings
    embeddings  = []
    subject_ids = []
    weeks       = []
    modalities  = []
    paths       = []
    failed      = []

    for i, row in enumerate(rows):
        nifti_path = row["Path"]
        label = f"{row['SubjectID']} | {row['Week']} | {row['Modality']}"
        print(f"[{i+1}/{len(rows)}] {label}")

        if not os.path.exists(nifti_path):
            print(f"    [!] File not found, skipping.")
            failed.append(label)
            continue

        try:
            emb = extract_embedding(nifti_path, model, processor, device)
            embeddings.append(emb)
            subject_ids.append(row["SubjectID"])
            weeks.append(row["Week"])
            modalities.append(row["Modality"])
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
    print(f"    Array shape: {embeddings_array.shape}  (N volumes × 768 dimensions)")

    if failed:
        print(f"\n[!] {len(failed)} volume(s) failed or were skipped:")
        for f in failed:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
