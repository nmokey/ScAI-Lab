"""
get_merlin_embeddings.py

Iterates over all per-mouse CT-Hi NIfTI crops in mouse_manifest.csv, passes
each through Stanford MIMI's Merlin 3D vision encoder (ImageEmbedding mode),
and saves the resulting embeddings in the standard .npz format consumed by
evaluate_embeddings.py.

Installation (one-time):
    pip install merlin-vlm

Output:
    <output_dir>/embeddings/merlin/merlin_embeddings.npz
        embeddings  : float32 array of shape (N, D) — one D-d vector per crop
        subject_ids : string array of shape (N,) — e.g. "NaF_WT_03"
        weeks       : string array of shape (N,) — e.g. "Week 12"
        modalities  : string array of shape (N,) — always "CT_HiRes" here
        paths       : string array of shape (N,) — absolute path to source NIfTI

    Evaluation outputs (from evaluate_embeddings.py) go alongside:
        <output_dir>/embeddings/merlin/metrics.csv
        <output_dir>/embeddings/merlin/report.txt
        <output_dir>/embeddings/merlin/plots/

Why CT only?
    Merlin was pre-trained on human abdominal CT volumes. PET data uses SUV
    units and a different intensity range, and is reserved for the custom MAE
    (Step 4).

Preprocessing note:
    Merlin's DataLoader handles all preprocessing internally and caches the
    result to <embed_dir>/cache/. The first run is slow; subsequent runs skip
    preprocessing for already-cached volumes.
"""

import argparse
import os
import csv
import yaml
import torch
import numpy as np

from merlin.data import DataLoader
from merlin import Merlin


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
    Skips rows where ct_hi_nifti is missing.
    """
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ct_path = row.get("ct_hi_nifti", "").strip()
            if ct_path:
                rows.append(row)
    print(f"[i] Found {len(rows)} CT-Hi crops in mouse_manifest.csv.")
    return rows


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N volumes (useful for testing).")
    args = parser.parse_args()

    cfg = load_config()
    output_dir    = cfg["paths"]["output_dir"]
    manifest_path = os.path.join(output_dir, "mouse_manifest.csv")
    embed_dir     = os.path.join(output_dir, "embeddings", "merlin")
    cache_dir     = os.path.join(embed_dir, "cache")
    output_path   = os.path.join(embed_dir, "merlin_embeddings.npz")
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        print(f"[!] mouse_manifest.csv not found at {manifest_path}")
        print("    Run build_nifti_dataset.py --stage 3 first.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Using device: {device}")
    if device.type == "cpu":
        print("    [!] Warning: Merlin on CPU will be very slow. A GPU is strongly recommended.")

    # Load model
    print("[i] Loading Merlin model (stanfordmimi/Merlin, ImageEmbedding=True)...")
    model = Merlin(ImageEmbedding=True)
    model.eval()
    model.to(device)
    print("[i] Model ready.\n")

    # Load manifest
    rows = load_ct_rows(manifest_path)
    if not rows:
        print("[!] No CT crops found in manifest. Exiting.")
        return

    # Filter to existing files and build datalist (preserving manifest order)
    valid_rows = []
    datalist   = []
    missing    = []

    for row in rows:
        ct_path = row["ct_hi_nifti"]
        if not os.path.exists(ct_path):
            label = f"{row['mouse_id']} | Week {row['week']}"
            print(f"[!] File not found, skipping: {label}")
            missing.append(label)
        else:
            valid_rows.append(row)
            datalist.append({"image": ct_path})

    if not datalist:
        print("[!] No valid CT crops found. Exiting.")
        return

    if args.limit:
        valid_rows = valid_rows[:args.limit]
        datalist   = datalist[:args.limit]
        print(f"[i] --limit {args.limit}: restricting to first {len(datalist)} volume(s).")

    print(f"[i] Processing {len(datalist)} volumes...\n")

    # shuffle=False keeps datalist order aligned with valid_rows
    dataloader = DataLoader(
        datalist=datalist,
        cache_dir=cache_dir,
        batchsize=1,
        shuffle=False,
        num_workers=0,
    )

    embeddings  = []
    subject_ids = []
    weeks       = []
    modalities  = []
    paths       = []
    failed      = []

    for i, batch in enumerate(dataloader):
        row        = valid_rows[i]
        week_label = f"Week {row['week']}"
        label      = f"{row['mouse_id']} | {week_label}"
        print(f"[{i+1}/{len(valid_rows)}] {label}")

        try:
            with torch.no_grad():
                outputs = model(batch["image"].to(device))
            emb = outputs[0].squeeze(0).cpu().float().numpy()  # (D,)
            embeddings.append(emb)
            subject_ids.append(row["mouse_id"])
            weeks.append(week_label)
            modalities.append("CT_HiRes")
            paths.append(row["ct_hi_nifti"])
            print(f"    [+] OK  (embedding shape: {emb.shape})")
        except Exception as e:
            print(f"    [!] Failed: {e}")
            failed.append(label)

    if not embeddings:
        print("[!] No embeddings were extracted. Exiting.")
        return

    # Save
    embeddings_array = np.stack(embeddings)  # (N, D)
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
