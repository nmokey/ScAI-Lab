"""
qa_visualize_all.py

Runs a GIF QA visualisation for every CT-Hi crop in mouse_manifest.csv and
saves the results in a directory tree that mirrors mice/ but contains only
GIFs — so you can browse them without touching the source NIfTIs.

Output layout (mirrors mice/ structure):
    {output_dir}/qa_gifs/{mouse_id}/week_{N}/ct_hi_scan.gif

Usage:
    python scripts/qa_visualize_all.py              # all mice
    python scripts/qa_visualize_all.py --limit 4    # first 4 (quick sanity check)
    python scripts/qa_visualize_all.py --mouse NaF_WT_01 NaF_WT_02
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"'{config_path}' not found. "
            "Copy config.yaml.example → config.yaml and fill in your paths."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_gif(nifti_path, out_path):
    """
    Generate a scrolling 3-plane GIF from a NIfTI volume and save to out_path.
    Identical rendering logic to visualize_nifti.py, but output path is explicit.
    """
    vol = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path))  # (Z, Y, X)
    n_z, n_y, n_x = vol.shape

    vmin, vmax = np.percentile(vol, 1), np.percentile(vol, 99)
    n_frames = min(max(n_z, n_y, n_x), 150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")

    kw = dict(cmap="gray", vmin=vmin, vmax=vmax)
    im_ax  = axes[0].imshow(vol[0, :, :], **kw)
    im_cor = axes[1].imshow(vol[:, 0, :], origin="lower", **kw)
    im_sag = axes[2].imshow(vol[:, :, 0], origin="lower", **kw)

    t_ax  = axes[0].set_title("", color="white", fontsize=11)
    t_cor = axes[1].set_title("", color="white", fontsize=11)
    t_sag = axes[2].set_title("", color="white", fontsize=11)
    plt.tight_layout()

    def _slice(frame, n):
        return round(frame * (n - 1) / max(n_frames - 1, 1))

    def update(frame):
        z, y, x = _slice(frame, n_z), _slice(frame, n_y), _slice(frame, n_x)
        im_ax.set_data(vol[z, :, :])
        im_cor.set_data(vol[:, y, :])
        im_sag.set_data(vol[:, :, x])
        t_ax.set_text(f"Axial  Z={z}/{n_z - 1}")
        t_cor.set_text(f"Coronal  Y={y}/{n_y - 1}")
        t_sag.set_text(f"Sagittal  X={x}/{n_x - 1}")

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, writer=animation.PillowWriter(fps=20))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (quick sanity check).")
    parser.add_argument("--mouse", nargs="+", default=None,
                        help="Process only the specified mouse IDs.")
    args = parser.parse_args()

    cfg = load_config()
    output_dir    = cfg["paths"]["output_dir"]
    manifest_path = os.path.join(output_dir, "mouse_manifest.csv")
    mice_dir      = os.path.join(output_dir, "mice")
    qa_dir        = os.path.join(output_dir, "qa_gifs")

    if not os.path.exists(manifest_path):
        print(f"[!] mouse_manifest.csv not found at {manifest_path}")
        print("    Run build_nifti_dataset.py --stage 3 first.")
        sys.exit(1)

    # Load manifest rows that have a CT-Hi path
    rows = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ct_hi_nifti", "").strip():
                rows.append(row)

    # Apply filters
    if args.mouse:
        rows = [r for r in rows if r["mouse_id"] in args.mouse]
    if args.limit:
        rows = rows[:args.limit]

    print(f"[i] Processing {len(rows)} CT-Hi volumes → {qa_dir}/")

    done, skipped, failed = 0, 0, 0

    for i, row in enumerate(rows):
        nifti_path = row["ct_hi_nifti"]
        mouse_id   = row["mouse_id"]
        week       = row["week"]

        # Derive mirrored output path
        rel_path = os.path.relpath(nifti_path, mice_dir)          # NaF_WT_01/week_12/ct_hi.nii.gz
        gif_name = os.path.splitext(os.path.splitext(           # strip .nii.gz
                       os.path.basename(rel_path))[0])[0] + "_scan.gif"
        out_path = os.path.join(qa_dir,
                                os.path.dirname(rel_path),
                                gif_name)

        label = f"{mouse_id} | Week {week}"
        print(f"[{i+1}/{len(rows)}] {label}")

        if not os.path.exists(nifti_path):
            print(f"    [!] Source file not found — skipping.")
            skipped += 1
            continue

        if os.path.exists(out_path):
            print(f"    [~] GIF already exists — skipping.")
            skipped += 1
            continue

        try:
            make_gif(nifti_path, out_path)
            print(f"    [+] → {os.path.relpath(out_path, output_dir)}")
            done += 1
        except Exception as e:
            print(f"    [!] Failed: {e}")
            failed += 1

    print(f"\n[+] Done.  Generated: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"    GIFs saved under: {qa_dir}/")


if __name__ == "__main__":
    main()
