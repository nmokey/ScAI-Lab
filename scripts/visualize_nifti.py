import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; must be set before importing pyplot
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def make_scan_gif(nifti_path):
    print(f"Loading {nifti_path}...")
    vol = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path))  # (Z, Y, X) in numpy
    n_z, n_y, n_x = vol.shape
    print(f"    Volume shape (Z, Y, X): {vol.shape}")

    # Percentile clipping gives robust contrast for both CT (HU) and PET (SUV)
    vmin, vmax = np.percentile(vol, 1), np.percentile(vol, 99)

    # Cap at 150 frames so the GIF stays a manageable file size;
    # slices are sampled proportionally in each direction.
    n_frames = min(max(n_z, n_y, n_x), 150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")

    kw = dict(cmap="gray", vmin=vmin, vmax=vmax)
    im_ax  = axes[0].imshow(vol[0, :, :],  **kw)
    im_cor = axes[1].imshow(vol[:, 0, :],  origin="lower", **kw)
    im_sag = axes[2].imshow(vol[:, :, 0],  origin="lower", **kw)

    t_ax  = axes[0].set_title("", color="white", fontsize=11)
    t_cor = axes[1].set_title("", color="white", fontsize=11)
    t_sag = axes[2].set_title("", color="white", fontsize=11)
    plt.tight_layout()

    def _slice(frame, n):
        """Map frame index → slice index, proportionally across n slices."""
        return round(frame * (n - 1) / max(n_frames - 1, 1))

    def update(frame):
        z = _slice(frame, n_z)
        y = _slice(frame, n_y)
        x = _slice(frame, n_x)

        im_ax.set_data(vol[z, :, :])
        im_cor.set_data(vol[:, y, :])
        im_sag.set_data(vol[:, :, x])

        t_ax.set_text(f"Axial  Z={z}/{n_z - 1}")
        t_cor.set_text(f"Coronal  Y={y}/{n_y - 1}")
        t_sag.set_text(f"Sagittal  X={x}/{n_x - 1}")

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)

    if nifti_path.endswith(".nii.gz"):
        out_path = nifti_path[:-7] + "_scan.gif"
    elif nifti_path.endswith(".nii"):
        out_path = nifti_path[:-4] + "_scan.gif"
    else:
        out_path = nifti_path + "_scan.gif"

    print(f"Rendering {n_frames} frames at 20 fps...")
    ani.save(out_path, writer=animation.PillowWriter(fps=20))
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_nifti.py <path_to_nifti_file>")
    else:
        make_scan_gif(sys.argv[1])
