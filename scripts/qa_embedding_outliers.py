#!/usr/bin/env python
"""
QA gate: flag crops whose embedding is an outlier against the population (F16).

Why this exists. `build_nifti_dataset.segment_animals` counts any quadrant with
>= 1 mL of bone-density voxels as a mouse. The scanner bed's curved fluid line
clears that bar, so a session with fewer than four mice can have a TUBE
segmented, cropped, and written under a real mouse's ID. Three such crops were
found on 2026-09-13 (NaF_KO_09, FDG_WT_01, FDG_WT_02 -- all Week 15), and they
had passed through embedding extraction, TBR extraction, the longitudinal MLP,
and the VLM before anyone looked.

A tube's embedding is nowhere near the mouse population: cosine to the centroid
0.78-0.86 against a median of 0.989 (MAD 0.005), i.e. z <= -25, with a clean gap
to the next crop at z = -5.6. This is a one-second check that catches all three.

Run it on every encoder's .npz after stage 2, and again after any re-crop:

    python scripts/qa_embedding_outliers.py --embeddings <path>.npz
    python scripts/qa_embedding_outliers.py --embeddings <path>.npz --z -8 --fail

Exit code is non-zero with --fail when anything is flagged, so it can gate a run.
"""

import argparse
import sys

import numpy as np


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-12), med, mad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--z", type=float, default=-10.0,
                    help="flag crops with robust z below this (default -10; the three known "
                         "tube crops are at -25 to -40 and the next crop is at -5.6)")
    ap.add_argument("--top", type=int, default=15, help="also print the N most anomalous regardless")
    ap.add_argument("--fail", action="store_true", help="exit 1 if anything is flagged")
    a = ap.parse_args()

    d = np.load(a.embeddings, allow_pickle=True)
    E = d["embeddings"].astype(np.float64)
    sids = d["subject_ids"].astype(str)
    weeks = d["weeks"].astype(str)

    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    mu = En.mean(0); mu /= np.linalg.norm(mu)
    cos = En @ mu
    z, med, mad = robust_z(cos)
    order = np.argsort(z)

    print(f"{len(E)} crops   cosine-to-centroid median={med:.4f}  MAD={mad:.4f}   threshold z<{a.z}")
    flagged = [i for i in order if z[i] < a.z]
    print(f"\nFLAGGED ({len(flagged)}):")
    for i in flagged:
        print(f"  {sids[i]:12} {weeks[i]:8}  cos={cos[i]:.4f}  z={z[i]:+.1f}")
    print(f"\nmost anomalous {a.top}:")
    for i in order[:a.top]:
        mark = " <-- FLAGGED" if z[i] < a.z else ""
        print(f"  {sids[i]:12} {weeks[i]:8}  cos={cos[i]:.4f}  z={z[i]:+.1f}{mark}")

    if flagged and a.fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
