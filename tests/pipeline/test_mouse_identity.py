"""
R4 -- can a failed quadrant silently reassign scans to the wrong mouse?

The mechanism
-------------
`build_nifti_dataset.segment_animals` walks the four quadrants and *skips* any that
holds no bone or less than 1 mL of it:

    for pos_idx, mask in enumerate(quadrant_masks):
        if not mask.any():      ... continue
        if len(q_xx) < min_voxels: ... continue
        bboxes.append({... "position": pos_idx + 1})

So the returned list is **compacted**: three populated quadrants out of four yield a
3-element list, and each entry still carries its true `position`.

`stage3_crop_and_write` then pairs bboxes with mouse numbers **by list index**, never
consulting `position`:

    n_crops = min(len(bboxes), len(mouse_nums))
    for i in range(n_crops):
        bbox = bboxes[i]
        mouse_num = mouse_nums[i]                       # <- positional, not by quadrant
        mouse_id = f"{cohort}_{genotype}_{mouse_num:02d}"

When every expected quadrant is populated, index i and position i+1 coincide and all is
well. When a quadrant that *should* have held a mouse is skipped -- weak bone contrast,
a mouse partly outside the FOV, a thresholding failure -- every subsequent mouse shifts
up by one, and the last mouse is dropped by the `min()`.

Why this matters more than a normal bug
---------------------------------------
Mouse identity is the join key for the entire project. A shifted assignment gives a scan
the wrong subject ID, and since genotype is derived from the subject ID
(`NaF_KO_07` -> "KO") and mouse numbers are persistent longitudinal identifiers, it also
gives it the wrong genotype and splices it into another animal's trajectory. That would
corrupt the encoder evaluation and the VLM equally -- upstream of every result in the
repo, not just the VLM ones.

The code does print a warning when counts disagree, but it prints to stdout mid-run and
then proceeds to write the mislabeled files anyway.

docs/DATA_MANIFEST.md flags several sessions with irregular mouse numbering
("KO 9,11,12", "KO 2,3", "WT 19,20 + WT 1,2"), so the ordering assumption is load-bearing
on real data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.unit

SIZE = 100          # voxels per axis
SPACING = 1.0       # mm -- segment_animals downsamples to ~1mm, so no downsampling here
BONE_HU = 800       # above the 300 HU bone threshold
AIR_HU = -1000
BLOB = 15           # 15^3 = 3375 voxels > the 1 mL (1000 voxel) minimum

# Quadrant convention from segment_animals, with identity direction and origin 0:
#   higher physical X = "left", lower physical Y = "lower"
#   pos 1 = lower-left   pos 2 = lower-right
#   pos 3 = upper-left   pos 4 = upper-right
CENTRE = SIZE // 2
QUADRANT_CENTRES = {
    1: (CENTRE + 25, CENTRE - 25),  # x > centre, y < centre
    2: (CENTRE - 25, CENTRE - 25),  # x < centre, y < centre
    3: (CENTRE + 25, CENTRE + 25),  # x > centre, y > centre
    4: (CENTRE - 25, CENTRE + 25),  # x < centre, y > centre
}


def _phantom(positions, blob=BLOB):
    """
    A synthetic multi-animal CT: a bone blob at each requested quadrant position.

    `positions` is a set of 1-4. Anything omitted is air, which is exactly what
    segment_animals sees when a mouse is missing or its bone fails to threshold.
    """
    import SimpleITK as sitk

    arr = np.full((SIZE, SIZE, SIZE), AIR_HU, dtype=np.int16)  # (Z, Y, X)
    half = blob // 2
    for pos in positions:
        cx, cy = QUADRANT_CENTRES[pos]
        arr[
            CENTRE - half : CENTRE + half,
            cy - half : cy + half,
            cx - half : cx + half,
        ] = BONE_HU

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((SPACING, SPACING, SPACING))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _segment(scripts, positions, n_expected=None):
    mod = scripts("build_nifti_dataset")
    img = _phantom(positions)
    bboxes = mod.segment_animals(img, n_expected or len(positions), "phantom")
    return mod, bboxes


# ---------------------------------------------------------------------------
# The phantom itself behaves as segment_animals expects
# ---------------------------------------------------------------------------

def test_phantom_quadrants_are_detected(scripts):
    """Precondition: a full 4-mouse phantom yields exactly positions 1-4."""
    _mod, bboxes = _segment(scripts, {1, 2, 3, 4})
    assert bboxes is not None and len(bboxes) == 4
    assert [b["position"] for b in bboxes] == [1, 2, 3, 4]


def test_empty_quadrant_is_skipped_and_list_is_compacted(scripts):
    """
    The mechanism, isolated: quadrant 2 is empty, so the returned list has three
    entries whose positions are 1, 3, 4 -- index 1 now holds position 3.
    """
    _mod, bboxes = _segment(scripts, {1, 3, 4}, n_expected=3)
    positions = [b["position"] for b in bboxes]
    assert positions == [1, 3, 4], f"expected compacted [1,3,4], got {positions}"
    assert bboxes[1]["position"] == 3, (
        "list index 1 holds quadrant position 3 -- index and position have diverged"
    )


# ---------------------------------------------------------------------------
# The consequence
# ---------------------------------------------------------------------------

def test_skipped_quadrant_does_not_misassign_mouse_identity(scripts, tmp_path):
    """
    F4, fixed. Four mice were scanned (manifest says mouse_nums = [1, 2, 3, 4]) but
    quadrant 2's segmentation failed.

    The old code paired bboxes to mouse numbers by list index and produced:
        mouse 1 -> quadrant 1   ok
        mouse 2 -> quadrant 3   WRONG
        mouse 3 -> quadrant 4   WRONG
        mouse 4 -> dropped entirely
    writing two scans under the wrong mouse ID (and therefore the wrong genotype, since
    genotype is parsed out of the subject ID) and losing one animal.

    The real function is now called directly, and must refuse rather than guess.
    """
    mod, bboxes = _segment(scripts, {1, 3, 4}, n_expected=4)
    assert [b["position"] for b in bboxes] == [1, 3, 4]

    session = {
        "base_id": "m54231", "week": 12, "tracer": "NaF", "genotype": "KO",
        "group": "Disease", "timepoint_h": "3h", "mouse_nums": [1, 2, 3, 4],
        "n_mice": 4, "notes": "",
    }

    with pytest.raises(RuntimeError) as exc:
        mod.stage3_crop_and_write(
            session=session, nifti_paths={"ct_hi": str(tmp_path / "ct.nii.gz")},
            bboxes=bboxes, output_root=str(tmp_path), dry_run=True,
        )

    message = str(exc.value)
    print(f"\n  {message.splitlines()[0]}")
    assert "[1, 3, 4]" in message, "the error should name the quadrants actually found"
    assert "m54231" in message, "the error should name the session so it can be inspected"


def test_count_mismatch_is_a_hard_error(scripts):
    """
    F4, fixed. When the segmenter finds fewer animals than the manifest declares, the
    pipeline now raises instead of cropping `min(len(bboxes), len(mouse_nums))` animals
    under shifted identities.
    """
    import inspect

    mod = scripts("build_nifti_dataset")
    src = inspect.getsource(mod.stage3_crop_and_write)

    assert "mouse_num = mouse_nums[i]" in src, "wiring changed -- re-derive R4"
    guard = src[src.index("if len(bboxes) != len(mouse_nums):") :]
    guard = guard[: guard.index("n_crops")]

    assert "raise" in guard, (
        "A bbox/mouse_nums count mismatch only prints a warning and then proceeds to "
        "write crops under shifted mouse IDs:\n"
        f"{guard.strip()}\n"
        "Silent data mislabeling should be fatal."
    )


# ---------------------------------------------------------------------------
# Scope: how much real data could be affected
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_no_session_lost_a_quadrant(data_root):
    """
    Bounds the blast radius on real data. `crop_position` is recorded per row in
    mouse_manifest.csv, so any session whose positions are not a contiguous 1..n run
    hit the compaction path and may carry shifted identities.
    """
    import csv

    manifest = data_root / "mouse_manifest.csv"
    if not manifest.exists():
        pytest.skip(f"mouse_manifest.csv not found at {manifest}")

    by_session = {}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["session_id"], row["week"])
            by_session.setdefault(key, []).append(
                (int(row["crop_position"]), int(row["mouse_num"]), row["mouse_id"])
            )

    suspect = {}
    for key, entries in by_session.items():
        positions = sorted(p for p, _, _ in entries)
        if positions != list(range(1, len(positions) + 1)):
            suspect[key] = sorted(entries)

    for key, entries in suspect.items():
        print(f"\n  {key}: positions {[p for p, _, _ in entries]} -- "
              f"mice {[m for _, m, _ in entries]}")

    assert not suspect, (
        f"{len(suspect)} session(s) have non-contiguous crop positions, meaning a "
        f"quadrant was skipped and mouse identities were assigned by shifted index. "
        f"Affected sessions listed above; each needs its mouse->quadrant mapping "
        f"verified by hand against the source DICOM."
    )
