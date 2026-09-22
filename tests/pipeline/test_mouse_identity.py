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


@pytest.mark.parametrize("positions,mouse_nums", [({1,2,3},[1,2,3,4]), ({1,2,3,4},[1,2,3])])
def test_count_mismatch_is_a_hard_error(scripts,tmp_path,positions,mouse_nums):
    """Behavioral guard for both missing animals and surplus tube/phantom detections."""
    mod,bboxes=_segment(scripts,positions,n_expected=len(mouse_nums))
    session=dict(base_id="test_no_override",week=12,tracer="NaF",genotype="KO",
                 group="Disease",timepoint_h="3h",mouse_nums=mouse_nums,n_mice=len(mouse_nums),notes="")
    with pytest.raises(RuntimeError,match="Refusing to assign mouse identities"):
        mod.stage3_crop_and_write(session,{"ct_hi":str(tmp_path/'ct.nii.gz')},bboxes,str(tmp_path),dry_run=True)


# ---------------------------------------------------------------------------
# Scope: how much real data could be affected
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_current_crop_identities_follow_filename_positions_and_overrides(data_root, scripts):
    """Noncontiguous positions can be correct after explicit phantom exclusions."""
    import csv
    import yaml
    repo=Path(__file__).resolve().parents[2]
    manifest=data_root/'mouse_manifest.csv'
    if not manifest.exists():pytest.skip(f'{manifest} unavailable')
    with manifest.open() as stream:crops=list(csv.DictReader(stream))
    with (repo/'manifest.csv').open() as stream:raw=list(csv.DictReader(stream))
    overrides=yaml.safe_load((repo/'quadrant_overrides.yaml').read_text())
    checks=scripts('validate_research_inputs').validate_crop_identities(crops,raw,overrides)
    assert len(checks)>0


@pytest.mark.parametrize('mapping,valid', [
    ({1:9,2:None,3:10,4:11},True),
    ({1:9,3:10,4:11},False),  # Unaccounted detected phantom.
    ({1:9,2:None,3:9,4:11},False),  # Duplicated mouse and missing mouse 10.
    ({1:9,2:None,3:10,4:None},False),  # Missing an expected mouse.
])
def test_override_requires_exact_mouse_coverage(scripts,tmp_path,monkeypatch,mapping,valid):
    mod,bboxes=_segment(scripts,{1,2,3,4})
    monkeypatch.setattr(mod,'load_quadrant_overrides',lambda:{'override_test':mapping})
    session=dict(base_id='override_test',week=15,tracer='NaF',genotype='KO',
                 group='Disease',timepoint_h='3h',mouse_nums=[9,10,11],n_mice=3,notes='',
                 ct_hi_scan_id='override_test')
    def run():
        return mod.stage3_crop_and_write(session,{'ct_hi':str(tmp_path/'ct.nii.gz')},bboxes,str(tmp_path),dry_run=True)
    if valid:
        rows=run()
        assert [(r['mouse_num'],r['crop_position']) for r in rows]==[(9,1),(10,3),(11,4)]
    else:
        with pytest.raises(RuntimeError,match='quadrant_overrides'):run()
