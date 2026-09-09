"""
R10 -- does the spine-anchored TBR-3 strategy locate the aorta where it claims?

The mechanism
-------------
`extract_tbr_features.tbr_strategy_3` indexes CT as `[x, y, z]` -- its own unpacking
says so:

    ct_nz, ct_ny, ct_nz_slices = ct_data.shape     # (x, y, z)
    bx0 = int(ct_nz * 0.15)                        # first axis used as X
    sl  = ct_data[:, :, ct_z]                      # -> a 2-D (x, y) slice

`np.where` on that 2-D slice returns (first-axis indices, second-axis indices) =
(x indices, y indices). The code unpacks them the other way round:

    ys, xs = np.where(labeled == spine_id)
    spine_cx = float(xs.mean())    # xs actually holds Y indices
    spine_cy = float(ys.mean())    # ys actually holds X indices

So the centroid coordinates are transposed, and the 3 mm "anterior to the spine" offset
    aorta_cy = spine_cy + aorta_offset_vox
is applied along the wrong anatomical axis. The bounds check compounds it, testing an
X-derived value against the Y extent (`if aorta_cy + aorta_radius_vox >= ct_ny`).

Why it still matters even though TBR-3 is unused
------------------------------------------------
`TBR_COL = "tbr2_p95_median"` -- the pipeline uses TBR-2, so no published number depends
on TBR-3 directly. But docs/design_decisions.md rejects TBR-3 on empirical grounds
("most anatomically specific but fragile -- NaN in >50% of slices due to spine detection
failures") and uses that to justify the population-level proxy that the entire VLM TBR
task is trained against. If the rejection rests on a coordinate bug rather than on
resolution limits, the choice of ground truth deserves revisiting.

The phantom
-----------
A cylinder of "bone" at a deliberately asymmetric position, plus a PET hot spot placed
where a correct implementation should look. Asymmetry is the point: with the spine on
the diagonal the swap would be invisible.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.unit

NX, NY, NZ = 120, 100, 60          # CT grid, deliberately non-square so axes are distinguishable
SPINE_X, SPINE_Y = 40, 30          # asymmetric: swapping x/y gives a different location
AORTA_OFFSET_VOX = 30              # the constant used inside tbr_strategy_3
BONE_HU, SOFT_HU = 800, 0


def _phantoms():
    """
    CT with a spine column at (SPINE_X, SPINE_Y), and PET with a hot spot at the
    anatomically correct aorta location: same X, offset in +Y.
    """
    ct = np.full((NX, NY, NZ), SOFT_HU, dtype=np.float32)
    ct[SPINE_X - 5 : SPINE_X + 5, SPINE_Y - 5 : SPINE_Y + 5, :] = BONE_HU

    # PET on a coarser grid, as in the real data (CT ~0.1mm, PET 0.54mm).
    pnx, pny, pnz = NX // 4, NY // 4, NZ // 4
    rng = np.random.default_rng(0)
    # Noisy background: a perfectly uniform one makes P95 == background, leaving
    # `mid_band < bg_thresh` empty and the ratio NaN for reasons unrelated to F9.
    pet = rng.gamma(4.0, 0.25, size=(pnx, pny, pnz)).astype(np.float32)

    sx, sy = pnx / NX, pny / NY
    hot_x = int(SPINE_X * sx)
    hot_y = int((SPINE_Y + AORTA_OFFSET_VOX) * sy)   # correct: offset along Y
    pet[max(0, hot_x - 1) : hot_x + 2, max(0, hot_y - 1) : hot_y + 2, :] = 50.0
    return ct, pet


def test_spine_centroid_axes_are_not_transposed(repo_root):
    """
    The unpacking order, read from the source so this test tracks the real code rather
    than freezing a copy of it.

    `sl` is a 2-D slice of a volume indexed [x, y, z], so `np.where` yields
    (x indices, y indices). Unpacking it as `ys, xs` transposes the centroid.
    """
    src = (repo_root / "scripts" / "extract_tbr_features.py").read_text()
    line = next(
        ln.strip() for ln in src.splitlines() if "np.where(labeled == spine_id)" in ln
    )
    print(f"\n  {line}")
    assert line.startswith("xs, ys"), (
        f"`{line}` unpacks a slice indexed [x, y] in the wrong order, transposing the "
        f"spine centroid. The 3 mm anterior offset `aorta_cy = spine_cy + offset` is then "
        f"applied along the wrong anatomical axis, and the bounds check compares an "
        f"X-derived value against the Y extent."
    )


@pytest.mark.probe
def test_tbr3_roi_finds_the_planted_hot_spot():
    """
    EXPECTED TO FAIL.

    End-to-end on the real function: the PET hot spot sits exactly where a correct
    aortic ROI belongs, so a working implementation should report a TBR well above the
    background of 1.0.
    """
    mod = pytest.importorskip("scipy") and None
    from conftest import load_script_module

    strategy = load_script_module("extract_tbr_features").tbr_strategy_3
    ct, pet = _phantoms()
    result = strategy(ct, pet)

    tbr = result.get("tbr3_tbr")
    print(f"\n  planted hot spot = 50x background; tbr_strategy_3 reports: {result}")

    assert tbr is not None and not np.isnan(tbr) and tbr > 5.0, (
        f"With a 50x hot spot placed at the correct aortic location, tbr_strategy_3 "
        f"returns tbr3_tbr={tbr}. The ROI is not landing where the docstring says it "
        f"does. docs/design_decisions.md rejects TBR-3 as anatomically fragile and uses "
        f"that to justify TBR-2 as the ground truth for the entire VLM TBR task -- if "
        f"the rejection rests on this coordinate bug, that choice needs revisiting."
    )


# ---------------------------------------------------------------------------
# TBR-1 and TBR-2 compute what they claim
# ---------------------------------------------------------------------------

def test_tbr1_matches_hand_computation(scripts):
    """TBR-1 is P99/median over nonzero voxels -- verified against numpy directly."""
    mod = scripts("extract_tbr_features")
    rng = np.random.default_rng(0)
    pet = rng.gamma(2.0, 2.0, size=(20, 20, 20)).astype(np.float32)

    got = mod.tbr_strategy_1(pet)
    nz = pet[pet > 0].flatten()
    assert got["tbr1_p99_median"] == pytest.approx(
        float(np.percentile(nz, 99) / np.median(nz)), rel=1e-6
    )
    assert got["tbr1_mean"] == pytest.approx(float(nz.mean()), rel=1e-6)


def test_tbr2_matches_hand_computation(scripts):
    """
    TBR-2 -- the column the whole pipeline actually uses. Restricts to the middle third
    in Z, then takes P95 over that band divided by the median of the sub-P95 remainder.
    """
    mod = scripts("extract_tbr_features")
    rng = np.random.default_rng(1)
    pet = rng.gamma(2.0, 2.0, size=(20, 20, 30)).astype(np.float32)

    got = mod.tbr_strategy_2(pet)

    nz = pet.shape[2]
    mid = pet[:, :, nz // 3 : 2 * nz // 3].flatten()
    mid = mid[mid > 0]
    thresh = np.percentile(mid, 95)
    expected = float(np.percentile(mid, 95)) / float(np.median(mid[mid < thresh]))

    assert got["tbr2_p95_median"] == pytest.approx(expected, rel=1e-6)


def test_tbr2_guards_against_too_few_voxels(scripts):
    """Fewer than 50 nonzero voxels in the mid-band must yield NaN, not a spurious ratio."""
    mod = scripts("extract_tbr_features")
    pet = np.zeros((10, 10, 30), dtype=np.float32)
    pet[0, 0, 12] = 5.0
    assert np.isnan(mod.tbr_strategy_2(pet)["tbr2_p95_median"])


def test_tbr2_is_sensitive_to_the_z_crop(scripts):
    """
    TBR-2's mid-Z band is defined as a *fraction* of the crop, so its value depends on
    how the animal was cropped. `crop_to_physical_bbox` pads by a fixed 70 mm in Z, but
    the underlying scan extent varies. Reported rather than asserted at a threshold --
    the point is that crop geometry is a free parameter feeding the ground truth.
    """
    mod = scripts("extract_tbr_features")
    rng = np.random.default_rng(2)
    base = rng.gamma(2.0, 2.0, size=(20, 20, 60)).astype(np.float32)
    base[:, :, 40:] *= 3.0  # a hot region toward one end, as bladder/bone would be

    full = mod.tbr_strategy_2(base)["tbr2_p95_median"]
    trimmed = mod.tbr_strategy_2(base[:, :, :50])["tbr2_p95_median"]
    print(f"\n  TBR-2 on full crop = {full:.3f}; on a 17% shorter crop = {trimmed:.3f} "
          f"({abs(trimmed - full) / full * 100:.1f}% change)")
    assert not np.isnan(full) and not np.isnan(trimmed)
