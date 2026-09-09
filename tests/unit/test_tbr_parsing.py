"""
R9 -- is the TBR ground truth preserved losslessly from generation to supervision?

The same numbers are parsed twice by two independent regexes:

  scripts/create_mouse_traj_dataset.py   writes the answer string
  vlm/data/vqa_dataset.py::_tbr_targets  parses it back into the regression target
  vlm/data/eval.py::_parse_tbr           parses it back for the text metric

Both regexes are `r"Week\\s+(\\d+):\\s*(\\d+(?:\\.\\d+)?)"`. Two consequences:

 1. No sign is matched, so a negative value would be silently read as its absolute
    value (or dropped, depending on the surrounding text).
 2. `_tbr_targets` uses **-1.0 as the padding sentinel** for "no data in this slot",
    so a genuine TBR of -1 would be indistinguishable from missing. Both the loss mask
    (`tbr_targets >= 0`) and the evaluator (`if gt_val != -1`) key on that sentinel.

TBR-2 is a ratio of a 95th percentile to a median of positive PET values, so it should
never be negative -- which makes the sentinel *currently* safe. The point of these tests
is to establish that as a checked invariant rather than an assumption, and to pin the
round-trip so a future change to the answer format cannot silently corrupt supervision.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.unit

WEEK_ORDER = ["Week 12", "Week 15", "Week 18", "Week 20"]


def _targets(answer, question="Predict the trajectory of aortic TBR?"):
    """Drive vqa_dataset._tbr_targets without constructing the full Dataset."""
    from data.vqa_dataset import MouseTrajDataset

    stub = object.__new__(MouseTrajDataset)
    return MouseTrajDataset._tbr_targets(stub, {"answer": answer, "question": question})


# ---------------------------------------------------------------------------
# Round-trip: generator -> target tensor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "future_weeks,values,expected",
    [
        # Full attendance: Week 15/18/20 -> deltas 3/6/8 -> slots 0/1/2.
        (["Week 15", "Week 18", "Week 20"], [21.0, 22.5, 24.25], [21.0, 22.5, 24.25, -1.0]),
        # The case design_decisions.md calls out: W15 + W20 only, so slot 1 stays empty.
        (["Week 15", "Week 20"], [21.0, 24.25], [21.0, -1.0, 24.25, -1.0]),
        # A single future scan.
        (["Week 20"], [24.25], [-1.0, -1.0, 24.25, -1.0]),
    ],
)
def test_generated_answer_round_trips_to_the_right_slots(scripts, future_weeks, values, expected):
    """
    The real generator writes the answer; the real parser reads it back. Slot
    assignment must follow the week delta, not the position in the string.
    """
    mod = scripts("create_mouse_traj_dataset")
    tbr_map = {"NaF_KO_01": dict(zip(future_weeks, values))}

    _q, answer = mod.format_tbr_qa("NaF_KO_01", "Week 12", future_weeks, tbr_map)
    assert answer is not None
    got = _targets(answer).tolist()

    print(f"\n  {future_weeks} -> {answer!r}\n    slots {got}")
    assert got == pytest.approx(expected), (
        f"round-trip lost or misplaced values: expected {expected}, got {got}"
    )


def test_eval_and_dataset_parsers_agree(scripts):
    """
    The two independent regexes must extract identical numbers. They are separate copies
    that can drift.
    """
    from data.eval import _parse_tbr

    mod = scripts("create_mouse_traj_dataset")
    weeks = ["Week 15", "Week 18", "Week 20"]
    tbr_map = {"NaF_WT_02": dict(zip(weeks, [19.5, 20.0, 21.75]))}
    _q, answer = mod.format_tbr_qa("NaF_WT_02", "Week 12", weeks, tbr_map)

    from_eval = _parse_tbr(answer)
    from_dataset = _targets(answer).tolist()
    slot_of = {3: 0, 6: 1, 8: 2}

    for delta, val in from_eval.items():
        assert from_dataset[slot_of[delta]] == pytest.approx(val), (
            f"eval._parse_tbr and vqa_dataset._tbr_targets disagree at delta {delta}"
        )


def test_missing_week_becomes_na_and_not_a_number(scripts):
    """A week with no TBR renders as 'NA' and must not populate a slot."""
    mod = scripts("create_mouse_traj_dataset")
    weeks = ["Week 15", "Week 18", "Week 20"]
    tbr_map = {"NaF_KO_03": {"Week 15": 21.0, "Week 20": 24.0}}  # Week 18 absent

    _q, answer = mod.format_tbr_qa("NaF_KO_03", "Week 12", weeks, tbr_map)
    assert "NA" in answer, f"expected an NA placeholder, got {answer!r}"
    assert _targets(answer).tolist() == pytest.approx([21.0, -1.0, 24.0, -1.0])


def test_genotype_only_records_carry_no_tbr_targets(scripts):
    """
    A genotype question must produce an all-sentinel target so the masked MSE
    contributes nothing for those records.
    """
    mod = scripts("create_mouse_traj_dataset")
    _q, answer = mod.format_genotype_qa("NaF_KO_01")
    got = _targets(answer, question=_q).tolist()
    assert got == [-1.0, -1.0, -1.0, -1.0], f"genotype record produced TBR targets: {got}"


# ---------------------------------------------------------------------------
# The sentinel collision
# ---------------------------------------------------------------------------

def test_negative_tbr_is_unrepresentable():
    """
    Documents the sharp edge. A negative value is not matched by the regex at all, so
    the slot silently stays at the padding sentinel -- the record looks like "no data"
    rather than "data we failed to read".

    Safe today only because TBR-2 is a ratio of positive quantities. Asserted here so
    the assumption is visible, and paired with the real-data check below.
    """
    got = _targets("The predicted trajectory for aortic TBR - Week 3: -5.00.").tolist()
    print(f"\n  answer contains 'Week 3: -5.00' -> slots {got}")
    assert got[0] == -1.0, "expected the negative value to be dropped to the sentinel"

    # And an actual -1 would be read as missing, indistinguishable from padding.
    assert _targets("... Week 3: -1.00.").tolist()[0] == -1.0


@pytest.mark.realdata
def test_real_tbr_values_are_positive(data_root):
    """
    The invariant the sentinel depends on. If any real TBR is <= 0, supervision is
    silently dropping records.
    """
    import csv

    csv_path = data_root / "embeddings" / "longitudinal" / "tbr_features_NaF.csv"
    if not csv_path.exists():
        pytest.skip(f"TBR table not found: {csv_path}")

    bad = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("tbr2_p95_median", "")
            if raw in ("", "nan"):
                continue
            if float(raw) <= 0:
                bad.append((row["subject_id"], row["week"], raw))

    assert not bad, (
        f"{len(bad)} TBR values are <= 0 and collide with the -1 padding sentinel, so "
        f"they are dropped from both the regression mask and the evaluator: {bad[:10]}. "
        f"Switch the sentinel to NaN."
    )
