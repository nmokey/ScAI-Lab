"""
F21 -- does GPT-paraphrased question wording still route and parse correctly?

create_mouse_vqa_dataset_from_openai.py (adapted from Arvind's BraTS script)
re-renders every VQA record's question/answer from a random paraphrase in
crump_aug_dataset.csv. Two things must survive that rewording:

  1. vlm/data/eval.py::_route() must still classify the record as
     genotype/tbr/combined -- previously (F15) it matched the literal words
     "status"/"tbr" in the question text, which the actual delivered
     paraphrase bank defeats for a large fraction of records (see docstring
     on _route()). It now prefers the record's own `content_type`.
  2. vlm/data/vqa_dataset.py::_tbr_targets and vlm/data/eval.py::_parse_tbr
     must still extract identical TBR numbers, since the multitask
     regression head is trained on them.

These tests exercise the actual crump_aug_dataset.csv shipped in the repo,
not a synthetic stand-in, so a future re-export of that CSV that breaks the
template contract fails here first.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.unit

CSV_PATH = Path(__file__).resolve().parents[2] / "crump_aug_dataset.csv"


def _aug_mod(scripts):
    return scripts("dataset/create_mouse_vqa_dataset_from_openai")


# ---------------------------------------------------------------------------
# Pool parsing
# ---------------------------------------------------------------------------

def test_pools_parse_and_cover_all_three_content_types(scripts):
    mod = _aug_mod(scripts)
    pools = mod.load_paraphrase_pools(CSV_PATH)
    assert set(pools) == {"genotype", "tbr", "combined"}
    for content_type, pairs in pools.items():
        assert len(pairs) >= 10, f"{content_type} pool suspiciously small: {len(pairs)}"


def test_every_paraphrase_carries_the_placeholders_its_content_type_needs(scripts):
    """
    A genotype paraphrase must offer {short_feature}/{status}; a tbr paraphrase
    {long_feature}/{n_weeks}/{trajectory}; combined, all five. If the CSV ever
    ships a malformed paraphrase missing one, augment_record would silently
    leave a literal "{status}" in rendered text instead of failing loudly.
    """
    mod = _aug_mod(scripts)
    pools = mod.load_paraphrase_pools(CSV_PATH)
    required = {
        "genotype": ["{short_feature}", "{status}"],
        "tbr": ["{long_feature}", "{n_weeks}", "{trajectory}"],
        "combined": ["{long_feature}", "{n_weeks}", "{trajectory}", "{short_feature}", "{status}"],
    }
    for content_type, placeholders in required.items():
        for q, a in pools[content_type]:
            combined_text = q + " " + a
            for ph in placeholders:
                assert ph in combined_text, (
                    f"{content_type} paraphrase missing {ph}: Q={q!r} A={a!r}"
                )


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def test_fill_template_replaces_every_placeholder(scripts):
    mod = _aug_mod(scripts)
    out = mod.fill_template("the {animal} {verb} {animal}", {"animal": "mouse", "verb": "chased"})
    assert out == "the mouse chased mouse"
    assert "{" not in out


def test_augment_record_fills_all_placeholders_for_every_real_paraphrase(scripts):
    """
    Render every paraphrase in the real CSV (not just a sample) against a
    representative record and check no literal "{...}" survives -- a stray
    placeholder would silently leak into a training example.
    """
    mod = _aug_mod(scripts)
    pools = mod.load_paraphrase_pools(CSV_PATH)

    base_records = {
        "genotype": {
            "content_type": "genotype",
            "question": "fixed q", "answer": "fixed a",
            "template_values": {"short_feature": "atherosclerosis", "status": "will develop"},
        },
        "tbr": {
            "content_type": "tbr",
            "question": "fixed q", "answer": "fixed a",
            "template_values": {"long_feature": "aortic TBR", "n_weeks": 8, "trajectory": "Week 3: 1.23, Week 8: 4.56"},
        },
        "combined": {
            "content_type": "combined",
            "question": "fixed q", "answer": "fixed a",
            "template_values": {
                "long_feature": "aortic TBR", "n_weeks": 8, "trajectory": "Week 3: 1.23, Week 8: 4.56",
                "short_feature": "atherosclerosis", "status": "will not develop",
            },
        },
    }

    import random
    for content_type, record in base_records.items():
        for q_template, a_template in pools[content_type]:
            rendered_q = mod.fill_template(q_template, record["template_values"])
            rendered_a = mod.fill_template(a_template, record["template_values"])
            assert "{" not in rendered_q and "}" not in rendered_q, f"leftover placeholder: {rendered_q!r}"
            assert "{" not in rendered_a and "}" not in rendered_a, f"leftover placeholder: {rendered_a!r}"


def test_records_without_template_values_pass_through_unchanged(scripts):
    """A record predating the template_values field must not be dropped or corrupted."""
    mod = _aug_mod(scripts)
    pools = mod.load_paraphrase_pools(CSV_PATH)
    record = {"content_type": "genotype", "question": "Q", "answer": "A", "pid": "NaF_WT_01"}
    import random
    out = mod.augment_record(record, pools, random.Random(0))
    assert out == record
    assert "question_fixed" not in out


# ---------------------------------------------------------------------------
# Routing survives paraphrasing (F21)
# ---------------------------------------------------------------------------

def test_route_uses_content_type_and_survives_every_real_paraphrase(scripts):
    """
    Render every single paraphrase for every content_type and confirm _route()
    recovers the right answer when given content_type -- this is the actual
    fix. Also records how many would have been misrouted under the old
    text-only heuristic, to keep that number visible rather than assumed.
    """
    mod = _aug_mod(scripts)
    from data.eval import _route

    pools = mod.load_paraphrase_pools(CSV_PATH)
    expected_route = {"genotype": "geno", "tbr": "tbr", "combined": "combined"}

    text_only_mismatches = 0
    for content_type, pairs in pools.items():
        for question_template, _answer_template in pairs:
            # Placeholders don't affect routing (it only reads literal words),
            # so route the template text directly.
            assert _route(question_template, content_type) == expected_route[content_type]
            if _route(question_template) != expected_route[content_type]:
                text_only_mismatches += 1

    total = sum(len(p) for p in pools.values())
    assert text_only_mismatches > 0, (
        "expected the real paraphrase bank to defeat the old text-only heuristic for "
        "at least some records -- if this now fails, re-check whether CSV content changed"
    )
    print(f"\n  {text_only_mismatches}/{total} paraphrases would have been misrouted by "
          f"question-text matching alone; content_type routing fixes all of them.")


# ---------------------------------------------------------------------------
# TBR numeric round-trip survives paraphrasing
# ---------------------------------------------------------------------------

def test_tbr_targets_identical_before_and_after_augmentation(scripts):
    mod = _aug_mod(scripts)
    from data.vqa_dataset import MouseTrajDataset

    pools = mod.load_paraphrase_pools(CSV_PATH)
    base_answer = "The predicted trajectory for aortic TBR - Week 3: 21.00, Week 6: 22.50, Week 8: 24.25."
    base_record = {
        "content_type": "tbr",
        "question": "Predict the mouse's trajectory of aortic TBR over the next 8 weeks?",
        "answer": base_answer,
        "template_values": {"long_feature": "aortic TBR", "n_weeks": 8,
                             "trajectory": "Week 3: 21.00, Week 6: 22.50, Week 8: 24.25"},
    }
    stub = object.__new__(MouseTrajDataset)
    base_targets = MouseTrajDataset._tbr_targets(stub, base_record).tolist()

    import random
    rng = random.Random(0)
    for _ in range(20):
        augmented = mod.augment_record(base_record, pools, rng)
        aug_targets = MouseTrajDataset._tbr_targets(stub, augmented).tolist()
        assert aug_targets == pytest.approx(base_targets), (
            f"TBR targets changed after paraphrasing: {augmented['answer']!r}"
        )
