"""
create_mouse_vqa_dataset_from_openai.py

Adapts create_brats_3d_vqa_dataset_from_openai.py (Arvind's BraTS
paraphrase-augmentation script) to the mouse atherosclerosis trajectory
dataset. Read this file alongside that one to see what changed and why.

Why the BraTS machinery doesn't carry over
-------------------------------------------
The BraTS script exists to solve a combinatorics problem: each segmentation
label has 4 independently-observable attributes (area/region/shape/
satellite) that get randomly combined into up to 15 question types per
label, so it organizes records by (seg_id, label, type), tracks which
attribute-combos have been used, and validates every sampled paraphrase
against the combo it's supposed to answer.

The mouse dataset has no such combo structure. create_mouse_traj_dataset.py
emits exactly one record per (subject, content_type) with content_type in
{"genotype", "tbr", "combined"}, and each of those maps 1:1 onto one of the
three OpenAI-generated paraphrase pools in crump_aug_dataset.csv (see
aug_config_mouse.yaml's `aug_prompts`, in the same order). So this script
drops the combo/organize/unorganize apparatus entirely: for each record it
just samples a random paraphrase template from its content_type's pool and
re-fills it.

Why records need `template_values`
-----------------------------------
BraTS's openai_df rows are pre-split into one row per generated question
(columns `transformed_q`/`transformed_a`), and ground-truth attribute values
are read from a separate `answer_vqa` field on each reference record. The
mouse CSV (crump_aug_dataset.csv) instead delivers one row per *prompt*, with
all ~50-65 generated Q/A pairs packed into a single blob per cell -- this
script parses those out. For the ground-truth values, rather than
regex-scraping them back out of already-rendered "the mouse will develop
atherosclerosis..." answer text (fragile: it would have to defeat both the
fixed-template renderer AND any future GPT rewording), create_mouse_traj_
dataset.py now stores the raw slot-fill values it already computed as
`template_values` on every record. This script only fills placeholders it
finds there.

Usage
-----
    python scripts/dataset/create_mouse_vqa_dataset_from_openai.py \\
        --vqa-dir /data1/Processed_NIfTI_Test/embeddings/vlm \\
        --aug-csv crump_aug_dataset.csv \\
        --seed 0

Reads mouse_{train,val,test,all}_vqa_traj.json from --vqa-dir (whichever of
these exist) and writes mouse_{split}_vqa_traj_aug.json alongside them (or
under --output-dir). Records without a populated `template_values` (e.g.
JSONs built before that field existed -- regenerate with
create_mouse_traj_dataset.py) are passed through unchanged, not dropped.
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

# The three fixed templates from aug_config_mouse.yaml's aug_prompts (and
# create_mouse_traj_dataset.py's format_*_qa), in the unrendered form the
# OpenAI prompt column carries them in. Used to identify which CSV row feeds
# which content_type without depending on row order.
FIXED_TEMPLATES = {
    "genotype": (
        "Q: What will be the eventual mouse status for {short_feature}? "
        "A: The eventual mouse status: the mouse {status} {short_feature}."
    ),
    "tbr": (
        "Q: Predict the mouse's trajectory of {long_feature} over the next {n_weeks} weeks? "
        "A: The predicted trajectory for {long_feature} - {trajectory}."
    ),
    "combined": (
        "Q: Predict the mouse's trajectory of {long_feature} over the next {n_weeks} weeks "
        "and the eventual status of {short_feature}? A: The predicted trajectory for "
        "{long_feature} - {trajectory}. The eventual status: the mouse {status} {short_feature}."
    ),
}

_QA_BLOCK_RE = re.compile(r"^Q:\s*(.*?)\s*\nA:\s*(.*)$", re.DOTALL)

SPLITS = ["train", "val", "test", "all"]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_pairs(blob):
    """Split one CSV cell's raw OpenAI output into (question, answer) pairs."""
    pairs = []
    for block in re.split(r"\n\s*\n", blob.strip()):
        block = block.strip()
        if not block:
            continue
        m = _QA_BLOCK_RE.match(block)
        if m is None:
            raise ValueError(f"Could not parse Q/A block out of crump_aug_dataset.csv: {block[:120]!r}...")
        pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


def load_paraphrase_pools(csv_path):
    """Returns dict: content_type -> list[(question_template, answer_template)]."""
    df = pd.read_csv(csv_path)
    pools = {}
    for _, row in df.iterrows():
        prompt  = row["question"]
        matches = [ct for ct, tmpl in FIXED_TEMPLATES.items() if tmpl in prompt]
        if len(matches) != 1:
            raise ValueError(
                f"{csv_path}: a row's prompt matched {len(matches)} known templates "
                f"(expected exactly 1). Prompt: {prompt[:200]!r}"
            )
        content_type = matches[0]
        if content_type in pools:
            raise ValueError(f"{csv_path}: two rows both matched content_type={content_type!r}")
        pools[content_type] = _parse_pairs(row["answer"])

    missing = set(FIXED_TEMPLATES) - set(pools)
    if missing:
        raise ValueError(f"{csv_path} is missing paraphrase pools for: {sorted(missing)}")
    return pools


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def fill_template(template, values):
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def augment_record(record, pools, rng):
    """
    Returns a copy of `record` with question/answer replaced by a random
    paraphrase filled from the record's own template_values. The fixed-
    template text is kept as question_fixed/answer_fixed for traceability.

    Records with no matching pool or no template_values (e.g. built before
    that field existed) pass through unchanged rather than being dropped --
    a partially-augmented dataset is still usable, a silently shrunk one
    is a data-leak risk on the next audit.
    """
    content_type    = record.get("content_type")
    template_values = record.get("template_values")
    pool = pools.get(content_type)
    if not pool or not template_values:
        return dict(record)

    q_template, a_template = rng.choice(pool)
    new_record = dict(record)
    new_record["question_fixed"] = record["question"]
    new_record["answer_fixed"]   = record["answer"]
    new_record["question"]       = fill_template(q_template, template_values)
    new_record["answer"]         = fill_template(a_template, template_values)
    return new_record


def augment_records(records, pools, seed):
    rng = random.Random(seed)
    return [augment_record(r, pools, rng) for r in records]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vqa-dir", required=True,
                   help="Directory holding mouse_{train,val,test,all}_vqa_traj.json, "
                        "the output of scripts/create_mouse_traj_dataset.py")
    p.add_argument("--aug-csv", default="crump_aug_dataset.csv")
    p.add_argument("--output-dir", default=None, help="Defaults to --vqa-dir")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args    = parse_args()
    vqa_dir = Path(args.vqa_dir)
    out_dir = Path(args.output_dir) if args.output_dir else vqa_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pools = load_paraphrase_pools(args.aug_csv)
    for content_type, pairs in pools.items():
        print(f"[i] {content_type:10s} pool: {len(pairs)} paraphrases")

    any_found = False
    for split in SPLITS:
        in_path = vqa_dir / f"mouse_{split}_vqa_traj.json"
        if not in_path.exists():
            print(f"[!] skip {in_path} (not found)")
            continue
        any_found = True

        with open(in_path) as f:
            records = json.load(f)

        augmented   = augment_records(records, pools, seed=args.seed)
        n_augmented = sum(1 for r in augmented if "question_fixed" in r)

        out_path = out_dir / f"mouse_{split}_vqa_traj_aug.json"
        with open(out_path, "w") as f:
            json.dump(augmented, f, indent=2)
        print(f"[+] {out_path.name}  ({n_augmented}/{len(augmented)} records paraphrased)")

    if not any_found:
        raise FileNotFoundError(
            f"No mouse_{{split}}_vqa_traj.json found under {vqa_dir}. "
            f"Run scripts/create_mouse_traj_dataset.py first."
        )


if __name__ == "__main__":
    main()
