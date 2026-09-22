"""The existing PET-proxy target, with fixed horizons and explicit missing values."""
import math
import re

FUTURE_WEEKS = ["Week 15", "Week 18", "Week 20"]
DELTAS = [3, 6, 8]


def numeric_targets(record):
    numeric = record.get("answer_vqa_numeric") or {}
    values = numeric.get("tbr")
    kind = record.get("content_type")
    if values is None:
        if record.get("target_schema_version"):
            raise ValueError("Structured targets are required for validated records")
        # Read-only compatibility with historical JSON; current builders always
        # write numeric values and never depend on question wording for routing.
        values = [-1.0] * 4
        if kind in ("tbr", "combined") or (kind is None and "tbr" in record.get("question", "").lower()):
            for week, value in re.findall(r"Week\s+(\d+):\s*(\d+(?:\.\d+)?)",record.get("answer","")):
                if int(week) in DELTAS:
                    values[DELTAS.index(int(week))] = float(value)
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("TBR targets must have three horizons and a missing padding slot")
    values = [float(v) for v in values]
    if any(not math.isfinite(v) or (v < 0 and v != -1) for v in values) or values[3] != -1:
        raise ValueError(f"Invalid TBR target values: {values}")
    if kind == "genotype" and any(v >= 0 for v in values):
        raise ValueError("Genotype-only records cannot carry TBR supervision")
    return values


def unique_subject_targets(records):
    subjects = {}
    seen = set()
    for row in records:
        key = (row["pid"], str(row["qid"]))
        if key in seen:
            raise ValueError(f"Duplicate record: {key}")
        seen.add(key)
        label = (row.get("answer_vqa_numeric") or {}).get("genotype")
        if label not in (0, 1):
            raise ValueError(f"Invalid genotype label for {row['pid']}")
        entry = subjects.setdefault(row["pid"],dict(genotype=label,tbr=[-1.0]*4))
        if entry["genotype"] != label:
            raise ValueError(f"Conflicting genotype labels for {row['pid']}")
        for slot, value in enumerate(numeric_targets(row)):
            old = entry["tbr"][slot]
            if old >= 0 and value >= 0 and abs(old-value)>1e-6:
                raise ValueError(f"Conflicting targets for {row['pid']}, slot {slot}")
            if value >= 0:
                entry["tbr"][slot] = value
    return subjects


def fold_target_statistics(records):
    import numpy as np
    subjects = unique_subject_targets(records)
    mean, std, counts = [0.0]*4, [1.0]*4, [0]*4
    for slot in range(3):
        values = [v["tbr"][slot] for v in subjects.values() if v["tbr"][slot]>=0]
        counts[slot]=len(values)
        if values:
            mean[slot]=float(np.mean(values))
            # Population std: each observed subject/horizon contributes once.
            std[slot]=max(float(np.std(values,ddof=0)),1e-6) if len(values)>1 else 1.0
    return dict(mean=mean,std=std,counts=counts,subjects=sorted(subjects),ddof=0)
