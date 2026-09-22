"""Build fixed inner-cross-fitted VLM tokens separately inside every outer holdout.

Uses the existing MLP, cosine objective, 300 epochs and normalized recurrent
rollout. No genotype conditioning or observed future query scans are used.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vlm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.research_io import atomic_json, file_sha256, object_sha256, verify_forecast_manifest
from train_longitudinal import build_pairs, train_fold, make_input, WEEK_ORDER


DEFAULT_SETTINGS = dict(epochs=300, hidden=512, lr=0.001, seed=0, inner_folds=5)


def embedding_lookup(embeddings, subject_ids, weeks):
    lookup = {}
    for value, sid, week in zip(embeddings, subject_ids, weeks):
        key = (str(sid), str(week))
        if key in lookup or not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
            raise ValueError(f"Duplicate or invalid embedding: {key}")
        lookup[key] = np.asarray(value, dtype=np.float32)
    return lookup


def fit_digest(lookup, members):
    digest = hashlib.sha256()
    for key in sorted(lookup):
        if key[0] in members:
            digest.update(json.dumps(key).encode())
            digest.update(lookup[key].tobytes())
    return digest.hexdigest()


def outer_forecasts(lookup, subjects, outer_subject, settings=None, device="cpu"):
    settings = dict(DEFAULT_SETTINGS, **(settings or {}))
    subjects = sorted(subjects)
    if len(subjects) != len(set(subjects)) or outer_subject not in subjects:
        raise ValueError("Invalid outer population")
    for sid in subjects:
        if (sid, "Week 12") not in lookup:
            raise ValueError(f"Missing observed baseline for {sid}")
    training = sorted(set(subjects) - {outer_subject})
    folds = settings["inner_folds"]
    if not 2 <= folds <= len(training):
        raise ValueError("Invalid inner fold count")
    partitions = []
    for fit_idx, query_idx in KFold(folds, shuffle=True, random_state=settings["seed"]).split(training):
        partitions.append(([training[i] for i in fit_idx], [training[i] for i in query_idx]))
    partitions.append((training, [outer_subject]))
    output, fits, queries = {}, {}, {}
    for index, (members, query_subjects) in enumerate(partitions):
        keys = sorted(key for key in lookup if key[0] in members)
        arrays = np.stack([lookup[key] for key in keys])
        X, Y, _, _, _ = build_pairs(arrays, np.array([k[0] for k in keys]),
                                    np.array([k[1] for k in keys]), use_conditioning=True, use_genotype=False)
        # Constant seed per fit role; no seed or conditioning derived from labels.
        seed = int(settings["seed"]) + index
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_fold(X, Y, X.shape[1], Y.shape[1], settings["hidden"],
                           settings["lr"], settings["epochs"], torch.device(device))
        norm = float(np.linalg.norm(Y, axis=1).mean())
        key = f"fit_{index}"
        fits[key] = dict(fit_subjects=members, training_digest=fit_digest(lookup, set(members)),
                         training_pair_count=len(X), norm_reference=norm, seed=seed,
                         genotype_conditioning=False, export_mode="baseline_rollout")
        for sid in query_subjects:
            cur = lookup[(sid, "Week 12")].copy()
            queries[sid] = key
            for slot, (w0, w1) in enumerate(zip(WEEK_ORDER, WEEK_ORDER[1:]), 1):
                x = make_input(cur, sid.split("_")[0], None, w0, w1, True, False)
                with torch.no_grad():
                    cur = model(torch.tensor(x[None], device=device)).cpu().numpy()[0]
                magnitude = float(np.linalg.norm(cur))
                if not np.isfinite(cur).all() or magnitude <= 0:
                    raise ValueError(f"Invalid forecast for {sid}/{w1}")
                cur = np.asarray(cur * (norm / magnitude), dtype=np.float32)
                output[f"{sid}_ts{slot}.npy"] = cur.copy()
        del model
    manifest = dict(format_version=1, outer_subject=outer_subject, subjects=subjects,
                    settings=settings, fits=fits, queries=queries,
                    baseline_sha256={sid:hashlib.sha256(lookup[(sid,"Week 12")].tobytes()).hexdigest() for sid in subjects})
    return output, manifest


def export_outer(lookup, subjects, sid, root, settings, device):
    import fcntl
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    directory = root / sid
    # File lock makes independently resumed exporters safe; never replace a valid
    # completed directory with a different protocol under the same path.
    with open(root / f".{sid}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = dict(settings=settings, training_digest=fit_digest(lookup,set(subjects)-{sid}),
                       baselines={q:hashlib.sha256(lookup[(q,"Week 12")].tobytes()).hexdigest() for q in subjects},
                       code_sha256={str(p.name):file_sha256(p) for p in
                                    (Path(__file__),Path(__file__).with_name("train_longitudinal.py"))})
        if directory.exists():
            existing = verify_forecast_manifest(directory, sid, subjects)
            if existing.get("request_sha256") != object_sha256(request):
                raise ValueError(f"Stale forecast cache: {directory}; use a new output root")
            return existing
        arrays, manifest = outer_forecasts(lookup, subjects, sid, settings, device)
        manifest["request_sha256"] = object_sha256(request)
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{sid}-") as tmp:
            temp = Path(tmp)
            for name, value in arrays.items():
                np.save(temp / name, value)
            manifest["files"] = {name:file_sha256(temp/name) for name in arrays}
            atomic_json(temp / "manifest.json", manifest)
            verify_forecast_manifest(temp, sid, subjects)
            # Move the directory out of TemporaryDirectory's cleanup scope.
            os.rename(temp, directory)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings",required=True); parser.add_argument("--records",required=True)
    parser.add_argument("--output-dir",required=True); parser.add_argument("--device",default="cpu")
    parser.add_argument("--outer-subject"); parser.add_argument("--epochs",type=int,default=300)
    args=parser.parse_args()
    data=np.load(args.embeddings,allow_pickle=True)
    lookup=embedding_lookup(data["embeddings"],data["subject_ids"],data["weeks"])
    subjects=sorted({row["pid"] for row in json.loads(Path(args.records).read_text())})
    settings=dict(DEFAULT_SETTINGS,epochs=args.epochs)
    for index,sid in enumerate([args.outer_subject] if args.outer_subject else subjects):
        manifest=export_outer(lookup,subjects,sid,args.output_dir,settings,args.device)
        print(f"Completed {sid}: {len(manifest['files'])} tokens, {len(manifest['fits'])} fits",flush=True)
    if not args.outer_subject:
        atomic_json(Path(args.output_dir)/"index.json",dict(format_version=1,subjects=subjects,
                    settings=settings,folds={s:file_sha256(Path(args.output_dir)/s/"manifest.json") for s in subjects}))


if __name__=="__main__": main()
