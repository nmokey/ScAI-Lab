"""Small provenance and atomic-I/O utilities shared by the validated pipeline."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def verify_forecast_manifest(directory, outer_subject, subjects):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    subjects = set(subjects)
    if manifest.get("format_version") != 1 or manifest.get("outer_subject") != outer_subject:
        raise ValueError("Forecast manifest belongs to a different protocol or outer fold")
    if set(manifest["subjects"]) != subjects or set(manifest["queries"]) != subjects:
        raise ValueError("Forecast population differs from the VLM population")
    for query, fit_key in manifest["queries"].items():
        fit = manifest["fits"][fit_key]
        members = set(fit["fit_subjects"])
        if not members or not members <= subjects - {outer_subject, query}:
            raise ValueError(f"Forecast leakage for {query}: {members}")
        if query == outer_subject and members != subjects - {outer_subject}:
            raise ValueError("Test forecasts must use the complete outer-training population")
        if fit.get("genotype_conditioning") is not False or fit.get("export_mode") != "baseline_rollout":
            raise ValueError("Only genotype-free baseline rollouts are permitted")
        for tag in ("ts1", "ts2", "ts3"):
            name = f"{query}_{tag}.npy"
            if file_sha256(directory / name) != manifest["files"].get(name):
                raise ValueError(f"Forecast file missing or changed: {name}")
    return manifest
