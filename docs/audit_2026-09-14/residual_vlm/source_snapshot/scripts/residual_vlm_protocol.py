"""Strict routing/provenance checks for the authorized residual-input VLM arm."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from stratified_vlm_protocol import make_spec, validate_split
from residual_rollout_forecast import SETTINGS, verify_fold
from build_nested_forecasts import embedding_lookup
from utils.research_io import file_sha256, object_sha256

REPO = Path(__file__).resolve().parents[1]
FORECAST_REPORT = REPO / 'docs/audit_2026-09-14/residual_rollout_experiment'
ADDED_SOURCES = ('residual_vlm_protocol.py', 'run_residual_vlm.py', 'evaluate_residual_vlm.py',
                 'residual_rollout_forecast.py')


def assert_matched(spec, original):
    current = copy.deepcopy(spec); previous = copy.deepcopy(original)
    current.pop('residual_forecast_protocol_sha256')
    for name in ADDED_SOURCES:
        current['source_sha256'].pop('scripts/' + name)
    for value in (current, previous):
        value.pop('nested_index_sha256')
        value['config']['data'].pop('nested_forecast_root')
    if current != previous:
        raise ValueError('Residual VLM differs from the original longitudinal run beyond the authorized forecast inputs/routing')


def verify_residual_inputs(root, fold, split, records):
    root = Path(root)
    protocol = json.loads((FORECAST_REPORT / 'PROTOCOL.json').read_text())
    if root != Path(protocol['forecast_output']) or split != protocol['population']:
        raise ValueError('Residual VLM forecasts belong to a different experiment or split')
    if file_sha256(protocol['embeddings_path']) != protocol['embeddings_sha256']:
        raise ValueError('Residual source embeddings changed')
    archive = np.load(protocol['embeddings_path'], allow_pickle=True)
    lookup = embedding_lookup(archive['embeddings'], archive['subject_ids'], archive['weeks'])
    index = json.loads((root / 'index.json').read_text())
    if index['protocol_sha256'] != object_sha256(protocol):
        raise ValueError('Residual forecast index has the wrong protocol')
    if file_sha256(root / fold['key'] / 'manifest.json') != index['folds'][fold['key']]:
        raise ValueError('Residual forecast manifest changed')
    manifest = verify_fold(root / fold['key'], lookup, split, fold, SETTINGS)
    if manifest['experiment_protocol_sha256'] != object_sha256(protocol):
        raise ValueError('Residual forecast fit belongs to the wrong experiment')
    # Exercise the exact VLM dataset loader independently of tokenizer/model setup.
    from data.vqa_dataset import MouseTrajDataset
    loader = object.__new__(MouseTrajDataset)
    loader.predicted_emb_dir = str(root / fold['key']); loader.img_tokens = 4
    unique = {r['pid']: r for r in records}
    if set(unique) != set(split['subjects']):
        raise ValueError('Residual VLM population mismatch')
    for sid, row in unique.items():
        baseline = load_file(row['embedding_path_ts0'])['embeddings'].contiguous().numpy()
        if hashlib.sha256(baseline.tobytes()).hexdigest() != manifest['baseline_sha256'][sid]:
            raise ValueError('Residual forecast and VLM baseline differ')
        expected = np.stack([baseline[0]] + [np.load(root / fold['key'] / f'{sid}_ts{k}.npy', allow_pickle=False) for k in (1, 2, 3)])
        np.testing.assert_array_equal(loader._load_embedding(row).numpy(), expected)
    return manifest


def make_residual_spec(params, records, truth, split, original_spec):
    validate_split(split, records)
    if params['data']['img_tokens'] != 4 or params['data'].get('predicted_emb_dir'):
        raise ValueError('Residual arm requires the nested four-token route')
    spec = make_spec(params, records, truth, split)
    spec['residual_forecast_protocol_sha256'] = file_sha256(FORECAST_REPORT / 'PROTOCOL.json')
    for name in ADDED_SOURCES:
        spec['source_sha256']['scripts/' + name] = file_sha256(REPO / 'scripts' / name)
    assert_matched(spec, original_spec)
    return spec
