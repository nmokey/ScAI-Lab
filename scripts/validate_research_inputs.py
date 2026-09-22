"""Validate current structured targets, baseline features, crop identity and provenance."""
import argparse
import csv
import json
import sys
from collections import defaultdict,Counter
from pathlib import Path

import numpy as np
import yaml
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,file_sha256
from utils.run_contract import validate_records
from utils.target_contract import unique_subject_targets


def validate_crop_identities(crops, raw, overrides):
    raw_by_id={r['scan_id']:r for r in raw}
    grouped=defaultdict(list)
    for row in crops:grouped[row['ct_source_scan']].append(row)
    checks=[]
    for scan,rows in grouped.items():
        source=raw_by_id[scan]
        expected=[int(v) for v in source['mouse_nums'].split(';')]
        rows=sorted(rows,key=lambda r:int(r['crop_position']))
        positions=[int(r['crop_position']) for r in rows]
        if len(set(positions))!=len(positions) or not set(positions)<={1,2,3,4}:
            raise ValueError(f'Invalid crop positions: {scan}')
        if scan in overrides:
            mapping=overrides[scan]['positions']
            actual=[int(mapping[pos]) for pos in positions]
        elif len(expected)==4:
            actual=[expected[pos-1] for pos in positions]
        else:
            if len(rows)!=len(expected):raise ValueError(f'Ambiguous partial scan occupancy: {scan}')
            actual=expected
        found=[int(r['mouse_num']) for r in rows]
        if len(set(found))!=len(found) or actual!=found or not set(found)<=set(expected):
            raise ValueError(f'Filename/override identity mismatch: {scan}')
        checks.append(dict(scan=scan,positions=positions,mouse_numbers=found,override=scan in overrides))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='/data1/Processed_NIfTI_Test')
    p.add_argument('--dataset',default='/data1/Processed_NIfTI_Test/embeddings/vlm/validated_20260914')
    p.add_argument('--report',required=True)
    args=p.parse_args();root=Path(args.data_root);dataset=Path(args.dataset);repo=Path(__file__).resolve().parents[1]
    records=json.loads((dataset/'mouse_all_vqa_traj.json').read_text());subjects=validate_records(records)
    archive=root/'embeddings/raddino/raddino_embeddings.npz';data=np.load(archive,allow_pickle=True)
    keys=list(zip(data['subject_ids'].astype(str),data['weeks'].astype(str)))
    if len(keys)!=len(set(keys)) or not np.isfinite(data['embeddings']).all():raise ValueError('Invalid or duplicate encoder rows')
    lookup=dict(zip(keys,data['embeddings']))
    errors=[]
    for row in records:
        baseline=load_file(row['embedding_path_ts0'])['embeddings'].numpy().reshape(-1)
        error=float(np.max(np.abs(baseline-lookup[(row['pid'],'Week 12')])));errors.append(error)
        if error!=0:raise ValueError(f'Baseline features differ: {row["pid"]}')
    new=unique_subject_targets(records)
    old_records=json.loads((root/'embeddings/vlm/mouse_all_vqa_traj.json').read_text())
    old=unique_subject_targets(old_records)
    if new!=old:raise ValueError('The repaired data changed eligible subjects or numerical targets')
    raw=list(csv.DictReader(open(repo/'manifest.csv')))
    crops=list(csv.DictReader(open(root/'mouse_manifest.csv')))
    crop_keys=[(r['mouse_id'],'Week '+r['week']) for r in crops]
    if len(crop_keys)!=len(set(crop_keys)) or set(crop_keys)!=set(keys):raise ValueError('Crop/embedding populations disagree')
    overrides=yaml.safe_load((repo/'quadrant_overrides.yaml').read_text())
    identity_checks=validate_crop_identities(crops,raw,overrides)
    proxy=json.loads((dataset/'proxy_recomputation.json').read_text())
    if proxy['changed_over_1e6']!=0:raise ValueError('Stored targets differ from recomputed current PET crops')
    for path,digest in proxy['source_sha256'].items():
        if file_sha256(path)!=digest:raise ValueError(f'PET source changed after validation: {path}')
    source_hashes={str(path):file_sha256(path) for path in data['paths'].astype(str)}
    input_manifest=json.loads((dataset/'input_manifest.json').read_text())
    if input_manifest['embeddings_sha256']!=file_sha256(archive):raise ValueError('Encoder archive changed')
    for key,path in [('mouse_manifest_sha256',root/'mouse_manifest.csv'),
                     ('overrides_sha256',repo/'quadrant_overrides.yaml'),
                     ('builder_sha256',repo/'scripts/create_mouse_traj_dataset.py'),
                     ('tbr_csv_sha256',Path(input_manifest['tbr_csv_path']))]:
        if input_manifest[key]!=file_sha256(path):raise ValueError(f'Input provenance changed: {path}')
    if proxy['recomputed_csv_sha256']!=input_manifest['tbr_csv_sha256']:
        raise ValueError('Dataset targets do not use the recomputed CSV')
    if proxy['original_csv_sha256']!=file_sha256(root/'embeddings/longitudinal/tbr_features_NaF.csv'):
        raise ValueError('Original proxy CSV changed after recomputation')
    groups={r['pid']:r['acquisition_group'] for r in records}
    report=dict(status='passed',records=len(records),subjects=len(subjects),embedding_rows=len(keys),
                baseline_max_abs_error=max(errors),numerical_targets_equal_historical=True,
                proxy_recomputed_rows=len(proxy['results']),proxy_max_abs_error=proxy['max_abs_difference'],
                acquisition_group_sizes=dict(Counter(groups.values())),
                identity_rule='Filename order lower-left, lower-right, upper-left, upper-right; explicit tube/phantom overrides',
                identity_checks=identity_checks,ct_source_sha256=source_hashes,
                data_sha256=file_sha256(dataset/'mouse_all_vqa_traj.json'),
                limitations=['Filename mapping is an accepted heuristic, not externally confirmed identity.',
                             'Proxy computation is verified; aortic/histology biological validity is not established.'])
    atomic_json(args.report,report);print(json.dumps({k:v for k,v in report.items() if k not in ('ct_source_sha256','identity_checks')},indent=2))


if __name__=='__main__':main()
