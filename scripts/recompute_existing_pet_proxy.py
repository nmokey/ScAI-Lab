"""Recompute the existing PET-2 endpoint without changing its eligible population.

Other historical CSV columns are copied for compatibility, not revalidated.
Use a separate output directory; this command refuses to overwrite the source CSV.
"""
import argparse
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vlm'))
from extract_tbr_features import tbr_strategy_2
from utils.research_io import atomic_json, file_sha256


def recompute(manifest, source_csv, output_dir):
    manifest, source_csv, output_dir = map(Path, (manifest, source_csv, output_dir))
    destination = output_dir / 'tbr_features_NaF.csv'
    if destination.resolve() == source_csv.resolve():
        raise ValueError('Recomputation must preserve the original CSV')
    with source_csv.open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        old = list(reader)
    lookup = {(r['subject_id'], r['week']): r for r in old}
    if len(lookup) != len(old) or not old:
        raise ValueError('Source targets must be nonempty and unique')
    with manifest.open() as stream:
        crops = list(csv.DictReader(stream))
    results, hashes, new_rows, seen = [], {}, [], set()
    for crop in crops:
        key = (crop['mouse_id'], 'Week ' + crop['week'])
        if crop['cohort'] != 'NaF' or key not in lookup:
            continue
        if key in seen or not crop['pet_nifti']:
            raise ValueError(f'Duplicate crop or missing PET for target: {key}')
        seen.add(key)
        path = Path(crop['pet_nifti'])
        digest = file_sha256(path)
        values = tbr_strategy_2(nib.load(path).get_fdata())
        if file_sha256(path) != digest:
            raise ValueError(f'PET crop changed during recomputation: {path}')
        previous = float(lookup[key]['tbr2_p95_median'])
        current = float(values['tbr2_p95_median'])
        if not np.isfinite([previous, current]).all():
            raise ValueError(f'Nonfinite proxy: {key}')
        results.append(dict(subject=key[0], week=key[1], old=previous,
                            recomputed=current, abs_difference=abs(previous-current)))
        new_rows.append(dict(lookup[key], **values))
        hashes[str(path)] = digest
    if seen != set(lookup):
        raise ValueError(f'Missing current PET crops: {sorted(set(lookup)-seen)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(new_rows)
    with tempfile.NamedTemporaryFile(mode='w', dir=output_dir, delete=False, newline='') as stream:
        temporary = Path(stream.name)
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    report = dict(results=results, source_sha256=hashes,
                  original_csv_sha256=file_sha256(source_csv),
                  recomputed_csv_sha256=file_sha256(destination),
                  extractor_sha256=file_sha256(Path(__file__).with_name('extract_tbr_features.py')),
                  manifest_sha256=file_sha256(manifest),
                  changed_over_1e6=sum(r['abs_difference'] > 1e-6 for r in results),
                  max_abs_difference=max(r['abs_difference'] for r in results),
                  validated_columns=list(values),
                  copied_unvalidated_columns=[f for f in fields if f not in values])
    atomic_json(output_dir / 'proxy_recomputation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    report = recompute(args.manifest, args.source_csv, args.output_dir)
    print(f"Recomputed {len(report['results'])} targets; maximum absolute change {report['max_abs_difference']}")


if __name__ == '__main__':
    main()
