"""Exact group-label permutation with full probe refits on current RAD-DINO data.

Read-only, CPU only. Uses the production probe, with its printing suppressed.
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from evaluate_embeddings import _loso_logistic
from eval_stats import join_mouse_group_ids, permutation_refit_p


def main():
    root = Path('/data1/Processed_NIfTI_Test')
    d = np.load(root / 'embeddings/raddino/raddino_embeddings.npz', allow_pickle=True)
    groups = join_mouse_group_ids(d['subject_ids'], d['weeks'], root / 'mouse_manifest.csv')
    y = np.array([int('_KO_' in s) for s in d['subject_ids']])
    count = 0

    def fit_predict(X, labels, g):
        nonlocal count
        count += 1
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, yt, scores = _loso_logistic(X, labels, g, 'audit genotype', return_proba=True)
        if count % 20 == 0:
            print(f'Refit progress {count}/121', flush=True)
        return np.array(yt), np.array(scores)

    obs, p, ci, meta = permutation_refit_p(fit_predict, d['embeddings'], y, groups)
    print(json.dumps(dict(n=len(y), groups=len(set(groups)), auc=obs,
                          exact_refit_p=p, refit_null95=ci, permutation=meta), indent=2))


if __name__ == '__main__':
    main()
