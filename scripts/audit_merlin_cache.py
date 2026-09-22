"""Compare one corrected crop's current preprocessing to its path-keyed cache.

CPU only; reads existing files and prints measurements without refreshing the cache.
"""
import hashlib
import json
import pickle
from pathlib import Path

import torch
from merlin.data.monai_transforms import ImageTransforms


def main():
    source = '/data1/Processed_NIfTI_Test/mice/NaF_KO_09/week_15/ct_hi.nii.gz'
    key = hashlib.md5(pickle.dumps({'image': source}, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    cache = Path('/data1/Processed_NIfTI_Test/embeddings/merlin/cache') / (key + '.pt')
    cached = torch.load(cache, weights_only=False).cpu()
    fresh = ImageTransforms({'image': source})['image'].cpu()
    print(json.dumps(dict(source=source, cache=str(cache), cached_shape=list(cached.shape),
                          fresh_shape=list(fresh.shape),
                          max_abs_difference=float((cached-fresh).abs().max()),
                          mean_abs_difference=float((cached-fresh).abs().mean()),
                          different_voxels=int(((cached-fresh).abs()>1e-6).sum())), indent=2))


if __name__ == '__main__':
    main()
