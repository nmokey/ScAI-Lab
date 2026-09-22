"""Verify corrected RAD-DINO crops and the repaired Merlin content-addressed cache."""
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vlm'))
from utils.research_io import atomic_json,file_sha256
from utils.image_cache import image_cache_key


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    args=p.parse_args();root=Path('/data1/Processed_NIfTI_Test');reports={}
    from get_raddino_embeddings import RadDino,embed_volume
    archive=np.load(root/'embeddings/raddino/raddino_embeddings.npz',allow_pickle=True)
    lookup={(str(s),str(w)):(e,str(path)) for s,w,e,path in zip(archive['subject_ids'],archive['weeks'],archive['embeddings'],archive['paths'])}
    corrected=[('NaF_KO_09','Week 15'),('NaF_KO_10','Week 15'),('NaF_KO_11','Week 15'),('FDG_WT_01','Week 15'),('FDG_WT_02','Week 15')]
    encoder=RadDino().to('cuda').eval();comparisons=[]
    for key in corrected:
        expected,path=lookup[key];first=embed_volume(path,encoder,32,-160,240)
        repeated=embed_volume(path,encoder,32,-160,240)
        repeat_error=float(abs(first-repeated).max());artifact_error=float(abs(first-expected).max())
        if repeat_error!=0 or artifact_error!=0:raise ValueError(f'RAD-DINO source verification failed: {key}, repeat={repeat_error}, artifact={artifact_error}')
        comparisons.append(dict(subject=key[0],week=key[1],source_sha256=file_sha256(path),repeat_error=repeat_error,artifact_error=artifact_error))
        print(comparisons[-1],flush=True)
    reports['corrected_raddino_crops']=comparisons
    del encoder;torch.cuda.empty_cache()
    from merlin.data import DataLoader
    from merlin.data.monai_transforms import ImageTransforms
    path=lookup[corrected[0]][1]
    with tempfile.TemporaryDirectory(prefix='scai-merlin-cache-v2-') as cache:
        loader=DataLoader(datalist=[{'image':path}],cache_dir=cache,batchsize=1,shuffle=False,num_workers=0)
        loader.dataset.hash_func=lambda item:image_cache_key(item,'real-transform-cache-regression-v2')
        initial=loader.dataset[0]['image'].cpu()
        cached=loader.dataset[0]['image'].cpu()
        fresh=ImageTransforms({'image':path})['image'].cpu()
        assert torch.equal(initial,cached) and torch.equal(cached,fresh)
        reports['merlin_cache']=dict(source=path,initial_cached_fresh_max_error=0.,new_cache_files=len(list(Path(cache).glob('*.pt'))))
    reports['status']='passed';atomic_json(args.report,reports);print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
