"""
Is Week-12-vs-Week-20 separation anatomy (growth) or acquisition (drift)?
Per crop: statistics of AIR-ONLY voxels (nothing biological there) and of the mouse body.
If air-only stats separate W12 from W20, the scanner changed; if only body stats do, the mice did.
"""
import numpy as np, SimpleITK as sitk, csv, sys, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
sys.path.insert(0,"/home/ryab/ScAI-Lab/scripts"); from eval_stats import join_mouse_group_ids
rows=[r for r in csv.DictReader(open("/data1/Processed_NIfTI_Test/mouse_manifest.csv")) if r["week"] in ("12","20")]
feats=[];meta=[]
for i,r in enumerate(rows):
    p=f"/data1/Processed_NIfTI_Test/mice/{r['mouse_id']}/week_{r['week']}/ct_hi.nii.gz"
    img=sitk.Shrink(sitk.ReadImage(p),[4,4,4]); a=sitk.GetArrayFromImage(img).astype(np.float32)
    # air = corner blocks of the crop, far from the mouse and holder
    z,y,x=a.shape; c=[a[:z//8,:y//8,:x//8],a[:z//8,:y//8,-x//8:],a[-z//8:,:y//8,:x//8],a[-z//8:,:y//8,-x//8:]]
    air=np.concatenate([b.ravel() for b in c]); air=air[air<-700]
    body=a[a>-400]; bone=a[a>300]
    feats.append([air.mean(),air.std(),np.percentile(air,5),np.percentile(air,95),   # acquisition-only
                  body.mean(),body.std(),len(body),len(bone),np.percentile(body,50)])  # body
    meta.append((r["mouse_id"],f"Week {r['week']}"))
    if i%20==0: print(f"  {i}/{len(rows)}",flush=True)
F=np.array(feats); sid=np.array([m[0] for m in meta]); wk=np.array([m[1] for m in meta])
grp=join_mouse_group_ids(sid,wk,"/data1/Processed_NIfTI_Test/mouse_manifest.csv"); y=(wk=="Week 20").astype(int)
def logo(Xf):
    yt,ys=[],[]
    for tr,te in LeaveOneGroupOut().split(Xf,y,grp):
        if len(set(y[tr]))<2: continue
        mu,sd=Xf[tr].mean(0),Xf[tr].std(0); sd[sd<1e-8]=1
        c=LogisticRegression(max_iter=3000).fit((Xf[tr]-mu)/sd,y[tr]); ys+=list(c.predict_proba((Xf[te]-mu)/sd)[:,1]); yt+=list(y[te])
    return roc_auc_score(yt,ys)
print(f"\nW12 vs W20, leave-one-mouse-group-out, n={len(y)} crops")
print(f"  AIR-only features (mean/sd/p5/p95 of air HU)      : AUC={logo(F[:,:4]):.3f}   <- scanner, not mouse")
print(f"  BODY features (HU stats, tissue+bone voxel counts): AUC={logo(F[:,4:]):.3f}   <- mouse (growth) and/or scanner")
print(f"  body/bone voxel COUNTS only (size)                : AUC={logo(F[:,6:8]):.3f}   <- growth")
for name,j in [("air mean HU",0),("air sd HU",1),("tissue voxels",6),("bone voxels",7)]:
    a,b=F[y==0,j],F[y==1,j]; print(f"  {name:14} W12 {a.mean():10.1f} ± {a.std():7.1f}   W20 {b.mean():10.1f} ± {b.std():7.1f}")
