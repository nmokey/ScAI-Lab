"""Render full-bed CT with the segmentation's quadrant divider and the manifest's mouse labels."""
import csv, collections, os, sys
import numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/data1/Processed_NIfTI_Test"; OUT=sys.argv[1]
picks=[("m54225","12","NaF"),("m54226","12","NaF"),("m54227","12","NaF"),("m54406","15","NaF"),("m54244","12","NaF")]
rows=list(csv.DictReader(open(f"{ROOT}/mouse_manifest.csv")))
by=collections.defaultdict(list)
for r in rows: by[(r["session_id"],r["week"],r["cohort"])].append((int(r["crop_position"]),r["mouse_id"],r["genotype"]))
fig,axes=plt.subplots(1,len(picks),figsize=(5*len(picks),5.5))
for ax,(sid,wk,coh) in zip(axes,picks):
    p=f"{ROOT}/sessions/week_{wk}/{coh}/{sid}/ct_hi.nii.gz"
    if not os.path.exists(p): ax.set_title(f"{sid}: no session file"); ax.axis("off"); continue
    img=sitk.ReadImage(p); arr=sitk.GetArrayFromImage(img)  # z,y,x
    # bone MIP along z, exactly the tissue the segmenter uses
    bone=(arr>300); mip=bone.sum(0).astype(float)
    size=img.GetSize(); o=np.array(img.GetOrigin())
    cx,cy,_=img.TransformIndexToPhysicalPoint([size[0]//2,size[1]//2,size[2]//2])
    # physical coords of image axes for extent; 'higher physical X = visually left' per the script
    xs=[img.TransformIndexToPhysicalPoint([i,0,0])[0] for i in (0,size[0]-1)]
    ys=[img.TransformIndexToPhysicalPoint([0,j,0])[1] for j in (0,size[1]-1)]
    ax.imshow(mip,cmap="gray",extent=[xs[0],xs[1],ys[0],ys[1]],origin="lower",aspect="equal")
    ax.axvline(cx,color="cyan",lw=1); ax.axhline(cy,color="cyan",lw=1)
    if xs[0]<xs[1]: ax.invert_xaxis()   # make higher-X appear on the left, matching the convention
    # quadrant label positions: pos1 lower-left(high X, low Y) pos2 lower-right pos3 upper-left pos4 upper-right
    xr=abs(xs[1]-xs[0])/4; yr=abs(ys[1]-ys[0])/4
    qpos={1:(cx+xr,cy-yr),2:(cx-xr,cy-yr),3:(cx+xr,cy+yr),4:(cx-xr,cy+yr)}
    have={pos:(m,g) for pos,m,g in by[(sid,wk,coh)]}
    for pos,(x,y) in qpos.items():
        lab=f"pos{pos}\n{have[pos][0]}" if pos in have else f"pos{pos}\n(none in manifest)"
        col="lime" if pos in have else "red"
        ax.text(x,y,lab,color=col,ha="center",va="center",fontsize=9,fontweight="bold",bbox=dict(fc="black",alpha=.6,ec="none"))
    ax.set_title(f"{sid}  wk{wk} {coh}  manifest n={len(have)}",fontsize=10); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/session_quadrants.png",dpi=110); print("saved")
