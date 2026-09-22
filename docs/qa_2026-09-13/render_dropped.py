"""Render every quadrant of m54406 and m54520 from the session volume, labelled with what the old code wrote there."""
import numpy as np, SimpleITK as sitk, matplotlib, sys; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/data1/Processed_NIfTI_Test"
sessions=[("m54406","15","NaF",{1:"→ written as NaF_KO_09",2:"→ written as NaF_KO_10",3:"→ written as NaF_KO_11",4:"DROPPED"}),
          ("m54520","15","FDG",{1:"→ written as FDG_WT_01",2:"→ written as FDG_WT_02",3:"DROPPED",4:"DROPPED"})]
fig,axes=plt.subplots(2,4,figsize=(14,7.5))
for r,(sid,wk,coh,lab) in enumerate(sessions):
    img=sitk.ReadImage(f"{R}/sessions/week_{wk}/{coh}/{sid}/ct_hi.nii.gz"); a=sitk.GetArrayFromImage(img)  # z,y,x
    size=img.GetSize(); cx,cy,_=img.TransformIndexToPhysicalPoint([size[0]//2,size[1]//2,size[2]//2])
    # physical coords per voxel column/row
    px=np.array([img.TransformIndexToPhysicalPoint([i,0,0])[0] for i in range(size[0])])
    py=np.array([img.TransformIndexToPhysicalPoint([0,j,0])[1] for j in range(size[1])])
    left=px>cx; lower=py<cy
    quads={1:(lower,left),2:(lower,~left),3:(~lower,left),4:(~lower,~left)}
    for q,(ym,xm) in quads.items():
        sub=a[:, ym][:, :, xm]; bone=sub>300
        ax=axes[r,q-1]; ax.imshow(bone.sum(1),cmap="gray",aspect="auto")  # side (coronal) view: skeleton obvious if mouse
        ax.set_title(f"{sid} pos{q}  {lab[q]}\nbone voxels={int(bone.sum()):,}",fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Side-view bone projections of each quadrant (a mouse shows a skeleton; a tube shows a curve)",fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=95); print("saved")
