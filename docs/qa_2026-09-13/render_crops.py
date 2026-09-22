import numpy as np, SimpleITK as sitk, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
R="/data1/Processed_NIfTI_Test/mice"
crops=[("NaF_KO_09","week_15","m54406 pos1 (mouse-like extent)"),("NaF_KO_10","week_15","m54406 pos2 (mouse-like)"),("NaF_KO_11","week_15","m54406 pos3 (NARROW 36x16mm)"),
       ("FDG_WT_01","week_15","m54520 pos1 (mouse-like)"),("NaF_KO_11","week_12","KO_11 wk12 for comparison"),("NaF_KO_11","week_18","KO_11 wk18 for comparison")]
fig,axes=plt.subplots(2,len(crops),figsize=(3.2*len(crops),6.5))
for i,(m,w,t) in enumerate(crops):
    p=f"{R}/{m}/{w}/ct_hi.nii.gz"
    try:
        a=sitk.GetArrayFromImage(sitk.ReadImage(p)); bone=a>300
        axes[0,i].imshow(bone.sum(0),cmap="gray"); axes[0,i].set_title(f"{m} {w}\n{t}",fontsize=8)
        axes[1,i].imshow(bone.sum(1),cmap="gray",aspect="auto"); axes[1,i].set_title(f"side view  shape={a.shape}",fontsize=8)
    except Exception as e:
        axes[0,i].set_title(f"{m} {w}\n{e}",fontsize=7)
    for ax in axes[:,i]: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=100); print("saved")
