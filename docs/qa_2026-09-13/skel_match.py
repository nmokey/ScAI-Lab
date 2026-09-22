"""
Can skeletal shape identify an individual mouse across weeks?

For one stable group (WT_03-06, m54226 wk12 -> m54400 wk15), rigid-register every
wk12 bone mask to every wk15 bone mask and score Dice. If the same-mouse pairs (per
the manifest) score clearly above the different-mouse pairs, anatomy carries identity
and the labeling can be tested on every stable group.
"""
import numpy as np, SimpleITK as sitk, sys, time, itertools
R="/data1/Processed_NIfTI_Test/mice"
mice=["NaF_WT_03","NaF_WT_04","NaF_WT_05","NaF_WT_06"]

def bone_mask(m, wk, mm=0.5):
    img=sitk.ReadImage(f"{R}/{m}/week_{wk}/ct_hi.nii.gz")
    f=[int(round(mm/s)) for s in img.GetSpacing()]
    img=sitk.Shrink(img,f)
    b=sitk.BinaryThreshold(img,300,10000,1,0)
    # keep the largest connected component (the skeleton), drop tube fragments
    cc=sitk.ConnectedComponent(b); cc=sitk.RelabelComponent(cc,sortByObjectSize=True)
    return sitk.Cast(cc==1,sitk.sitkFloat32)

def dice_after_rigid(fixed, moving):
    tx=sitk.CenteredTransformInitializer(fixed,moving,sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg=sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares(); reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,minStep=1e-3,numberOfIterations=150)
    reg.SetOptimizerScalesFromPhysicalShift(); reg.SetInitialTransform(tx,inPlace=False)
    reg.SetShrinkFactorsPerLevel([4,2,1]); reg.SetSmoothingSigmasPerLevel([2,1,0])
    T=reg.Execute(fixed,moving)
    warped=sitk.Resample(moving,fixed,T,sitk.sitkNearestNeighbor,0.0)
    a=sitk.GetArrayFromImage(fixed)>0.5; b=sitk.GetArrayFromImage(warped)>0.5
    return 2*(a&b).sum()/max(a.sum()+b.sum(),1)

t0=time.time()
w12={m:bone_mask(m,12) for m in mice}; w15={m:bone_mask(m,15) for m in mice}
print(f"masks built in {time.time()-t0:.0f}s; bone voxels @0.5mm:",{m:int(sitk.GetArrayFromImage(w12[m]).sum()) for m in mice})
D=np.zeros((4,4))
for i,a in enumerate(mice):
    for j,b in enumerate(mice):
        D[i,j]=dice_after_rigid(w15[b],w12[a])
        print(f"  {a} wk12 -> {b} wk15 : Dice={D[i,j]:.3f}{'  <-- same mouse per manifest' if i==j else ''}",flush=True)
print("\nDice matrix (rows=wk12 mouse, cols=wk15 mouse):"); print(np.round(D,3))
same=np.diag(D); diff=D[~np.eye(4,dtype=bool)]
print(f"\nsame-mouse Dice: mean {same.mean():.3f} (min {same.min():.3f})   different-mouse: mean {diff.mean():.3f} (max {diff.max():.3f})")
from scipy.optimize import linear_sum_assignment
r,c=linear_sum_assignment(-D); print("best assignment (Hungarian):",[(mice[i],mice[j]) for i,j in zip(r,c)], "| matches manifest:", all(i==j for i,j in zip(r,c)))
