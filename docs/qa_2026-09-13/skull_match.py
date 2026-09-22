"""Skull-only rigid registration: pose-invariant, individually shaped. Same group as before."""
import numpy as np, SimpleITK as sitk, time
from scipy.optimize import linear_sum_assignment
R="/data1/Processed_NIfTI_Test/mice"; mice=["NaF_WT_03","NaF_WT_04","NaF_WT_05","NaF_WT_06"]

def skull(m, wk, mm=0.3, slab_mm=28):
    img=sitk.ReadImage(f"{R}/{m}/week_{wk}/ct_hi.nii.gz")
    f=[max(1,int(round(mm/s))) for s in img.GetSpacing()]; img=sitk.Shrink(img,f)
    a=sitk.GetArrayFromImage(img)  # z,y,x
    bone=a>300
    prof=bone.sum(axis=(1,2)).astype(float)               # bone area per z-slice
    n=len(prof); k=int(slab_mm/img.GetSpacing()[2])
    # skull = the end whose first k slices (inside the body's bone extent) hold the most bone
    zs=np.where(prof>0)[0]; lo,hi=zs[0],zs[-1]
    head_lo=prof[lo:lo+k].sum(); head_hi=prof[hi-k+1:hi+1].sum()
    z0,z1=(lo,lo+k) if head_lo>head_hi else (hi-k+1,hi+1)
    sub=np.zeros_like(bone); sub[z0:z1]=bone[z0:z1]
    m_=sitk.GetImageFromArray(sub.astype(np.uint8)); m_.CopyInformation(img)
    cc=sitk.RelabelComponent(sitk.ConnectedComponent(m_),sortByObjectSize=True)
    return sitk.Cast(cc==1,sitk.sitkFloat32), ("low-z" if head_lo>head_hi else "high-z"), int(sub.sum())

def dice_rigid(fixed,moving):
    tx=sitk.CenteredTransformInitializer(fixed,moving,sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.MOMENTS)
    reg=sitk.ImageRegistrationMethod(); reg.SetMetricAsMeanSquares(); reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=0.5,minStep=1e-4,numberOfIterations=300)
    reg.SetOptimizerScalesFromPhysicalShift(); reg.SetInitialTransform(tx,inPlace=False)
    reg.SetShrinkFactorsPerLevel([4,2,1]); reg.SetSmoothingSigmasPerLevel([1.5,0.8,0])
    T=reg.Execute(fixed,moving); w=sitk.Resample(moving,fixed,T,sitk.sitkNearestNeighbor,0.0)
    a=sitk.GetArrayFromImage(fixed)>0.5; b=sitk.GetArrayFromImage(w)>0.5
    return 2*(a&b).sum()/max(a.sum()+b.sum(),1)

t0=time.time(); S12={}; S15={}
for m in mice:
    S12[m]=skull(m,12); S15[m]=skull(m,15)
    print(f"  {m}: wk12 skull at {S12[m][1]} ({S12[m][2]} vox), wk15 skull at {S15[m][1]} ({S15[m][2]} vox)")
print(f"skulls extracted in {time.time()-t0:.0f}s")
D=np.zeros((4,4))
for i,a in enumerate(mice):
    for j,b in enumerate(mice):
        D[i,j]=dice_rigid(S15[b][0],S12[a][0]); print(f"  {a}->{b}: {D[i,j]:.3f}{'  <-- same' if i==j else ''}",flush=True)
print("\nDice matrix (rows wk12, cols wk15):\n",np.round(D,3))
same=np.diag(D); diff=D[~np.eye(4,dtype=bool)]
print(f"same-mouse mean {same.mean():.3f} (min {same.min():.3f}) | different-mouse mean {diff.mean():.3f} (max {diff.max():.3f})")
r,c=linear_sum_assignment(-D); print("Hungarian:",[(mice[i][-2:],mice[j][-2:]) for i,j in zip(r,c)],"| matches manifest:",all(i==j for i,j in zip(r,c)))
