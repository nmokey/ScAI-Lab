"""Compare geometry of every session NIfTI, and raw DICOM orientation tags for the WT_03-06 group."""
import glob, os, collections, numpy as np, SimpleITK as sitk, csv
R="/data1/Processed_NIfTI_Test/sessions"
print("=== session NIfTI geometry (direction cosines / origin sign / spacing), grouped ===")
groups=collections.defaultdict(list)
for p in sorted(glob.glob(f"{R}/week_*/*/*/ct_hi.nii.gz")):
    r=sitk.ImageFileReader(); r.SetFileName(p); r.ReadImageInformation()
    d=tuple(round(x,3) for x in r.GetDirection()); o=tuple(np.sign(np.round(r.GetOrigin(),1))); sp=tuple(round(x,3) for x in r.GetSpacing()); sz=r.GetSize()
    groups[(d,o,sp,sz)].append(p.split("/")[-2]+"@"+p.split("/")[-4])
for k,v in groups.items():
    print(f"  n={len(v):2d}  dir={k[0]}  origin_sign={k[1]}  spacing={k[2]}  size={k[3]}")
    print(f"        e.g. {v[:6]}")
