import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np

# --- Configuration & Heuristics ---

# Safe output directory specifically for testing
OUTPUT_DIR = "/data1/Processed_NIfTI_Test/"

# --- Single Subject Target ---
# Update these specific strings to point to ONE subject on your server
TEST_WEEK = "Week 12"
TEST_TRACER = "18F-FDG"
TEST_SUBJECT = "m54253"
# Point this directly to the lowest-level folder containing the actual slices
TEST_DICOM_FOLDER = "/data1/Dicom Data/Week 12/18F-FDG/m54253/dicom_m54253"

# Target File Sizes (Bytes) with ~2% tolerance
PET_SIZE_TARGET = 115 * 1024
PET_TOLERANCE = 0.05 * PET_SIZE_TARGET 

CT_HI_RES_TARGET = 2814 * 1024
CT_HI_TOLERANCE = 0.05 * CT_HI_RES_TARGET

CT_LO_RES_TARGET = 705 * 1024
CT_LO_TOLERANCE = 0.05 * CT_LO_RES_TARGET

def is_size_match(filepath, target, tolerance):
    try:
        size = os.path.getsize(filepath)
        return abs(size - target) <= tolerance
    except OSError:
        return False

def convert_dicom_to_nifti_sitk(file_list, subject_id, modality):
    if not file_list:
        return None

    print(f"    ... Reading {len(file_list)} files for {modality} via SimpleITK ...")
    
    reader = sitk.ImageSeriesReader()
    # Explicitly pass our size-filtered list to SimpleITK
    reader.SetFileNames(file_list)

    try:
        image = reader.Execute()
        
        spacing = image.GetSpacing()
        print(f"    [i] {modality} Spacing detected: {spacing}")
        
        if not (np.isclose(spacing[0], spacing[1]) and np.isclose(spacing[1], spacing[2])):
             print(f"        -> Note: Anisotropic voxels detected.")

        return image

    except Exception as e:
        print(f"    [!] SITK Error reading {modality}: {e}")
        return None

def save_sitk_image(image, subject, week, tracer, modality):
    filename = f"{subject}_{week}_{tracer}_{modality}.nii.gz"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    try:
        sitk.WriteImage(image, output_path)
        print(f"    [+] Saved {filename} to {OUTPUT_DIR}")
    except Exception as e:
        print(f"    [!] Error saving {filename}: {e}")

def sort_slices_by_z(file_list):
    """
    Sorts a list of DICOM files by their physical Z-axis position.
    stop_before_pixels=True ensures this runs instantly without loading image data.
    """
    def get_z_coord(filepath):
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            # ImagePositionPatient is [X, Y, Z]. We want Z (index 2).
            return float(dcm.ImagePositionPatient[2])
        except Exception:
            return 0.0 # Fallback if header is completely corrupt
            
    return sorted(file_list, key=get_z_coord)

def process_subject(week, tracer, subject, dicom_folder_path):
    print(f"--> Processing: {week} | {tracer} | {subject}")
    
    all_files = glob.glob(os.path.join(dicom_folder_path, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]

    pet_files = []
    ct_hi_files = []
    ct_lo_files = []

    for f in all_files:
        if is_size_match(f, PET_SIZE_TARGET, PET_TOLERANCE):
            pet_files.append(f)
        elif is_size_match(f, CT_HI_RES_TARGET, CT_HI_TOLERANCE):
            ct_hi_files.append(f)
        elif is_size_match(f, CT_LO_RES_TARGET, CT_LO_TOLERANCE):
            ct_lo_files.append(f)

    print(f"    [i] Found {len(pet_files)} PET slices, {len(ct_hi_files)} Hi-Res CT slices, {len(ct_lo_files)} Low-Res CT slices.")
    
    # Sort each modality's files by their Z-axis position to ensure correct volume assembly
    pet_files = sort_slices_by_z(pet_files)
    ct_hi_files = sort_slices_by_z(ct_hi_files)
    ct_lo_files = sort_slices_by_z(ct_lo_files)

    # Process PET
    if pet_files:
        pet_img = convert_dicom_to_nifti_sitk(pet_files, subject, "PET")
        if pet_img:
            save_sitk_image(pet_img, subject, week, tracer, "PET")
    else:
        print(f"    [!] No PET data found.")

    # Process CT
    ct_img = None
    ct_modality_label = "CT_HiRes"

    if ct_hi_files:
        ct_img = convert_dicom_to_nifti_sitk(ct_hi_files, subject, "CT_HiRes")
    
    # Fallback to Low-Res
    if ct_img is None and ct_lo_files:
        print(f"    [i] Switching to Low-Res CT fallback...")
        ct_img = convert_dicom_to_nifti_sitk(ct_lo_files, subject, "CT_LowRes")
        ct_modality_label = "CT_LowRes"

    if ct_img:
        save_sitk_image(ct_img, subject, week, tracer, ct_modality_label)
    else:
        print(f"    [!] No valid CT volume could be built.")

def main():
    print(f"--- Starting Single Subject Test ---")
    print(f"Target: {TEST_SUBJECT} ({TEST_WEEK}, {TEST_TRACER})")
    
    if not os.path.exists(TEST_DICOM_FOLDER):
        print(f"[!] Error: The path '{TEST_DICOM_FOLDER}' does not exist.")
        print("    Please double-check the TEST_DICOM_FOLDER variable.")
        return

    process_subject(TEST_WEEK, TEST_TRACER, TEST_SUBJECT, TEST_DICOM_FOLDER)
    print(f"--- Test Complete ---")

if __name__ == "__main__":
    main()