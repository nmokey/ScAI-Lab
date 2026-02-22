import os
import glob
import csv
import yaml
import SimpleITK as sitk
import pydicom
import numpy as np


# --- Load Config ---

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. "
            "Copy config.yaml.example to config.yaml and fill in your paths."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)

cfg = load_config()

OUTPUT_DIR     = cfg['paths']['output_dir']
MANIFEST_PATH  = os.path.join(OUTPUT_DIR, "manifest.csv")
IGNORE_EXTENSIONS = {'.im3', '.vol', '.raw'}

_heuristics = cfg['file_size_heuristics']
_tol = _heuristics['tolerance']

PET_SIZE_TARGET    = _heuristics['pet_kb'] * 1024
PET_TOLERANCE      = _tol * PET_SIZE_TARGET

CT_HI_RES_TARGET   = _heuristics['ct_hi_res_kb'] * 1024
CT_HI_TOLERANCE    = _tol * CT_HI_RES_TARGET

CT_LO_RES_TARGET   = _heuristics['ct_lo_res_kb'] * 1024
CT_LO_TOLERANCE    = _tol * CT_LO_RES_TARGET

# --- Test Subject (single-subject run only) ---
_test = cfg['test_subject']
TEST_WEEK    = _test['week']
TEST_TRACER  = _test['tracer']
TEST_SUBJECT = _test['subject_id']
TEST_DICOM_FOLDER = os.path.join(
    cfg['paths']['data_root'],
    TEST_WEEK,
    TEST_TRACER,
    TEST_SUBJECT,
    f"dicom_{TEST_SUBJECT}",
)


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


def _append_manifest(subject, week, modality, nifti_path):
    write_header = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['SubjectID', 'Week', 'Modality', 'Path'])
        if write_header:
            writer.writeheader()
        writer.writerow({'SubjectID': subject, 'Week': week, 'Modality': modality, 'Path': nifti_path})


def save_sitk_image(image, subject, week, tracer, modality):
    week_safe   = week.replace(" ", "_")
    tracer_safe = tracer.replace("-", "_")
    filename    = f"{subject}_{week_safe}_{tracer_safe}_{modality}.nii.gz"
    output_path = os.path.join(OUTPUT_DIR, filename)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        sitk.WriteImage(image, output_path)
        print(f"    [+] Saved {filename} to {OUTPUT_DIR}")
        _append_manifest(subject, week, modality, output_path)
    except Exception as e:
        print(f"    [!] Error saving {filename}: {e}")


def sort_slices_by_z(file_list):
    """
    Sorts a list of DICOM files by their physical Z-axis position.
    stop_before_pixels=True ensures this runs instantly without loading image data.
    Files with unreadable headers are excluded with a warning.
    """
    def get_z_coord(filepath):
        try:
            dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
            return float(dcm.ImagePositionPatient[2])
        except Exception:
            return None

    tagged = [(get_z_coord(f), f) for f in file_list]
    valid = [(z, f) for z, f in tagged if z is not None]
    excluded_count = len(tagged) - len(valid)
    if excluded_count:
        print(f"    [!] Warning: Excluded {excluded_count} slice(s) with unreadable Z-position.")
    return [f for _, f in sorted(valid, key=lambda x: x[0])]


def process_subject(week, tracer, subject, dicom_folder_path):
    print(f"--> Processing: {week} | {tracer} | {subject}")

    all_files = glob.glob(os.path.join(dicom_folder_path, "**", "*"), recursive=True)
    all_files = [
        f for f in all_files
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() not in IGNORE_EXTENSIONS
    ]

    pet_files    = []
    ct_hi_files  = []
    ct_lo_files  = []

    for f in all_files:
        if is_size_match(f, PET_SIZE_TARGET, PET_TOLERANCE):
            pet_files.append(f)
        elif is_size_match(f, CT_HI_RES_TARGET, CT_HI_TOLERANCE):
            ct_hi_files.append(f)
        elif is_size_match(f, CT_LO_RES_TARGET, CT_LO_TOLERANCE):
            ct_lo_files.append(f)

    print(f"    [i] Found {len(pet_files)} PET slices, {len(ct_hi_files)} Hi-Res CT slices, {len(ct_lo_files)} Low-Res CT slices.")

    pet_files   = sort_slices_by_z(pet_files)
    ct_hi_files = sort_slices_by_z(ct_hi_files)
    ct_lo_files = sort_slices_by_z(ct_lo_files)

    # Process PET
    if pet_files:
        pet_img = convert_dicom_to_nifti_sitk(pet_files, subject, "PET")
        if pet_img:
            save_sitk_image(pet_img, subject, week, tracer, "PET")
    else:
        print(f"    [!] No PET data found.")

    # Process CT (Hi-Res preferred; fall back to Low-Res)
    ct_img = None
    ct_modality_label = "CT_HiRes"

    if ct_hi_files:
        ct_img = convert_dicom_to_nifti_sitk(ct_hi_files, subject, "CT_HiRes")

    if ct_img is None and ct_lo_files:
        print(f"    [i] No Hi-Res CT found — using Low-Res CT fallback.")
        ct_img = convert_dicom_to_nifti_sitk(ct_lo_files, subject, "CT_LowRes")
        ct_modality_label = "CT_LowRes"

    if ct_img:
        save_sitk_image(ct_img, subject, week, tracer, ct_modality_label)
    else:
        print(f"    [!] No valid CT volume could be built.")


def main():
    print(f"--- Starting Single Subject Test ---")
    print(f"Target: {TEST_SUBJECT} ({TEST_WEEK}, {TEST_TRACER})")
    print(f"DICOM folder: {TEST_DICOM_FOLDER}")

    if not os.path.exists(TEST_DICOM_FOLDER):
        print(f"[!] Error: The path '{TEST_DICOM_FOLDER}' does not exist.")
        print("    Check data_root, week, tracer, and subject_id in config.yaml.")
        return

    process_subject(TEST_WEEK, TEST_TRACER, TEST_SUBJECT, TEST_DICOM_FOLDER)
    print(f"--- Test Complete ---")


if __name__ == "__main__":
    main()
