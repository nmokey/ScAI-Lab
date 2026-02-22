import SimpleITK as sitk
import matplotlib.pyplot as plt
import sys

def show_ortho_slices(nifti_path):
    # Load the generated NIfTI volume
    print(f"Loading {nifti_path}...")
    img = sitk.ReadImage(nifti_path)
    img_array = sitk.GetArrayFromImage(img)
    
    # Get the middle index for each dimension (Z, Y, X in numpy)
    z_mid = img_array.shape[0] // 2
    y_mid = img_array.shape[1] // 2
    x_mid = img_array.shape[2] // 2

    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (XY plane)
    axes[0].imshow(img_array[z_mid, :, :], cmap='gray')
    axes[0].set_title(f'Axial (Z={z_mid})')
    axes[0].axis('off')

    # Coronal (XZ plane)
    axes[1].imshow(img_array[:, y_mid, :], cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (Y={y_mid})')
    axes[1].axis('off')

    # Sagittal (YZ plane)
    axes[2].imshow(img_array[:, :, x_mid], cmap='gray', origin='lower')
    axes[2].set_title(f'Sagittal (X={x_mid})')
    axes[2].axis('off')

    plt.tight_layout()
    # Save the plot as an image so you can easily view it (useful for headless servers)
    output_img = nifti_path.replace('.nii.gz', '_qa.png')
    plt.savefig(output_img)
    print(f"Saved QA image to {output_img}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_nifti.py <path_to_nifti_file>")
    else:
        show_ortho_slices(sys.argv[1])