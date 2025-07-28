import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import numpy as np

# input_file = "/Users/giuliamonopoli/Desktop/PhD/Data/MnM2/dataset_information.csv"

# df = pd.read_csv(input_file)
# df = df.dropna(subset=['SUBJECT_CODE'])

# df['SUBJECT_CODE'] = df['SUBJECT_CODE'].astype(int).astype(str).str.zfill(3)
# df_nor = df[df['DISEASE'] == 'NOR']
# patients = df_nor['SUBJECT_CODE'].tolist()


def is_valid_slice(slice_mask, max_components=1, solidity_threshold=0.9):
    """
    Check if a slice is valid based on the number of connected components and the solidity 
    (area / convex hull area) of the largest component.
    Args:
        slice_mask (np.ndarray): Binary mask of the slice.
        max_components (int): Maximum allowed connected components (excluding background).
        solidity_threshold (float): Minimum acceptable solidity.
    Returns:
        bool: True if the slice is valid; False otherwise.
    """
    # Connected components check
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_mask.astype(np.uint8))
    num_components = num_labels - 1  # Exclude background
    if num_components > max_components:
        return False

    contours, _ = cv2.findContours(slice_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    solidity = float(area) / hull_area
    return solidity >= solidity_threshold


def create_h5_file(patients: list, base_path: str, output_dir: str):
    for patient in patients:
        nii_file_path = os.path.join(base_path, patient, 'seg_stack.nii.gz') # Segmentation file path
        nii_mri = os.path.join(base_path, patient, 'img_stack.nii.gz') # MRI file path
        try:
            mri_image = nib.load(nii_mri)
            voxel_dimensions = mri_image.header['pixdim'][1:4]  # [x, y, z]
            slice_gap = voxel_dimensions[2] 
            resolution = voxel_dimensions[0:3]  # [x, y]
            
            nii_img = nib.load(nii_file_path)
            segmentation_data = nii_img.get_fdata()
            label_1_mask = np.where(segmentation_data == 2, 1, 0).astype(np.int16)
            label_1_mask = label_1_mask[:, :, np.any(label_1_mask, axis=(0, 1))]
            label_1_mask = np.transpose(label_1_mask, (2, 0, 1))
            
            rv_mask = np.where(segmentation_data == 3, 1, 0).astype(np.int16)
            rv_mask = rv_mask[:, :, np.any(rv_mask, axis=(0, 1))]
            rv_mask = np.transpose(rv_mask, (2, 0, 1))

            num_slices = label_1_mask.shape[0]  
            valid_slices = []

            for i in range(num_slices):
                slice_mask = label_1_mask[i, :, :]
                if is_valid_slice(slice_mask):
                    valid_slices.append(i)

            if len(valid_slices) <  num_slices-2:
                print(f"Warning: Too many slices have been deselected for patient {patient} ")
                print(f"Number of valid slices: {len(valid_slices)} out of {num_slices}")
                print(f"Valid slices: {valid_slices}")
            # output_dir = f'/Users/giuliamonopoli/Desktop/PhD/meshing/controls/ES_files/'
            mkdir = os.path.dirname(output_dir)
            if not os.path.exists(mkdir):
                os.makedirs(mkdir)
            output_path = os.path.join(output_dir, f"{patient}_original_segmentation.h5")
        
            with h5py.File(output_path, 'w') as h5_file:
    #         # Save LVmask
                h5_file.create_dataset('LVmask', data=label_1_mask[valid_slices, :, :])
            
                h5_file.create_dataset('resolution', data=resolution)
                h5_file.create_dataset('RV_mask', data=rv_mask)

            print("Data saved to data.h5")
        # exit()

        # Plot valid slices
    #     grid_size = int(np.ceil(np.sqrt(len(valid_slices))))  # Determine grid size
    #     fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
                                
    #     for idx, i in enumerate(valid_slices):
    #         row = idx // grid_size
    #         col = idx % grid_size
    #         label_1_slice = label_1_mask[:, :, i]
    #         axes[row, col].imshow(label_1_slice, cmap='gray')
    #         axes[row, col].set_title(f"Slice {i+1}")
    #         axes[row, col].axis('off')

    #     # Hide any unused subplots
    #     for j in range(idx + 1, grid_size * grid_size):
    #         fig.delaxes(axes.flatten()[j])

    #     plt.tight_layout()
    #     plt.show()
        except FileNotFoundError:
            print(f"File not found: {nii_file_path}")   
# import numpy as np
# import os
# from pathlib import Path
# import matplotlib.pyplot as plt
# import nibabel as nib
# import cv2
# import json
# import pandas as pd
# from pydicom import dcmread
# import h5py
# def get_value_for_case(case_name):
#         """
#         Reads the Excel file and returns the value for the given case_name.
#         Skips the first row which contains titles.
#         """
#         case_name = int(case_name)
#         input_file="/Users/giuliamonopoli/Desktop/PhD /shaping_mad/slice_gap.xlsx"
#         try:
#             df = pd.read_excel(input_file,names=["Case", "Value"])
#             row = df.loc[df['Case'] == case_name]
#             if not row.empty:
#                 value = row['Value'].values[0]  
#                 return int(value) if pd.notnull(value) else None
#             else:
#                 return None  
#         except Exception as e:
#             print(f"Error reading the file: {e}")
#             return None

# patients = os.listdir("/Users/giuliamonopoli/Desktop/PhD /Data/ES_files/")
# patients = [171]#[x for x in patients if x != '.DS_Store']
# for subject_id in patients:
#     print(f"Processing subject {subject_id}")

#     dcm_folder_sax = Path(f"/Users/giuliamonopoli/Desktop/PhD /Data/ES_files/{subject_id}/DICOM_files")
#     annotation_path =   "merged_annotations_no_bounding_box.json" 
#     segmentation_file = f"/Users/giuliamonopoli/Desktop/PhD /Data/ES_files/{subject_id}/NIfTI_files/{subject_id}_myo.nii"
#     output_folder = f"/Users/giuliamonopoli/Desktop/PhD /meshing/Segmentations_ES"
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     output_path = os.path.join(output_folder, f"{subject_id}_original_segmentation.h5")

#     sax_files = sorted([f for f in os.listdir(dcm_folder_sax) if f.endswith(".dcm") and f!=".dcm"], key=lambda x: int(x.split('sliceloc_')[1].split('.')[0]))
#     dcm_sax = [dcmread(dcm_folder_sax / f) for f in sax_files]
#     dcm_file = dcm_sax[3]
#     slice_gap = get_value_for_case(subject_id)

#     seg = nib.load(segmentation_file).get_fdata()

#     imres = float(dcm_file.PixelSpacing[0]), float(dcm_file.PixelSpacing[1]), slice_gap
#     print(f"Resolution: {imres}")
#     reordered_segmentation_data = seg[:, :, ::-1]
#     # reordered_segmentation_data = seg
# # remove empty slices
#     reordered_segmentation_data = reordered_segmentation_data[:, :, np.any(reordered_segmentation_data, axis=(0, 1))]
# # use z as first dimension
#     reordered_segmentation_data = np.transpose(reordered_segmentation_data, (2, 0, 1))
#     print(f"Segmentation shape: {reordered_segmentation_data.shape}")
#     with h5py.File(output_path, 'w') as h5_file:
#         # Save LVmask
#         h5_file.create_dataset('LVmask', data=reordered_segmentation_data)
    
#         h5_file.create_dataset('resolution', data=imres)

#     print("Data saved to data.h5")



# with h5py.File("/Users/giuliamonopoli/Desktop/PhD /meshing/Segmentations_ES/171_original_segmentation.h5", 'r') as h5_file:
#     segg = h5_file['LVmask'][:]
#     imres = h5_file['resolution'][:]
# num_slices = segg.shape[0]  # Assuming slices are along the third dimension
# grid_size = int(np.ceil(np.sqrt(num_slices))) # Determine grid size

# fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

# for i in range(num_slices):
#     row = i // grid_size
#     col = i % grid_size
#     label_1_slice = segg[i, :, :]
#     axes[row, col].imshow(label_1_slice, cmap='gray')
#     axes[row, col].set_title(f"Slice {i+1}")
#     axes[row, col].axis('off')
# plt.show()