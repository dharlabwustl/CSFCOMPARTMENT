import nibabel as nib
import os,subprocess,sys,glob
import numpy as np
import argparse
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
def save_nifti_without_affine(matrix, filename):
    """
    Save a 3D matrix as a NIfTI file without explicitly providing an affine matrix.
    An identity affine matrix will be used by default.

    Parameters:
    - matrix: 3D numpy array representing the data (e.g., binary mask).
    - filename: The file path to save the NIfTI file.
    """
    # Create a NIfTI image using the matrix, without an explicit affine matrix
    # Nibabel will automatically use an identity affine matrix if none is provided
    nifti_img = nib.Nifti1Image(matrix, affine=np.eye(4))

    # Save the NIfTI image to the specified filename
    nib.save(nifti_img, filename)

def create_obb_mask_from_image_mask(binary_mask):
    # Step 1: Find the non-zero points (coordinates where the mask is non-zero)
    points = np.column_stack(np.nonzero(binary_mask))

    # Step 2: Apply PCA to find the principal components (directions of maximum variance)
    pca = PCA(n_components=3)
    pca.fit(points)

    # Step 3: Rotate the points to align them with the principal axes
    points_rotated = pca.transform(points)

    # Step 4: Find the minimum and maximum points along each principal axis
    min_bounds = np.min(points_rotated, axis=0)
    max_bounds = np.max(points_rotated, axis=0)

    # Compute the dimensions of the oriented bounding box in rotated space
    obb_dimensions = max_bounds - min_bounds

    # Step 5: Compute the 8 corner points of the oriented bounding box in the rotated space
    corner_points_rotated = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
    ])

    # Step 6: Rotate the corner points back to the original space
    corner_points_original = pca.inverse_transform(corner_points_rotated)

    # Function to create a mask from the OBB
    def create_obb_mask(shape, corner_points):
        # Create an empty 3D mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Generate a convex hull from the corner points
        hull = ConvexHull(corner_points)

        # Get the voxel grid of the mask
        x, y, z = np.indices(shape)

        # Stack the voxel grid into an (N, 3) shape
        grid_points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)

        # For each point, check if it's inside the convex hull
        inside_hull = Delaunay(corner_points).find_simplex(grid_points) >= 0

        # Fill the mask with the points inside the OBB
        mask.ravel()[inside_hull] = 1

        return mask

    # Step 7: Create the 3D mask of the oriented bounding box
    obb_mask = create_obb_mask(binary_mask.shape, corner_points_original)

    return obb_mask

# # Example usage:
# # Define a sample 3D binary mask (replace this with your actual 3D mask)
# binary_mask = np.zeros((10, 10, 10), dtype=np.uint8)
# binary_mask[2:8, 2:8, 2:8] = 1  # Example filled cube

# Call the function to get the oriented bounding box mask
# obb_mask = create_obb_mask_from_image_mask(binary_mask)

# # Output the result
# print("3D Oriented Bounding Box Mask:")
# print(obb_mask)

# Optionally, visualize using a 3D visualization tool like matplotlib or mayavi

ventricle_mask=nib.load(sys.argv[1]).get_fdata()
ventricle_obb_mask = create_obb_mask_from_image_mask(ventricle_mask)
save_nifti_without_affine(obb_mask, os.path.join(sys.argv[2],'ventricle_obb_mask.nii'))

# non_zero_slice_num=[]
# # print(ventricle_mask.get_fdata().shape[2])
# for slice_num in range(ventricle_mask.shape[2]):
#     this_slice_sum=np.sum(ventricle_mask[:,:,slice_num])
#     if this_slice_sum >0 :
#         # print(this_slice_sum)
#         non_zero_slice_num.append(slice_num)
# if len(non_zero_slice_num)>0:
#     upper_lower_limit_vent=[sessionID,scanID,original_nifti_filename,min(non_zero_slice_num),max(non_zero_slice_num)]
# print(upper_lower_limit_vent)
# upper_lower_limit_vent_df=pd.DataFrame(upper_lower_limit_vent).T
# upper_lower_limit_vent_df.columns=['SESSION_ID','SCAN_ID','NIFTI_FILENAME','LOWER_SLICE_NUM','UPPER_SCLICE_NUM']
# print(upper_lower_limit_vent_df)
# upper_lower_limit_vent_df.to_csv(os.path.join(SAVE_PATH,original_nifti_filename.split('.nii')[0]+'_ventricle_bounds.csv'),index=False)


