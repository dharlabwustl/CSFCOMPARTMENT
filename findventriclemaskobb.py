import nibabel as nib
from utilities_simple import *
import os,subprocess,sys,glob
import numpy as np
import argparse
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay

import numpy as np
import scipy.ndimage as ndi
import numpy as np
import scipy.ndimage as ndi
from skimage import filters

import numpy as np
import scipy.ndimage as ndi
##############################
import numpy as np
import cv2
import numpy as np
from sklearn.decomposition import PCA
#########################################################
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance

def get_obb_mask(binary_mask):
    """
    Fit the Oriented Bounding Box (OBB) to the binary mask and return the mask of the OBB.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)

    Returns:
    - obb_mask: 3D numpy array (mask of the OBB)
    - obb_corners: 8 corner points of the OBB in the original space
    """
    # Find non-zero points (the object)
    points = np.column_stack(np.nonzero(binary_mask))

    if points.size == 0:
        # If no object, return an empty mask
        return np.zeros_like(binary_mask), []

    # Apply PCA to get the principal axes of the object
    pca = PCA(n_components=3)
    pca.fit(points)

    # Rotate the points to align with principal axes
    points_rotated = pca.transform(points)

    # Get the bounding box in the rotated space
    min_bounds = np.min(points_rotated, axis=0)
    max_bounds = np.max(points_rotated, axis=0)

    # Get the 8 corners of the bounding box in rotated space
    corners_rotated = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
    ])

    # Rotate the corners back to the original space
    obb_corners = pca.inverse_transform(corners_rotated)

    # Create an empty mask and fill it with the OBB
    obb_mask = np.zeros_like(binary_mask)

    # Find indices for the bounding box
    x_min, y_min, z_min = np.floor(np.min(obb_corners, axis=0)).astype(int)
    x_max, y_max, z_max = np.ceil(np.max(obb_corners, axis=0)).astype(int)

    # Fill the OBB mask
    obb_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1

    return obb_mask, obb_corners

def subdivide_obb(obb_corners):
    """
    Subdivide the OBB into 4 smaller boxes and find their centroids.

    Parameters:
    - obb_corners: 8 corner points of the OBB

    Returns:
    - centroids: List of 4 centroids of the subdivided boxes
    """
    # Calculate the centroid of the OBB
    obb_center = np.mean(obb_corners, axis=0)

    # Subdivide the bounding box by dividing it into four smaller boxes
    min_corner = np.min(obb_corners, axis=0)
    max_corner = np.max(obb_corners, axis=0)
    mid_corner = (min_corner + max_corner) / 2

    # Create 4 smaller boxes (subdividing along all 3 axes)
    centroids = [
        (min_corner + mid_corner) / 2,  # Lower box
        [mid_corner[0], (min_corner[1] + mid_corner[1]) / 2, (min_corner[2] + mid_corner[2]) / 2],  # Mid left
        [mid_corner[0], (max_corner[1] + mid_corner[1]) / 2, (max_corner[2] + mid_corner[2]) / 2],  # Mid right
        (max_corner + mid_corner) / 2,  # Upper box
    ]
    return centroids

def find_closest_non_zero_voxel(binary_mask, centroids):
    """
    For each centroid, find the closest non-zero voxel in the binary mask.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - centroids: List of centroids from the subdivided boxes

    Returns:
    - closest_voxels: List of coordinates of the closest non-zero voxels to the centroids
    """
    non_zero_voxels = np.column_stack(np.nonzero(binary_mask))

    if non_zero_voxels.size == 0:
        return []

    closest_voxels = []

    for centroid in centroids:
        distances = distance.cdist([centroid], non_zero_voxels)
        closest_idx = np.argmin(distances)
        closest_voxels.append(non_zero_voxels[closest_idx])

    return closest_voxels

def process_3d_mask(binary_mask):
    """
    Main function to get the OBB mask, subdivide it, find centroids, and return the closest
    non-zero voxels to the centroids.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)

    Returns:
    - closest_voxels: Coordinates of the closest non-zero voxels to the centroids
    - obb_mask: 3D mask of the OBB
    """
    # Step 1: Get the OBB mask
    obb_mask, obb_corners = get_obb_mask(binary_mask)

    # Step 2: Subdivide the OBB and find the centroids of the subdivided boxes
    centroids = subdivide_obb(obb_corners)

    # Step 3: Find the closest non-zero voxels to the centroids in the original binary mask
    closest_voxels = find_closest_non_zero_voxel(binary_mask, centroids)

    return closest_voxels, obb_mask

# # Example usage:
# binary_mask = np.zeros((100, 100, 100), dtype=np.uint8)
# binary_mask[30:70, 30:70, 30:70] = 1  # Example filled 3D block
#
# # Process the 3D mask to get the closest non-zero voxels and OBB mask
# closest_voxels, obb_mask = process_3d_mask(binary_mask)
#
# print("Closest non-zero voxel coordinates to the centroids:")
# print(closest_voxels)
#
# print("\nOBB Mask:")
# print(obb_mask)



#########################################################

def smooth_3d_mask(binary_mask, sigma=1):
    """
    Smooth the boundary of a 3D binary mask using a Gaussian filter.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - sigma: Standard deviation for Gaussian kernel (default is 1)

    Returns:
    - smoothed_mask: 3D numpy array (smoothed binary mask).
    """
    # Step 1: Apply a Gaussian filter to smooth the edges of the mask
    smoothed = filters.gaussian(binary_mask.astype(float), sigma=sigma)

    # Step 2: Threshold the smoothed mask back to binary
    smoothed_mask = smoothed > 0.5

    return smoothed_mask.astype(np.uint8)
def fit_ellipsoid_to_3d_mask(binary_mask):
    """
    Fit a best-fit ellipsoid to a 3D binary mask and return a binary mask representing the region
    inside the ellipsoid.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)

    Returns:
    - ellipsoid_mask: 3D numpy array (binary mask of the ellipsoid).
    """
    # Step 1: Find the non-zero points (coordinates where the mask is non-zero)
    points = np.column_stack(np.nonzero(binary_mask))

    if points.size == 0:
        # If there are no points in the mask, return an empty mask
        return np.zeros_like(binary_mask)

    # Step 2: Apply PCA to find the principal components (directions of maximum variance)
    pca = PCA(n_components=3)
    pca.fit(points)

    # Get the center of the ellipsoid (mean of the points)
    center = np.mean(points, axis=0)

    # Step 3: Get the axes of the ellipsoid (radii) from the PCA components
    radii = np.sqrt(np.var(points @ pca.components_.T, axis=0))  # Standard deviations along PCA axes

    # Step 4: Generate a grid of points for the mask
    grid_x, grid_y, grid_z = np.indices(binary_mask.shape)

    # Step 5: Translate grid points relative to the ellipsoid center
    grid_points = np.stack([grid_x - center[0], grid_y - center[1], grid_z - center[2]], axis=-1)
    grid_points = grid_points.reshape(-1, 3)

    # Step 6: Rotate the grid points to align with the principal axes
    rotated_grid_points = grid_points @ pca.components_

    # Step 7: Normalize the rotated grid points by the radii of the ellipsoid
    normalized_points = rotated_grid_points / radii

    # Step 8: Compute the distance from the center in the normalized space
    distance_from_center = np.sum(normalized_points**2, axis=1)

    # Step 9: Points inside the ellipsoid will have a distance <= 1
    ellipsoid_mask = (distance_from_center <= 1).reshape(binary_mask.shape)

    return ellipsoid_mask.astype(np.uint8)




def fit_and_fill_ellipse_2d(slice_2d):
    """
    Fit an ellipse to a 2D binary mask slice, fill the ellipse, and return the filled slice.

    Parameters:
    - slice_2d: 2D numpy array (binary mask of a single slice)

    Returns:
    - filled_slice: 2D binary mask with the filled ellipse.
    """
    # Find the non-zero points (coordinates where the mask is non-zero)
    points = cv2.findNonZero(slice_2d.astype(np.uint8))

    # If there are no points in the slice, return an empty slice
    if points is None:
        return np.zeros_like(slice_2d)

    # Fit an ellipse to the points
    ellipse = cv2.fitEllipse(points)

    # Create an empty mask for the slice
    filled_slice = np.zeros_like(slice_2d)

    # Draw and fill the ellipse on the mask
    cv2.ellipse(filled_slice, ellipse, color=1, thickness=-1)  # Thickness -1 fills the ellipse

    return filled_slice

def fit_ellipse_to_3d_mask(binary_mask):
    """
    For each slice in the 3D binary mask, fit and fill an ellipse that encompasses all the pixels
    of that slice, and return a 3D mask of the stacked ellipses.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)

    Returns:
    - ellipse_mask: 3D numpy array with ellipses fitted and filled for each slice.
    """
    # Get the shape of the 3D mask
    z_slices, height, width = binary_mask.shape

    # Initialize an empty 3D mask to store the ellipses
    ellipse_mask = np.zeros((z_slices, height, width), dtype=np.uint8)

    # Iterate through each slice and fit an ellipse
    for z in range(z_slices):
        # Fit and fill the ellipse for the current slice
        # if np.sum(binary_mask[z])>10:
        try:
            ellipse_mask[z] = fit_and_fill_ellipse_2d(binary_mask[z])
        except:
            ellipse_mask[z] =binary_mask[z]
            pass


    return ellipse_mask

# # Example usage:
# # Replace this with your actual 3D binary mask
# binary_mask = np.zeros((10, 100, 100), dtype=np.uint8)
# cv2.circle(binary_mask[5], (50, 50), 30, 1, -1)  # Example filled 2D circle in slice 5
# cv2.rectangle(binary_mask[6], (30, 30), (70, 70), 1, -1)  # Example filled rectangle in slice 6
#
# # Fit ellipses for each 2D slice and create the 3D mask of stacked ellipses
# ellipse_3d_mask = fit_ellipse_to_3d_mask(binary_mask)
#
# # Print or visualize the result
# print("3D Mask with Fitted and Filled Ellipses for Each Slice:")
# print(ellipse_3d_mask)


##################################

def erode_3d_mask(binary_mask, iterations=1):
    """
    Erode a 3D binary mask by removing layers from the object.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - iterations: Number of erosion iterations to apply (default is 1)

    Returns:
    - eroded_mask: 3D binary mask that has been eroded.
    """
    # Perform binary erosion to shrink the object
    eroded_mask = ndi.binary_erosion(binary_mask, iterations=iterations).astype(np.uint8)

    return eroded_mask

def fill_holes_in_3d_mask(binary_mask):
    """
    Given a 3D binary mask, fill the holes inside the mask and return the updated mask.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)

    Returns:
    - filled_mask: 3D binary mask with the holes filled.
    """
    # Use binary_fill_holes to fill holes inside the mask

    filled_mask = ndi.binary_fill_holes(binary_mask).astype(np.uint8)
    filled_mask = ndi.binary_fill_holes(filled_mask).astype(np.uint8)
    filled_mask = ndi.binary_fill_holes(filled_mask).astype(np.uint8)
    return filled_mask

def dilate_3d_mask(binary_mask, iterations=1):
    """
    Dilate a 3D binary mask to make it bigger by expanding the object.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - iterations: Number of dilation iterations to apply (default is 1)

    Returns:
    - dilated_mask: 3D binary mask that has been dilated.
    """
    # Perform binary dilation to make the object larger
    # dilated_mask=erode_3d_mask(binary_mask, iterations=1)
    dilated_mask = ndi.binary_dilation(binary_mask, iterations=iterations).astype(np.uint8)

    return dilated_mask

def fill_dilate_and_fill_3d_mask(binary_mask, dilation_iterations=1):
    """
    Fill the holes in a 3D binary mask, dilate the mask to make it bigger, and then fill the
    holes again.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - dilation_iterations: Number of dilation iterations to apply (default is 1)

    Returns:
    - result_mask: 3D binary mask that has been filled, dilated, and filled again.
    """
    # Step 1: Fill holes in the mask
    filled_mask_1 = fill_holes_in_3d_mask(binary_mask)

    # Step 2: Dilate the filled mask to make it bigger
    dilated_mask = dilate_3d_mask(filled_mask_1, iterations=dilation_iterations)

    # Step 3: Fill holes again after dilation
    filled_mask_2 = fill_holes_in_3d_mask(dilated_mask)

    return filled_mask_2



######################################################
##################################################################
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

ventricle_mask=Infarct_Mask_filename_June20_data_512=resizeinto_512by512_and_flip(nib.load(sys.argv[1]).get_fdata())
ventricle_obb_mask = create_obb_mask_from_image_mask(ventricle_mask)
csf_mask_nib=nib.load(sys.argv[2])
# save_nifti_without_affine(ventricle_obb_mask, os.path.join(sys.argv[3],'ventricle_obb_mask.nii'))
array_img = nib.Nifti1Image(ventricle_obb_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
nib.save(array_img, os.path.join(sys.argv[3],'ventricle_obb_mask.nii'))
array_img = nib.Nifti1Image(ventricle_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
nib.save(array_img, os.path.join(sys.argv[3],'ventricle.nii'))

non_zero_slice_num=[]
ventricle_mask=nib.load(os.path.join(sys.argv[3],'ventricle_obb_mask.nii')).get_fdata()
# print(ventricle_mask.get_fdata().shape[2])
for slice_num in range(ventricle_mask.shape[2]):
    this_slice_sum=np.sum(ventricle_mask[:,:,slice_num])
    if this_slice_sum >0 :
        # print(this_slice_sum)
        non_zero_slice_num.append(slice_num)
if len(non_zero_slice_num)>0:
    upper_lower_limit_vent=["sessionID","scanID","original_nifti_filename",min(non_zero_slice_num),max(non_zero_slice_num)]
print(upper_lower_limit_vent)
upper_lower_limit_vent_df=pd.DataFrame(upper_lower_limit_vent).T
upper_lower_limit_vent_df.columns=['SESSION_ID','SCAN_ID','NIFTI_FILENAME','LOWER_SLICE_NUM','UPPER_SCLICE_NUM']
print(upper_lower_limit_vent_df)
# upper_lower_limit_vent_df.to_csv(os.path.join(SAVE_PATH,original_nifti_filename.split('.nii')[0]+'_ventricle_bounds.csv'),index=False)
upper_lower_limit_vent_df.to_csv(os.path.join(sys.argv[3],'ventricle_bounds.csv'),index=False)

ventricle_mask=nib.load( os.path.join(sys.argv[3],'ventricle.nii')).get_fdata()
# # Example usage:
# # Replace this with your actual 3D binary mask
# binary_mask = np.zeros((10, 100, 100), dtype=np.uint8)
# cv2.circle(binary_mask[5], (50, 50), 30, 1, -1)  # Example filled 2D circle in slice 5
# cv2.rectangle(binary_mask[6], (30, 30), (70, 70), 1, -1)  # Example filled rectangle in slice 6

# Fit ellipses for each 2D slice and create the 3D mask of stacked ellipses
# ellipse_3d_mask = fit_ellipse_to_3d_mask(ventricle_mask)
#
# # Print or visualize the result
# print("3D Mask with Fitted and Filled Ellipses for Each Slice:")
# print(ellipse_3d_mask)
# Example usage:
# Replace this with your actual 3D binary mask
# binary_mask = np.zeros((100, 100, 100), dtype=np.uint8)
# binary_mask[30:70, 30:70, 30:70] = 1  # Example filled 3D block

# Fit an ellipsoid to the mask and get the ellipsoid binary mask

#
# # Print or visualize the result
# print("3D Mask of the Best-Fit Ellipsoid:")
# print(ellipsoid_mask)

# filled_contour_mask =fill_dilate_and_fill_3d_mask(ventricle_mask, dilation_iterations=20) #process_3d_binary_mask(ventricle_mask, sigma=1) # process_3d_binary_mask(ventricle_mask)
# filled_contour_mask = fit_ellipsoid_to_3d_mask(filled_contour_mask)
# filled_contour_mask = smooth_3d_mask(filled_contour_mask, sigma=3)
# array_img = nib.Nifti1Image(filled_contour_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
# Example usage:
# binary_mask = np.zeros((100, 100, 100), dtype=np.uint8)
# binary_mask[30:70, 30:70, 30:70] = 1  # Example filled 3D block

# Process the 3D mask to get the closest non-zero voxels and OBB mask
closest_voxels, obb_mask = process_3d_mask(ventricle_mask)

print("Closest non-zero voxel coordinates to the centroids:")
print(closest_voxels)

print("\nOBB Mask:")
print(obb_mask)
print('closest_voxels')
print(np.array(closest_voxels.shape))
nib.save(array_img, os.path.join(sys.argv[3],'ventricle_contour.nii'))
#
# # Example usage:
# # Replace this with your actual 3D binary mask
# binary_mask = np.zeros((10, 10, 10), dtype=np.uint8)
# binary_mask[2:8, 2:8, 2:8] = 1  # Example filled 3D block
# binary_mask[4, 4, 4] = 0  # Create a hole inside the object
#
# # Perform the fill, dilate, and fill again process
# result_mask = fill_dilate_and_fill_3d_mask(binary_mask, dilation_iterations=2)
#
# # Print the result
# print("Original Mask with Hole:")
# print(binary_mask)
#
# print("\nMask after Fill, Dilate, and Fill Again:")
# print(result_mask)