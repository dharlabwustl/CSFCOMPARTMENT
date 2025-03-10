#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Segmentation: Region Growing</h1>
# 
# In this notebook we use one of the simplest segmentation approaches, region growing. We illustrate 
# the use of three variants of this family of algorithms. The common theme for all algorithms is that a voxel's neighbor is considered to be in the same class if its intensities are similar to the current voxel. The definition of similar is what varies:
# 
# * <b>ConnectedThreshold</b>: The neighboring voxel's intensity is within explicitly specified thresholds.
# * <b>ConfidenceConnected</b>: The neighboring voxel's intensity is within the implicitly specified bounds $\mu\pm c\sigma$, where $\mu$ is the mean intensity of the seed points, $\sigma$ their standard deviation and $c$ a user specified constant.
# * <b>VectorConfidenceConnected</b>: A generalization of the previous approach to vector valued images, for instance multi-spectral images or multi-parametric MRI. The neighboring voxel's intensity vector is within the implicitly specified bounds using the Mahalanobis distance $\sqrt{(\mathbf{x}-\mathbf{\mu})^T\Sigma^{-1}(\mathbf{x}-\mathbf{\mu})}<c$, where $\mathbf{\mu}$ is the mean of the vectors at the seed points, $\Sigma$ is the covariance matrix and $c$ is a user specified constant.
# 
# We will illustrate the usage of these three filters using a cranial MRI scan (T1 and T2) and attempt to segment one of the ventricles.

######## In[11]:


# To use interactive plots (mouse clicks, zooming, panning) we use the notebook back end. We want our graphs 
# to be embedded in the notebook, inline mode, this combination is defined by the magic "%matplotlib notebook".
# import numpy as np
# import scipy.linalg
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from vtk import *
import sys,inspect,subprocess
from scipy.ndimage import distance_transform_edt
# import six
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import glob
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
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from itertools import product

import nibabel as nib
import numpy as np

def process_nifti_mask(nifti_mask_path, output_path):
    """
    Given a NIfTI 3D binary mask, this function:
    - Finds the lower and upper slice limits based on voxel percentage thresholds.
    - Removes all voxels above the upper limit and below the lower limit.
    - Saves the modified mask as a new NIfTI file.

    Parameters:
        nifti_mask_path (str): Path to the input NIfTI mask file.
        output_path (str): Path to save the new processed NIfTI file.

    Returns:
        lower_slice (int): The lower slice index that meets the 2% threshold.
        upper_slice (int): The upper slice index that meets the 0.1% threshold.
    """
    # Load the NIfTI file
    img = nib.load(nifti_mask_path)
    data = img.get_fdata().astype(bool)  # Convert to binary mask (1 = mask, 0 = background)

    # Compute total number of non-zero voxels
    total_voxels = np.sum(data)

    if total_voxels == 0:
        return None, None  # No non-zero voxels in the mask

    # Compute the sum of non-zero voxels per slice along the Z-axis
    slice_sums = np.sum(data, axis=(0, 1))  # Sum of voxels per slice

    # Compute percentage contribution of each slice
    slice_percentages = (slice_sums / total_voxels) * 100

    # Find the first slice where % of non-zero voxels >= 2%
    lower_slice_candidates = np.where(slice_percentages >= 2.0)[0]
    lower_slice = int(lower_slice_candidates[0]) if lower_slice_candidates.size > 0 else None

    # Find the last slice where % of non-zero voxels >= 0.1%
    upper_slice_candidates = np.where(slice_percentages >= 0.1)[0]
    upper_slice = int(upper_slice_candidates[-1]) if upper_slice_candidates.size > 0 else None

    # Modify the mask: Set voxels outside the range to zero
    if lower_slice is not None and upper_slice is not None:
        data[:, :, :lower_slice] = 0  # Remove below lower limit
        data[:, :, upper_slice+1:] = 0  # Remove above upper limit

        # Save the processed mask as a new NIfTI file
        new_img = nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)
        nib.save(new_img, output_path)
        print(f"Processed mask saved to: {output_path}")

    return lower_slice, upper_slice

def expand_mask_distance(mask, expansion_factor=1.2):
    """
    Expands the mask using distance transform to increase volume by a given factor.

    Parameters:
        mask (numpy array): Binary mask (3D) where 1 represents the region of interest.
        expansion_factor (float): The desired volume increase (e.g., 1.2 for 20% increase).

    Returns:
        expanded_mask (numpy array): The expanded binary mask.
    """
    original_volume = np.sum(mask)

    # Compute distance transform (distance from non-mask to the nearest mask voxel)
    dist_transform = distance_transform_edt(~mask.astype(bool))

    # Find threshold distance to achieve the desired volume expansion
    sorted_distances = np.sort(dist_transform[dist_transform > 0])  # Exclude background
    target_volume = original_volume * expansion_factor

    if target_volume >= mask.size:  # Prevents expansion beyond the image bounds
        print("Expansion factor too large, filling entire volume.")
        return np.ones_like(mask, dtype=np.uint8)

    threshold_index = min(len(sorted_distances) - 1, int(target_volume - original_volume))
    threshold = sorted_distances[threshold_index]

    # Create expanded mask
    expanded_mask = dist_transform <= threshold

    return expanded_mask
def find_upper_lower_slices_1(nifti_mask_path):
    """
    Given a NIfTI 3D binary mask, this function returns the indices
    of the first (lower) and last (upper) slices that contain non-zero values.

    Parameters:
        nifti_mask_path (str): Path to the NIfTI mask file.

    Returns:
        lower_slice (int): Index of the lower (first) slice with non-zero values.
        upper_slice (int): Index of the upper (last) slice with non-zero values.
                           Returns (None, None) if no non-zero slices are found.
    """
    # Load the image using nibabel
    img = nib.load(nifti_mask_path)
    data = img.get_fdata()  # Get the image data as a numpy array

    # Sum across the first two dimensions for each slice along the third axis.
    # This gives us an array where each element represents the total intensity of a slice.
    slice_sums = np.sum(data, axis=(0, 1))

    # Find indices where the slice has non-zero sum (i.e., contains non-zero voxels)
    non_zero_slices = np.where(slice_sums > 0)[0]

    if non_zero_slices.size == 0:
        # No non-zero slices found
        return None, None

    # The first and last indices of non-zero slices represent the lower and upper bounds.
    lower_slice = int(non_zero_slices[0])
    upper_slice = int(non_zero_slices[-1])

    return lower_slice, upper_slice

# import nibabel as nib
# import numpy as np

def find_upper_lower_slices(nifti_mask_path):
    """
    Given a NIfTI 3D binary mask, this function returns the indices
    of the first (lower) and last (upper) slices that meet the threshold criteria.

    The lower slice is the first slice (starting from the bottom) that contains
    at least 2% of the total nonzero voxels.

    The upper slice is the last slice (starting from the top) that contains
    at least 0.1% of the total nonzero voxels.

    Parameters:
        nifti_mask_path (str): Path to the NIfTI mask file.

    Returns:
        lower_slice (int): Index of the lower (first) slice meeting the threshold.
        upper_slice (int): Index of the upper (last) slice meeting the threshold.
                           Returns (None, None) if no valid slices are found.
    """
    # Load the NIfTI file
    img = nib.load(nifti_mask_path)
    data = img.get_fdata().astype(bool)  # Convert to binary mask (1 = mask, 0 = background)

    # Compute total number of non-zero voxels
    total_voxels = np.sum(data)

    if total_voxels == 0:
        return None, None  # No non-zero voxels in the mask

    # Compute the sum of non-zero voxels per slice along the Z-axis
    slice_sums = np.sum(data, axis=(0, 1))  # Sum of voxels per slice

    # Compute percentage contribution of each slice
    slice_percentages = (slice_sums / total_voxels) * 100

    # Find the first slice where % of non-zero voxels >= 2%
    lower_slice_candidates = np.where(slice_percentages >= 2.0)[0]
    lower_slice = int(lower_slice_candidates[0]) if lower_slice_candidates.size > 0 else None

    # Find the last slice where % of non-zero voxels >= 0.1%
    upper_slice_candidates = np.where(slice_percentages >= 0.1)[0]
    upper_slice = int(upper_slice_candidates[-1]) if upper_slice_candidates.size > 0 else None

    return lower_slice, upper_slice


def compute_obb(binary_mask):
    """
    Compute the Oriented Bounding Box (OBB) for the non-zero elements in a 3D binary mask using PCA.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)

    Returns:
    - obb_corners: The 8 corner points of the OBB (in the original space)
    - pca_components: The principal components (directions of the OBB)
    - obb_center: The center point of the OBB (in the original space)
    - radii: The half-lengths of the OBB along the principal axes
    """
    # Find non-zero points (the object)
    points = np.column_stack(np.nonzero(binary_mask))

    if points.size == 0:
        # If no object, return empty results
        return None, None, None, None

    # Apply PCA to get the principal axes of the object
    pca = PCA(n_components=3)
    pca.fit(points)

    # Get the OBB center and its half-lengths (radii)
    obb_center = np.mean(points, axis=0)
    points_transformed = pca.transform(points)
    radii = (np.max(points_transformed, axis=0) - np.min(points_transformed, axis=0)) / 2

    # Get the 8 corners of the OBB
    obb_corners_local = np.array(list(product([-1, 1], repeat=3))) * radii
    obb_corners = pca.inverse_transform(obb_corners_local + pca.transform([obb_center])[0])

    return obb_corners, pca.components_, obb_center, radii

def create_obb_binary_mask(binary_mask, obb_corners):
    """
    Create a binary mask representing the OBB by filling in the OBB region.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - obb_corners: The 8 corner points of the OBB (in the original space)

    Returns:
    - obb_mask: 3D numpy array with the OBB region filled with 1s
    """
    # Create an empty mask of the same shape as the original binary mask
    obb_mask = np.zeros_like(binary_mask)

    # Get the bounding box of the OBB (min and max coordinates along each axis)
    x_min, y_min, z_min = np.floor(np.min(obb_corners, axis=0)).astype(int)
    x_max, y_max, z_max = np.ceil(np.max(obb_corners, axis=0)).astype(int)

    # Fill the region inside the OBB (axis-aligned for simplicity in this example)
    obb_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1

    return obb_mask

def proportional_subdivide_obb(obb_center, radii, pca_components, n_subdivisions):
    """
    Subdivide the Oriented Bounding Box (OBB) proportionally to its radii (dimensions) along the
    principal axes.

    Parameters:
    - obb_center: The center of the OBB (in the original space)
    - radii: The half-lengths of the OBB along each principal axis
    - pca_components: The principal components (rotation matrix of the OBB)
    - n_subdivisions: Number of divisions along the longest axis; the other axes will be divided
      proportionally.

    Returns:
    - centroids: A list of centroids of each sub-box (in the original space)
    """
    # Step 1: Proportionally subdivide based on the radii along each axis
    longest_axis_idx = np.argmax(radii)
    longest_axis_length = radii[longest_axis_idx]

    # Calculate the number of subdivisions along each axis based on the proportion to the longest axis
    subdivisions_per_axis = np.ceil(n_subdivisions * (radii / longest_axis_length)).astype(int)

    # Ensure that we at least have one subdivision along each axis
    subdivisions_per_axis[subdivisions_per_axis == 0] = 1

    # Step 2: Generate the ranges for the subdivisions along each axis in the local OBB space
    ranges = [np.linspace(-radii[i], radii[i], subdivisions_per_axis[i] + 1) for i in range(3)]

    centroids = []

    for i, j, k in product(range(subdivisions_per_axis[0]), range(subdivisions_per_axis[1]), range(subdivisions_per_axis[2])):
        # Min and max corners in local OBB space
        sub_box_min_local = np.array([ranges[0][i], ranges[1][j], ranges[2][k]])
        sub_box_max_local = np.array([ranges[0][i+1], ranges[1][j+1], ranges[2][k+1]])

        # Compute the centroid in local space
        centroid_local = (sub_box_min_local + sub_box_max_local) / 2

        # Convert centroid to original space using the OBB's rotation and translation
        centroid_original = pca_components.T @ centroid_local + obb_center
        centroids.append(centroid_original)

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

def process_binary_mask_with_obb(binary_mask, n_subdivisions=8):
    """
    Process the binary mask to compute the OBB, subdivide it, and find closest non-zero voxels.
    Also returns the OBB binary mask.

    Parameters:
    - binary_mask: 3D numpy array (binary mask of the object)
    - n_subdivisions: Number of equal-sized sub-boxes to divide the OBB into

    Returns:
    - closest_voxels: Coordinates of the closest non-zero voxels to the centroids
    - obb_mask: Binary mask of the OBB
    """
    # Step 1: Compute the OBB (Oriented Bounding Box)
    obb_corners, pca_components, obb_center, radii = compute_obb(binary_mask)

    if obb_corners is None:
        return None, None

    # Step 2: Subdivide the OBB and find the centroids of the subdivided boxes
    centroids = proportional_subdivide_obb(obb_center, radii, pca_components, n_subdivisions)

    # Step 3: Find the closest non-zero voxels to the centroids in the original binary mask
    closest_voxels = find_closest_non_zero_voxel(binary_mask, centroids)

    # Step 4: Create the binary mask of the OBB
    obb_mask = create_obb_binary_mask(binary_mask, obb_corners)

    return closest_voxels, obb_mask

# Example usage:


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

def sortSecond(val):
    return val[1]
def calculate_volume(nii_img,mask_img):

    resol= np.prod(np.array(nii_img.header["pixdim"][1:4]))
    mask_data_flatten= mask_img.flatten()
    num_pixel_gt_0=mask_data_flatten[np.where(mask_data_flatten>0)]
    return (resol * num_pixel_gt_0.size)/1000

def slicenum_at_end(image):
    image_copy=np.zeros([image.shape[1],image.shape[2],image.shape[0]])
    for i in range(image.shape[0]):
        image_copy[:,:,i]=image[i,:,:]

    return image_copy



def subtract_binary_1(binary_imageBig,binary_imageSmall):
    binary_imageBigCopy=np.copy(binary_imageBig)
    binary_imageBigCopy[binary_imageSmall>0]=0
    return binary_imageBigCopy



def get_ventricles_range(numpy_array_3D_mask):
    zoneV_min_z=0
    zoneV_max_z=0
    counter=0
    for each_slice_num in range(0,numpy_array_3D_mask.shape[0]):
        pixel_gt_0 = np.sum(numpy_array_3D_mask[each_slice_num,:,:])
        if pixel_gt_0>0.0:
            if counter==0:
                zoneV_min_z=each_slice_num
                counter=counter+1
            zoneV_max_z=each_slice_num
#    print("zoneV_min_z")
#    print(zoneV_min_z)
#    print("zoneV_max_z")
#    print(zoneV_max_z)
    return zoneV_min_z,zoneV_max_z



def divideintozones_v1(filename_gray,filename_mask,filename_bet):
    try:
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image

        file_gray = filename_gray
        reader_gray = sitk.ImageFileReader()
        reader_gray.SetImageIO("NiftiImageIO")
        reader_gray.SetFileName(file_gray)


        gray_scale_file=filename_gray
        gray_image=nib.load(gray_scale_file)



        file =filename_mask
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(file)
        img_T1 = reader.Execute();
        img_T1_Copy=img_T1
        imagenparray=sitk.GetArrayFromImage(img_T1)

        if np.sum(imagenparray)>200:
            img_T1=img_T1*255

            img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1) ,sitk.sitkUInt8)

            file1 = filename_bet
            reader1 = sitk.ImageFileReader()
            reader1.SetImageIO("NiftiImageIO")
            reader1.SetFileName(file1)
            img_T1_bet = reader1.Execute();
            cc1 = sitk.ConnectedComponent(img_T1_bet>0)
            stats1 = sitk.LabelIntensityStatisticsImageFilter()
            stats1.Execute(cc1,img_T1_bet)
    #
            cc = sitk.ConnectedComponent(img_T1_255>0)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(cc,img_T1)

            maxsize_comp_1=0
            id_of_maxsize_comp_1=0

            for l in range(len(stats1.GetLabels())):
                if stats1.GetPhysicalSize(stats1.GetLabels()[l])>maxsize_comp_1:
                    maxsize_comp_1=stats1.GetPhysicalSize(stats1.GetLabels()[l])

                    id_of_maxsize_comp_1=l
            csf_ids=[]
            for l in range(len(stats.GetLabels())):

                csf_ids.append([l,stats.GetPhysicalSize(stats.GetLabels()[l])])
            csf_ids.sort(key = sortSecond, reverse = True)
            # subprocess.call("echo " + "SUCCEEDED AT ::{}  > error.txt".format(inspect.stack()[0][3]) ,shell=True )
            first_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[0][0]]))
            second_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[1][0]]))
            bet_centroid=np.array(stats.GetCentroid(stats.GetLabels()[id_of_maxsize_comp_1]))
            first2bet_centroid=np.linalg.norm(first_seg_centroid - bet_centroid)
            second2bet_centroid=np.linalg.norm(second_seg_centroid - bet_centroid)
            if first2bet_centroid< second2bet_centroid:
                id_of_maxsize_comp=csf_ids[0][0]

            else:
                if stats.GetPhysicalSize(stats.GetLabels()[csf_ids[1][0]]) > 10000:
                    id_of_maxsize_comp=csf_ids[1][0]

                else:
                    id_of_maxsize_comp=csf_ids[0][0]

            initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])]
            seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)

            zoneV_min_z,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
            subtracted_image=subtract_binary_1(sitk.GetArrayFromImage(img_T1_Copy),sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
            subtracted_image=sitk.GetImageFromArray(subtracted_image)
            above_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            above_ventricle_image[0:zoneV_max_z+1,:,:]=0
            covering_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            covering_ventricle_image[0:zoneV_min_z+1,:,:]=0
            covering_ventricle_image[zoneV_max_z+1:above_ventricle_image.shape[0],:,:]=0
            below_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            below_ventricle_image[zoneV_min_z:above_ventricle_image.shape[0],:,:]=0

            above_ventricle_image_sitkimg=sitk.GetImageFromArray(above_ventricle_image)
            above_ventricle_image_sitkimg.CopyInformation(img_T1_bet)
            # sulci_vol_below_vent=calculate_volume(gray_image,below_ventricle_image)
            below_ventricle_image_sitkimg=sitk.GetImageFromArray(below_ventricle_image)
            below_ventricle_image_sitkimg.CopyInformation(img_T1_bet)
            # sulci_vol_at_vent=calculate_volume(gray_image,covering_ventricle_image)

            covering_ventricle_image_sitkimg=sitk.GetImageFromArray(covering_ventricle_image)
            covering_ventricle_image_sitkimg.CopyInformation(img_T1_bet)


            subtracted_image.CopyInformation( img_T1_bet)
            sitk.WriteImage(subtracted_image, filename_gray.split(".nii")[0]+ "_sulci_total.nii.gz", True)

            seg_explicit_thresholds.CopyInformation( img_T1_bet)
            sitk.WriteImage(seg_explicit_thresholds, filename_gray.split(".nii")[0]+ "_ventricle_total.nii.gz", True)

            sitk.WriteImage(above_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_above_ventricle.nii.gz", True)

            sitk.WriteImage(below_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_below_ventricle.nii.gz", True)

            sitk.WriteImage(covering_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_at_ventricle.nii.gz", True)

            subprocess.call("echo " + "SUCCEEDED AT ::{}  > error.txt".format(inspect.stack()[0][3]) ,shell=True )


    except:
        subprocess.call("echo " + "FAILED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )



    return  sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent
    # return sulci_vol, ventricle_vol,leftcountven*resol,rightcountven*resol,leftcountsul*resol,rightcountsul*resol,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent #seg_explicit_thresholds, subtracted_image



def divideintozones_v1_with_vent_bound(filename_gray,filename_mask,filename_bet,zoneV_min_z,zoneV_max_z):
    try:
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image

        file_gray = filename_gray
        reader_gray = sitk.ImageFileReader()
        reader_gray.SetImageIO("NiftiImageIO")
        reader_gray.SetFileName(file_gray)


        gray_scale_file=filename_gray
        gray_image=nib.load(gray_scale_file)
        zoneV_min_z1=zoneV_min_z
        zoneV_max_z1=zoneV_max_z

        file =filename_mask
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(file)
        img_T1_1 = reader.Execute();
        # img_T1_Copy=img_T1
        ####################################################
        img_T1_temp_np=sitk.GetArrayFromImage(img_T1_1)
        img_T1_temp_np_alllabels=np.copy(img_T1_temp_np)
        img_T1_temp_np_alllabels=slicenum_at_end(img_T1_temp_np_alllabels)
        img_T1_temp_np[img_T1_temp_np>1]=0.0
        img_T1_1_forsubtract_np=np.copy(img_T1_temp_np)
        img_T1_1_forsubtract_itk=sitk.GetImageFromArray(img_T1_1_forsubtract_np)
        img_T1_1_forsubtract_itk.CopyInformation(img_T1_1)
        img_T1_temp_np[0:zoneV_min_z1,:,:]=0.0
        img_T1_temp_np[zoneV_max_z1+1:img_T1_temp_np.shape[0],:,:]=0.0
        img_T1_temp_itk=sitk.GetImageFromArray(img_T1_temp_np)
        img_T1_temp_itk.CopyInformation(img_T1_1)


        img_T1_temp_np_Ven=np.copy(img_T1_temp_np)
        #    img_T1_temp_np_Ven[0:zoneV_min_z1,:,:]=0.0
        #    img_T1_temp_np_Ven[zoneV_max_z1:img_T1_temp_np.shape[0],:,:]=0.0
        img_T1_temp_np_Ven_itk=sitk.GetImageFromArray(img_T1_temp_np_Ven)
        img_T1_temp_np_Ven_itk.CopyInformation(img_T1_1)

        img_T2=img_T1_temp_itk
        img_T2_Copy=img_T2

        img_T1=img_T1_temp_np_Ven_itk
        img_T1_Copy=img_T1
    


        ###############################################
        imagenparray=sitk.GetArrayFromImage(img_T1)

        if np.sum(imagenparray)>200:
            img_T1=img_T1*255

            img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1) ,sitk.sitkUInt8)

            file1 = filename_bet
            reader1 = sitk.ImageFileReader()
            reader1.SetImageIO("NiftiImageIO")
            reader1.SetFileName(file1)
            img_T1_bet = reader1.Execute();
            cc1 = sitk.ConnectedComponent(img_T1_bet>0)
            stats1 = sitk.LabelIntensityStatisticsImageFilter()
            stats1.Execute(cc1,img_T1_bet)
    #
            cc = sitk.ConnectedComponent(img_T1_255>0)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(cc,img_T1)

            maxsize_comp_1=0
            id_of_maxsize_comp_1=0

            for l in range(len(stats1.GetLabels())):
                if stats1.GetPhysicalSize(stats1.GetLabels()[l])>maxsize_comp_1:
                    maxsize_comp_1=stats1.GetPhysicalSize(stats1.GetLabels()[l])

                    id_of_maxsize_comp_1=l
            csf_ids=[]
            for l in range(len(stats.GetLabels())):

                csf_ids.append([l,stats.GetPhysicalSize(stats.GetLabels()[l])])
            csf_ids.sort(key = sortSecond, reverse = True)
            # subprocess.call("echo " + "SUCCEEDED AT ::{}  > error.txt".format(inspect.stack()[0][3]) ,shell=True )
            first_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[0][0]]))
            second_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[1][0]]))
            bet_centroid=np.array(stats.GetCentroid(stats.GetLabels()[id_of_maxsize_comp_1]))
            first2bet_centroid=np.linalg.norm(first_seg_centroid - bet_centroid)
            second2bet_centroid=np.linalg.norm(second_seg_centroid - bet_centroid)
            if first2bet_centroid< second2bet_centroid:
                id_of_maxsize_comp=csf_ids[0][0]

            else:
                if stats.GetPhysicalSize(stats.GetLabels()[csf_ids[1][0]]) > 10000:
                    id_of_maxsize_comp=csf_ids[1][0]

                else:
                    id_of_maxsize_comp=csf_ids[0][0]

            initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])]
            seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)

            # zoneV_min_z,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
            subtracted_image=subtract_binary_1(sitk.GetArrayFromImage(img_T1_1),sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
            subtracted_image=sitk.GetImageFromArray(subtracted_image)
            above_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            above_ventricle_image[0:zoneV_max_z+1,:,:]=0
            covering_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            covering_ventricle_image[0:zoneV_min_z+1,:,:]=0
            covering_ventricle_image[zoneV_max_z+1:above_ventricle_image.shape[0],:,:]=0
            below_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            below_ventricle_image[zoneV_min_z:above_ventricle_image.shape[0],:,:]=0

            above_ventricle_image_sitkimg=sitk.GetImageFromArray(above_ventricle_image)
            above_ventricle_image_sitkimg.CopyInformation(img_T1_bet)
            # sulci_vol_below_vent=calculate_volume(gray_image,below_ventricle_image)
            below_ventricle_image_sitkimg=sitk.GetImageFromArray(below_ventricle_image)
            below_ventricle_image_sitkimg.CopyInformation(img_T1_bet)
            # sulci_vol_at_vent=calculate_volume(gray_image,covering_ventricle_image)

            covering_ventricle_image_sitkimg=sitk.GetImageFromArray(covering_ventricle_image)
            covering_ventricle_image_sitkimg.CopyInformation(img_T1_bet)


            subtracted_image.CopyInformation( img_T1_bet)
            sitk.WriteImage(subtracted_image, filename_gray.split(".nii")[0]+ "_sulci_total.nii.gz", True)

            seg_explicit_thresholds.CopyInformation( img_T1_bet)
            sitk.WriteImage(seg_explicit_thresholds, filename_gray.split(".nii")[0]+ "_ventricle_total.nii.gz", True)

            sitk.WriteImage(above_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_above_ventricle.nii.gz", True)

            sitk.WriteImage(below_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_below_ventricle.nii.gz", True)

            sitk.WriteImage(covering_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_at_ventricle.nii.gz", True)

            subprocess.call("echo " + "SUCCEEDED AT ::{}  > error.txt".format(inspect.stack()[0][3]) ,shell=True )
            subprocess.call("echo " + "SUCCEEDED AT ::{}:{}:{}  > error.txt".format(inspect.stack()[0][3],zoneV_max_z,zoneV_min_z) ,shell=True )


    except     Exception as e:
    # Print the error message
        print(f"Error: {e}")
        subprocess.call("echo " + "FAILED AT ::{}::{}  >> error.txt".format(inspect.stack()[0][3],e) ,shell=True )




    return  sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent
    # return sulci_vol, ventricle_vol,leftcountven*resol,rightcountven*resol,leftcountsul*resol,rightcountsul*resol,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent #seg_explicit_thresholds, subtracted_image

def divideintozones_with_vent_obb_with_four_centroid(filename_gray,filename_mask,filename_bet,filename_vent_obb,closest_voxels,zoneV_min_z,zoneV_max_z):
    try:
        print('closest_voxels at divideintozones_with_vent_obb_with_four_centroid')
        print(closest_voxels)
        print([(closest_voxels[0][0],closest_voxels[0][1],closest_voxels[0][2])])
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image
        #######################
        reader_filename_vent_nonlinmask = sitk.ImageFileReader()
        reader_filename_vent_nonlinmask.SetImageIO("NiftiImageIO")
        reader_filename_vent_nonlinmask.SetFileName('/workinginput/ventricle_contour.nii')
        ventricle_nonlin_mask = reader_filename_vent_nonlinmask.Execute()
        ventricle_nonlin_mask_np=sitk.GetArrayFromImage(ventricle_nonlin_mask)

        #################################

        reader_filename_vent_obb = sitk.ImageFileReader()
        reader_filename_vent_obb.SetImageIO("NiftiImageIO")
        reader_filename_vent_obb.SetFileName(filename_vent_obb)
        ventricle_obb = reader_filename_vent_obb.Execute()
        ventricle_obb_np=sitk.GetArrayFromImage(ventricle_obb)
        ########################
        file_gray = filename_gray
        reader_gray = sitk.ImageFileReader()
        reader_gray.SetImageIO("NiftiImageIO")
        reader_gray.SetFileName(file_gray)


        gray_scale_file=filename_gray
        gray_image=nib.load(gray_scale_file)
        zoneV_min_z1=zoneV_min_z
        zoneV_max_z1=zoneV_max_z

        file =filename_mask
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(file)
        img_T1_1 = reader.Execute();
        # img_T1_Copy=img_T1
        ####################################################
        img_T1_temp_np=sitk.GetArrayFromImage(img_T1_1)
        img_T1_temp_np_alllabels=np.copy(img_T1_temp_np)
        img_T1_temp_np_alllabels=slicenum_at_end(img_T1_temp_np_alllabels)
        img_T1_temp_np[img_T1_temp_np>1]=0.0
        img_T1_1_forsubtract_np=np.copy(img_T1_temp_np)
        img_T1_1_forsubtract_itk=sitk.GetImageFromArray(img_T1_1_forsubtract_np)
        img_T1_1_forsubtract_itk.CopyInformation(img_T1_1)
        img_T1_temp_np[ventricle_obb_np<1]=0.0
        # img_T1_temp_np[ventricle_nonlin_mask_np<1]=0.0
        # img_T1_temp_np[0:zoneV_min_z1,:,:]=0.0
        # img_T1_temp_np[zoneV_max_z1+1:img_T1_temp_np.shape[0],:,:]=0.0
        img_T1_temp_itk=sitk.GetImageFromArray(img_T1_temp_np)
        img_T1_temp_itk.CopyInformation(img_T1_1)


        img_T1_temp_np_Ven=np.copy(img_T1_temp_np)
        #    img_T1_temp_np_Ven[0:zoneV_min_z1,:,:]=0.0
        #    img_T1_temp_np_Ven[zoneV_max_z1:img_T1_temp_np.shape[0],:,:]=0.0
        img_T1_temp_np_Ven_itk=sitk.GetImageFromArray(img_T1_temp_np_Ven)
        img_T1_temp_np_Ven_itk.CopyInformation(img_T1_1)

        img_T2=img_T1_temp_itk
        img_T2_Copy=img_T2

        img_T1=img_T1_temp_np_Ven_itk
        img_T1_Copy=img_T1



        ###############################################
        imagenparray=sitk.GetArrayFromImage(img_T1)

        if np.sum(imagenparray)>200:
            img_T1=img_T1*255

            img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1) ,sitk.sitkUInt8)

            file1 = filename_bet
            reader1 = sitk.ImageFileReader()
            reader1.SetImageIO("NiftiImageIO")
            reader1.SetFileName(file1)
            img_T1_bet = reader1.Execute();
            cc1 = sitk.ConnectedComponent(img_T1_bet>0)
            stats1 = sitk.LabelIntensityStatisticsImageFilter()
            stats1.Execute(cc1,img_T1_bet)
            #
            cc = sitk.ConnectedComponent(img_T1_255>0)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(cc,img_T1)

            maxsize_comp_1=0
            id_of_maxsize_comp_1=0

            for l in range(len(stats1.GetLabels())):
                if stats1.GetPhysicalSize(stats1.GetLabels()[l])>maxsize_comp_1:
                    maxsize_comp_1=stats1.GetPhysicalSize(stats1.GetLabels()[l])

                    id_of_maxsize_comp_1=l
            csf_ids=[]
            for l in range(len(stats.GetLabels())):

                csf_ids.append([l,stats.GetPhysicalSize(stats.GetLabels()[l])])
            csf_ids.sort(key = sortSecond, reverse = True)
            # subprocess.call("echo " + "SUCCEEDED AT ::{}  > error.txt".format(inspect.stack()[0][3]) ,shell=True )
            first_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[0][0]]))
            second_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[1][0]]))
            bet_centroid=np.array(stats.GetCentroid(stats.GetLabels()[id_of_maxsize_comp_1]))
            first2bet_centroid=np.linalg.norm(first_seg_centroid - bet_centroid)
            second2bet_centroid=np.linalg.norm(second_seg_centroid - bet_centroid)
            if first2bet_centroid< second2bet_centroid:
                id_of_maxsize_comp=csf_ids[0][0]

            else:
                if stats.GetPhysicalSize(stats.GetLabels()[csf_ids[1][0]]) > 10000:
                    id_of_maxsize_comp=csf_ids[1][0]

                else:
                    id_of_maxsize_comp=csf_ids[0][0]

            initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])]
            print('initial_seed_point_indexes::{}'.format(type(initial_seed_point_indexes[0])))
            initial_seed_point_indexes=[(int(closest_voxels[0][0]),int(closest_voxels[0][1]),int(closest_voxels[0][2]))]
            print('initial_seed_point_indexes_nw::{}'.format(type(initial_seed_point_indexes[0])))
            seg_explicit_thresholds =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[1][0]),int(closest_voxels[1][1]),int(closest_voxels[1][2]))]
            seg_explicit_thresholds1 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[2][0]),int(closest_voxels[2][1]),int(closest_voxels[2][2]))]
            seg_explicit_thresholds2 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[3][0]),int(closest_voxels[3][1]),int(closest_voxels[3][2]))]
            seg_explicit_thresholds3 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[4][0]),int(closest_voxels[4][1]),int(closest_voxels[4][2]))]
            seg_explicit_thresholds4 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[5][0]),int(closest_voxels[5][1]),int(closest_voxels[5][2]))]
            seg_explicit_thresholds5 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[6][0]),int(closest_voxels[6][1]),int(closest_voxels[6][2]))]
            seg_explicit_thresholds6 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[7][0]),int(closest_voxels[7][1]),int(closest_voxels[7][2]))]
            seg_explicit_thresholds7 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[8][0]),int(closest_voxels[8][1]),int(closest_voxels[8][2]))]
            seg_explicit_thresholds8 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[9][0]),int(closest_voxels[9][1]),int(closest_voxels[9][2]))]
            seg_explicit_thresholds9 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[10][0]),int(closest_voxels[10][1]),int(closest_voxels[10][2]))]
            seg_explicit_thresholds10 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            initial_seed_point_indexes=[(int(closest_voxels[11][0]),int(closest_voxels[11][1]),int(closest_voxels[11][2]))]
            seg_explicit_thresholds11 =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)


            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds > 0, seg_explicit_thresholds1 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds2 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds3 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds4 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds5 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds5 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds6 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds7 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds8 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds9 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds10 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds11 > 0)

            zoneV_min_z,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
            subtracted_image=subtract_binary_1(sitk.GetArrayFromImage(img_T1_1),sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
            subtracted_image=sitk.GetImageFromArray(subtracted_image)
            above_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            above_ventricle_image[0:zoneV_max_z+1,:,:]=0
            covering_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            covering_ventricle_image[0:zoneV_min_z,:,:]=0
            covering_ventricle_image[zoneV_max_z+1:above_ventricle_image.shape[0],:,:]=0
            below_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
            below_ventricle_image[zoneV_min_z:above_ventricle_image.shape[0],:,:]=0

            above_ventricle_image_sitkimg=sitk.GetImageFromArray(above_ventricle_image)
            above_ventricle_image_sitkimg.CopyInformation(img_T1_bet)
            # sulci_vol_below_vent=calculate_volume(gray_image,below_ventricle_image)
            below_ventricle_image_sitkimg=sitk.GetImageFromArray(below_ventricle_image)
            below_ventricle_image_sitkimg.CopyInformation(img_T1_bet)
            # sulci_vol_at_vent=calculate_volume(gray_image,covering_ventricle_image)

            covering_ventricle_image_sitkimg=sitk.GetImageFromArray(covering_ventricle_image)
            covering_ventricle_image_sitkimg.CopyInformation(img_T1_bet)


            subtracted_image.CopyInformation( img_T1_bet)
            sitk.WriteImage(subtracted_image, filename_gray.split(".nii")[0]+ "_sulci_total.nii.gz", True)

            seg_explicit_thresholds.CopyInformation( img_T1_bet)
            sitk.WriteImage(seg_explicit_thresholds, filename_gray.split(".nii")[0]+ "_ventricle_total.nii.gz", True)

            sitk.WriteImage(above_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_above_ventricle.nii.gz", True)

            sitk.WriteImage(below_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_below_ventricle.nii.gz", True)

            sitk.WriteImage(covering_ventricle_image_sitkimg, filename_gray.split(".nii")[0]+ "_sulci_at_ventricle.nii.gz", True)

            subprocess.call("echo " + "SUCCEEDED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
            subprocess.call("echo " + "SUCCEEDED AT ::{}:{}:{}  >> error.txt".format(inspect.stack()[0][3],zoneV_max_z,zoneV_min_z) ,shell=True )


    except     Exception as e:
        # Print the error message
        print(f"Error: {e}")
        subprocess.call("echo " + "FAILED AT ::{}::{}  >> error.txt".format(inspect.stack()[0][3],e) ,shell=True )




    return  sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent
import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

def scale_mask(nifti_mask_path, output_path, scale_factor=1.5):
    """
    Scales a 3D binary mask by a given factor while keeping it centered.

    Parameters:
        nifti_mask_path (str): Path to the input NIfTI mask file.
        output_path (str): Path to save the scaled NIfTI file.
        scale_factor (float): Scaling factor for mask enlargement (default=1.5).

    Returns:
        None
    """
    # Load NIfTI mask
    img = nib.load(nifti_mask_path)
    mask = img.get_fdata().astype(np.uint8)

    # Compute the center of mass of the mask
    center = ndi.center_of_mass(mask)

    # Define affine transformation matrix for scaling
    scale_matrix = np.array([
        [1/scale_factor, 0, 0, 0],
        [0, 1/scale_factor, 0, 0],
        [0, 0, 1/scale_factor, 0],
        [0, 0, 0, 1]
    ])

    # Compute translation to keep the mask centered
    translation = np.array(mask.shape) / 2 - np.array(center)
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    # Apply the scaling transformation
    scaled_mask = ndi.affine_transform(mask.astype(float), scale_matrix, offset=-translation, order=1)

    # Convert back to binary mask (thresholding)
    scaled_mask = (scaled_mask > 0.5).astype(np.uint8)

    # Save the new scaled mask as a NIfTI file
    new_img = nib.Nifti1Image(scaled_mask, img.affine, img.header)
    nib.save(new_img, output_path)

    print(f"Scaled mask saved to: {output_path}")

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

def scale_mask_xyz(nifti_mask_path, output_path, scale_factor=1.5):
    """
    Scales a 3D binary mask in X, Y, and Z dimensions but ensures the final image
    has the same number of slices as the original.

    Parameters:
        nifti_mask_path (str): Path to the input NIfTI mask file.
        output_path (str): Path to save the scaled NIfTI file.
        scale_factor (float): Scaling factor for all dimensions (default=1.5).

    Returns:
        None
    """
    # Load NIfTI mask
    img = nib.load(nifti_mask_path)
    mask = img.get_fdata().astype(np.uint8)

    # Get original shape
    orig_shape = mask.shape

    # Compute the center of mass to maintain alignment
    center = ndi.center_of_mass(mask)

    # Scale in X, Y, and Z
    scaled_mask = ndi.zoom(mask, zoom=scale_factor, order=1)

    # Resample Z-axis back to original number of slices
    target_z_slices = orig_shape[2]
    scaled_z_indices = np.linspace(0, scaled_mask.shape[2] - 1, target_z_slices)
    resampled_mask = np.zeros((scaled_mask.shape[0], scaled_mask.shape[1], target_z_slices))

    for i, z in enumerate(scaled_z_indices):
        resampled_mask[:, :, i] = ndi.map_coordinates(scaled_mask,
                                                      np.array([[x for x in range(scaled_mask.shape[0])],
                                                                [y for y in range(scaled_mask.shape[1])],
                                                                [np.full((scaled_mask.shape[0], scaled_mask.shape[1]), z)]]
                                                               ).reshape(3, -1),
                                                      order=1).reshape(scaled_mask.shape[:2])

    # Ensure the shape matches the original image
    new_mask = np.zeros_like(mask)

    # Compute new scaled shape
    new_x, new_y, _ = resampled_mask.shape
    orig_x, orig_y, _ = orig_shape

    # Compute start positions to center the mask
    start_x = max((orig_x - new_x) // 2, 0)
    start_y = max((orig_y - new_y) // 2, 0)

    # Compute end positions
    end_x = min(start_x + new_x, orig_x)
    end_y = min(start_y + new_y, orig_y)

    # Place the resampled mask within the original frame
    new_mask[start_x:end_x, start_y:end_y, :] = resampled_mask[:end_x-start_x, :end_y-start_y, :]

    # Convert to binary mask
    new_mask = (new_mask > 0.5).astype(np.uint8)

    # Save the new scaled mask as a NIfTI file
    new_img = nib.Nifti1Image(new_mask, img.affine, img.header)
    nib.save(new_img, output_path)

    print(f"Scaled mask saved to: {output_path}")

def expand_nifti_mask(nifti_path, output_stl_path, scale_factor=1.5):
    # Load NIfTI mask
    nii = nib.load(nifti_path)
    mask = nii.get_fdata() > 0  # Ensure binary mask

    # Compute Euclidean Distance Transform (EDT)
    dist_inside = ndi.distance_transform_edt(mask)
    dist_outside = ndi.distance_transform_edt(~mask)

    # Determine expansion threshold based on desired scale factor
    max_dist = np.max(dist_inside)  # Largest internal distance
    expansion_threshold = max_dist * (scale_factor - 1)  # Extra distance to expand

    # Create expanded mask
    expanded_mask = (dist_inside >= -expansion_threshold) | (dist_outside <= expansion_threshold)

    # Convert to Mesh using Marching Cubes
    verts, faces, _, _ = measure.marching_cubes(expanded_mask.astype(np.uint8), level=0.5)

    # Create Trimesh object and save as STL
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_stl_path)

    print(f"Expanded 3D model saved to: {output_stl_path}")

def compute_obb_1(data):
    """
    Computes the Oriented Bounding Box (OBB) of a 3D ventricle mask
    and returns a new binary mask containing the filled OBB.

    The OBB is clipped to ensure it does not exceed the original highest and lowest slices.

    Parameters:
        data (numpy.ndarray): 3D binary mask of the ventricle (1 = mask, 0 = background).

    Returns:
        obb_mask (numpy.ndarray): 3D binary mask of the computed OBB.
    """
    data = data.astype(bool)  # Ensure binary mask (1 = mask, 0 = background)

    # Extract nonzero voxel coordinates
    coords = np.array(np.nonzero(data)).T  # Shape (N, 3), where N is the number of foreground voxels

    if coords.shape[0] == 0:
        print("No nonzero voxels found in the mask.")
        return np.zeros_like(data, dtype=np.uint8)  # Return empty mask

    # Get the original lowest and highest Z-slices
    original_z_min = np.min(coords[:, 2])
    original_z_max = np.max(coords[:, 2])

    # Apply PCA to find the orientation of the mask
    pca = PCA(n_components=3)
    transformed_coords = pca.fit_transform(coords)

    # Compute the bounding box in the transformed space
    min_bounds = np.min(transformed_coords, axis=0)
    max_bounds = np.max(transformed_coords, axis=0)

    # Generate grid points inside the bounding box in PCA space
    grid_x, grid_y, grid_z = np.mgrid[
                             min_bounds[0]:max_bounds[0]:1,
                             min_bounds[1]:max_bounds[1]:1,
                             min_bounds[2]:max_bounds[2]:1
                             ]

    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    # Transform the grid points back to original space
    obb_points = pca.inverse_transform(grid_points)

    # Create a new mask for the OBB
    obb_mask = np.zeros_like(data, dtype=np.uint8)

    # Convert OBB voxel locations into integer indices
    obb_voxel_indices = np.round(obb_points).astype(int)

    # Ensure indices are within valid range
    valid_indices = np.all((obb_voxel_indices >= 0) & (obb_voxel_indices < np.array(data.shape)), axis=1)
    obb_voxel_indices = obb_voxel_indices[valid_indices]

    # Clip the OBB to respect the original Z-limits
    obb_voxel_indices = obb_voxel_indices[
        (obb_voxel_indices[:, 2] >= original_z_min) & (obb_voxel_indices[:, 2] <= original_z_max)
        ]

    # Set the OBB region in the mask
    obb_mask[obb_voxel_indices[:, 0], obb_voxel_indices[:, 1], obb_voxel_indices[:, 2]] = 1

    # Fill the interior of the bounding box to ensure a complete volume
    obb_mask = binary_fill_holes(obb_mask).astype(np.uint8)

    return obb_mask

# ##########################
cistern_mask=resizeinto_512by512_and_flip(nib.load(sys.argv[1]).get_fdata())

cistern_obb_mask=compute_obb_1(cistern_mask)
# # Example usage:
####################
print('sys.argv[2]::{}'.format(sys.argv[2]))
csf_mask_nib=nib.load(sys.argv[2])
# save_nifti_without_affine(ventricle_obb_mask, os.path.join(sys.argv[3],'ventricle_obb_mask.nii'))
array_img = nib.Nifti1Image(cistern_obb_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
nib.save(array_img, os.path.join(sys.argv[3],'cistern_obb_mask.nii'))
array_img = nib.Nifti1Image(cistern_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
nib.save(array_img, os.path.join(sys.argv[3],'cistern_after_deepreg.nii'))
