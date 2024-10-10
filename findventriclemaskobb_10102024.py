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
    # closest_voxels = find_closest_non_zero_voxel(binary_mask, centroids)

    return centroids, obb_mask

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
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds > 0, seg_explicit_thresholds1 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds2 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds3 > 0)

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

            subprocess.call("echo " + "SUCCEEDED AT ::{}  > error.txt".format(inspect.stack()[0][3]) ,shell=True )
            subprocess.call("echo " + "SUCCEEDED AT ::{}:{}:{}  > error.txt".format(inspect.stack()[0][3],zoneV_max_z,zoneV_min_z) ,shell=True )


    except     Exception as e:
        # Print the error message
        print(f"Error: {e}")
        subprocess.call("echo " + "FAILED AT ::{}::{}  >> error.txt".format(inspect.stack()[0][3],e) ,shell=True )




    return  sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent



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
centroids, obb_mask = process_3d_mask(ventricle_mask)

print("Closest non-zero voxel coordinates to the centroids:")
print(centroids)

print("\nOBB Mask:")
print(obb_mask)
print('closest_voxels')
print(np.array(centroids).shape)
nib.save(array_img, os.path.join(sys.argv[3],'ventricle_contour.nii'))
filename_gray=sys.argv[4]
filename_mask=sys.argv[2]
filename_bet=sys.argv[5]
filename_vent_obb=os.path.join(sys.argv[3],'ventricle_obb_mask.nii')
zoneV_min_z=0
zoneV_max_z=0
closest_voxels=find_closest_non_zero_voxel(nib.load(filename_mask).get_fdata(), centroids)
divideintozones_with_vent_obb_with_four_centroid(filename_gray,filename_mask,filename_bet,filename_vent_obb,closest_voxels,zoneV_min_z,zoneV_max_z)
