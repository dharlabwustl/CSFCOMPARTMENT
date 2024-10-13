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
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from itertools import product

import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate

from matplotlib.path import Path
import sys,inspect,subprocess
# import six
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import glob
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

def divideintozones_with_vent_convexhull(filename_gray,filename_mask,filename_bet,filename_vent_conv_hull,closest_voxels):
    try:
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image
        #######################


        #################################

        reader_filename_vent_obb = sitk.ImageFileReader()
        reader_filename_vent_obb.SetImageIO("NiftiImageIO")
        reader_filename_vent_obb.SetFileName(filename_vent_conv_hull)
        ventricle_obb = reader_filename_vent_obb.Execute()
        ventricle_obb_np=sitk.GetArrayFromImage(ventricle_obb)
        ########################
        file_gray = filename_gray
        reader_gray = sitk.ImageFileReader()
        reader_gray.SetImageIO("NiftiImageIO")
        reader_gray.SetFileName(file_gray)


        gray_scale_file=filename_gray
        gray_image=nib.load(gray_scale_file)
        # zoneV_min_z1=zoneV_min_z
        # zoneV_max_z1=zoneV_max_z

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

            # initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])]
            # seg_explicit_thresholds=img_T1 ##sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
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


            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds > 0, seg_explicit_thresholds1 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds2 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds3 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds4 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds5 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds5 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds6 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds7 > 0)
            seg_explicit_thresholds = sitk.Or(seg_explicit_thresholds, seg_explicit_thresholds8 > 0)

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

def load_nii_mask(file_path):
    """
    Load a binary mask from a NIfTI (.nii) file using nibabel.
    """
    nii_img = nib.load(file_path)
    binary_mask = nii_img.get_fdata()  # Get the data as a numpy array
    binary_mask = binary_mask > 0  # Convert to a binary mask (assuming non-zero values are part of the mask)

    return binary_mask, nii_img.affine, nii_img.header
def create_3d_model_from_mask(binary_mask, stl_filename):
    """
    Step 1: Create a 3D model (STL) from the binary mask using Marching Cubes
    """
    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0)

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Save the mesh as an STL file
    mesh.export(stl_filename)
    print(f"3D surface model saved as {stl_filename}")

    return mesh
def create_convex_hull_and_fix_normals(mesh, output_stl_filename):
    """
    Step 2: Compute the convex hull and ensure that all normals are pointing outwards.
    """
    # Create the convex hull of the mesh
    hull_mesh = mesh.convex_hull

    # Ensure normals are outward
    hull_mesh.fix_normals()

    # Save the convex hull mesh as an STL file
    hull_mesh.export(output_stl_filename)
    print(f"Convex hull with fixed normals saved as {output_stl_filename}")

    return hull_mesh
def scale_mesh_fixed_center(stl_filename, output_filename, scale_factor):
    """
    Scales a 3D mesh from an STL file, keeping the center fixed, and saves the scaled mesh as a new STL file.

    Parameters:
    - stl_filename: str, path to the input STL file.
    - output_filename: str, path to save the scaled STL file.
    - scale_factor: float or list of 3 floats (x, y, z scaling factors).
                    If a single float is provided, it scales uniformly in all directions.
    """
    # Load the STL file
    mesh = trimesh.load_mesh(stl_filename)

    # Find the centroid of the mesh
    centroid = mesh.centroid

    # Translate the mesh to the origin (centroid becomes [0, 0, 0])
    mesh.vertices -= centroid

    # Apply scaling to the mesh
    if isinstance(scale_factor, (float, int)):  # Uniform scaling
        mesh.apply_scale(scale_factor)
    elif isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 3:  # Non-uniform scaling (x, y, z)
        mesh.vertices *= scale_factor
    else:
        raise ValueError("scale_factor must be a float (uniform scaling) or a list of 3 floats (x, y, z scaling).")

    # Translate the mesh back to its original position
    mesh.vertices += centroid

    # Save the scaled mesh as an STL file
    mesh.export(output_filename)
    print(f"Scaled mesh (with fixed center) saved as {output_filename}")
def stl_to_binary_mask(stl_filename,inputniftifilename, output_nifti_filename, volume_shape):
    """
    Converts an STL file to a binary mask and saves it as a NIfTI file (no affine required).

    Parameters:
    - stl_filename: str, path to the input STL file.
    - output_nifti_filename: str, path to save the binary mask as a NIfTI file.
    - volume_shape: tuple, shape of the output binary mask (e.g., (x, y, z)).
    """
    # Load the STL file
    inputniftifilename_nib=nib.load(inputniftifilename)
    mesh = trimesh.load_mesh(stl_filename)

    # Create an empty binary mask with the given volume shape
    binary_mask = np.zeros(volume_shape, dtype=bool)

    # Create a grid of voxel coordinates
    grid = np.indices(volume_shape).reshape(3, -1).T

    # Check which of the voxel grid coordinates are inside the mesh (assuming voxel size is 1 unit)
    inside = mesh.contains(grid)

    # Update the binary mask with the points that are inside the mesh
    binary_mask[tuple(grid[inside].T)] = 1

    # Save the binary mask as a NIfTI file
    nifti_img = nib.Nifti1Image(binary_mask.astype(np.uint8), affine=inputniftifilename_nib.affine,header=inputniftifilename_nib.header) #np.eye(4))  # Identity affine (default)
    nib.save(nifti_img, output_nifti_filename)

    print(f"Binary mask saved as {output_nifti_filename}")
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


def binarymask_to_convexhull_mask(input_nii_path,output_nii_path):
    # Step 1: Load the input binary mask from a NIfTI file
    # input_nii_path = 'workinginput/ventricle.nii'
    binary_mask, affine, header = load_nii_mask(input_nii_path)

    # Step 2: Create 3D model from binary mask and save the STL file
    surface_mesh_stl = 'surface_model.stl'
    surface_mesh = create_3d_model_from_mask(binary_mask, surface_mesh_stl)

    # Step 3: Create convex hull from the 3D model and fix the normals
    convex_hull_stl = 'convex_hull_fixed_normals.stl'
    convex_hull_mesh = create_convex_hull_and_fix_normals(surface_mesh, convex_hull_stl)
    scale_mesh_fixed_center(convex_hull_stl, 'convex_hull_stl_1.stl', 1.05)
    # Step 4: Create binary mask from convex hull aligned with the original mask
    # convex_hull_binary_mask = create_binary_mask_from_hull(convex_hull_mesh, binary_mask.shape, affine)

    # # Step 5: Save the binary mask as a new NIfTI file
    # output_nii_path = 'convex_hull_binary_mask.nii'
    # save_nii_mask(convex_hull_binary_mask, affine, header, output_nii_path)
    #
    # print(f"Convex hull binary mask saved to: {output_nii_path}")

    convex_hull_mesh = trimesh.load('convex_hull_stl_1.stl') #'convex_hull_fixed_normals.stl')
    stl_to_binary_mask('convex_hull_stl_1.stl',input_nii_path, output_nii_path, binary_mask.shape)
ventricle_mask=resizeinto_512by512_and_flip(nib.load(sys.argv[1]).get_fdata())
centroids, ventricle_obb_mask = process_binary_mask_with_obb(ventricle_mask,n_subdivisions=9)
csf_mask_nib=nib.load(sys.argv[2])
array_img = nib.Nifti1Image(ventricle_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
nib.save(array_img, os.path.join(sys.argv[3],'ventricle.nii'))
input_nii_path=os.path.join(sys.argv[3],'ventricle.nii') #'/media/atul/WDJan20222/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/workinginput/ventricle.nii'
output_nii_path=os.path.join(sys.argv[3],'ventricle_convexhull_mask.nii') #'../TESTING_CSF_SEPERATION/workinginput/ventricle_convexhull_mask.nii'
filename_gray=sys.argv[4]
filename_mask=sys.argv[2]
filename_bet=sys.argv[5]
filename_vent_conv_hull=output_nii_path #os.path.join(sys.argv[3],'ventricle_convexhull_mask.nii')
binarymask_to_convexhull_mask(input_nii_path,output_nii_path)
closest_voxels=find_closest_non_zero_voxel(nib.load(filename_mask).get_fdata(), centroids)
divideintozones_with_vent_convexhull(filename_gray,filename_mask,filename_bet,filename_vent_conv_hull,closest_voxels)
