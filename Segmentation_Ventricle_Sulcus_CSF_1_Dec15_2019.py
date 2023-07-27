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
