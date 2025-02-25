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


def divideintozones_with_vent_obb(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z):
    try:
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

            initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])] ##img_T1 ##
            seg_explicit_thresholds =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)

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

def divideintozones_with_vent_obb_ven_hem_given(filename_gray,filename_mask,filename_bet,filename_vent_obb,zoneV_min_z,zoneV_max_z):
    try:
        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image
        #######################


        reader_filename_vent_nonlinmask = sitk.ImageFileReader()
        reader_filename_vent_nonlinmask.SetImageIO("NiftiImageIO")
        reader_filename_vent_nonlinmask.SetFileName('/workinginput/ventricle_contour.nii')
        ventricle_nonlin_mask = reader_filename_vent_nonlinmask.Execute()

        ##############################
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
        sah_hem_mask=glob.glob('/input/*_resaved_4DL_seg_total.nii.gz')[0]
        sah_hem_mask_itk = sitk.ImageFileReader()
        sah_hem_mask_itk.SetImageIO("NiftiImageIO")
        sah_hem_mask_itk.SetFileName(sah_hem_mask)
        sah_hem_mask_itk_object = sah_hem_mask_itk.Execute()
        sah_hem_mask_itk_object_np=sitk.GetArrayFromImage(sah_hem_mask_itk_object)
        ####################################################
        img_T1_temp_np=sitk.GetArrayFromImage(img_T1_1)
        img_T1_temp_np_alllabels=np.copy(img_T1_temp_np)
        img_T1_temp_np_alllabels=slicenum_at_end(img_T1_temp_np_alllabels)
        img_T1_temp_np[img_T1_temp_np>1]=0.0
        img_T1_temp_np_copy=np.copy(img_T1_temp_np)
        img_T1_temp_np_copy[sah_hem_mask_itk_object_np>0.001]=1
        img_T1_1_forsubtract_np=np.copy(img_T1_temp_np)
        img_T1_1_forsubtract_itk=sitk.GetImageFromArray(img_T1_1_forsubtract_np)
        img_T1_1_forsubtract_itk.CopyInformation(img_T1_1)
        img_T1_temp_np[ventricle_obb_np<1]=0.0
        ## ventricle_hemorrhage mask:
        img_T1_temp_np_copy_itk=sitk.GetImageFromArray(img_T1_temp_np_copy)
        img_T1_temp_np_copy_itk.CopyInformation(img_T1_1)
        img_T1_1=img_T1_temp_np_copy_itk
        vent_hem_mask_list=glob.glob('/input/*_resaved_4DL_seg_ventri.nii.gz')

        if len(vent_hem_mask_list)>0:
            print(f"I am at{vent_hem_mask_list}")
            vent_hem_mask=vent_hem_mask_list[0]
            vent_hem_mask_itk = sitk.ImageFileReader()
            vent_hem_mask_itk.SetImageIO("NiftiImageIO")
            vent_hem_mask_itk.SetFileName(vent_hem_mask)
            vent_hem_mask_itk_object = vent_hem_mask_itk.Execute()
            vent_hem_mask_itk_object_np=sitk.GetArrayFromImage(vent_hem_mask_itk_object)
            vent_hem_mask_itk_object_np[0:zoneV_min_z,:,:]=0
            # vent_hem_mask_itk_object_np[zoneV_max_z+1:vent_hem_mask_itk_object_np.shape[0],:,:]=0
            img_T1_temp_np[vent_hem_mask_itk_object_np>0.001]=1.0



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
            subprocess.call("echo " + "SUCCEEDED AT ::{}  >> error.txt".format(inspect.stack()[0][3]) ,shell=True )
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

            initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])] ##img_T1 ##
            seg_explicit_thresholds =sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
            img_T1_temp_np_1=sitk.GetArrayFromImage(seg_explicit_thresholds)
            # img_T1_temp_np_1[vent_hem_mask_itk_object_np>0.001]=1.0
            # img_T1_temp_np_1[vent_hem_mask_itk_object_np > 0] = vent_hem_mask_itk_object_np[vent_hem_mask_itk_object_np > 0]

            seg_explicit_thresholds=sitk.GetImageFromArray(img_T1_temp_np_1)

            zoneV_min_z2,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
            img_T1_1_arr=sitk.GetArrayFromImage(img_T1_1)
            # img_T1_1_arr[sah_hem_mask_itk_object_np > 0] = sah_hem_mask_itk_object_np[sah_hem_mask_itk_object_np > 0]


            subtracted_image=subtract_binary_1(sitk.GetArrayFromImage(img_T1_1),sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
            subtracted_image=subtract_binary_1(img_T1_1_arr,sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
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


#
import nibabel as nib
import numpy as np
import os

def load_nifti(file_path):
    """ Load a NIfTI file and return its data and affine. """
    nifti_img = nib.load(file_path)
    nifti_data = nifti_img.get_fdata()
    return nifti_data, nifti_img.affine, nifti_img.header

def save_nifti(data, affine, header, file_path):
    """ Save a NumPy array as a NIfTI file. """
    nifti_img = nib.Nifti1Image(data, affine, header)
    nib.save(nifti_img, file_path)

def divideintozones_with_vent_obb_noregiongrow(csf_mask_path, ventricle_mask_path, output_dir):
    # Load NIfTI masks
    csf_data, affine, header = load_nifti(csf_mask_path)
    ventricle_data, _, _ = load_nifti(ventricle_mask_path)

    # Get slices where ventricle mask is present
    slice_sums = np.sum(ventricle_data, axis=(0, 1))  # Sum along x, y axes
    ventricle_slices = np.where(slice_sums > 0)[0]

    if ventricle_slices.size == 0:
        print("No ventricle region found.")
        return

    lower_bound = ventricle_slices[0]
    upper_bound = ventricle_slices[-1]

    # Create masks for above and below ventricle
    above_ventricle = np.zeros_like(csf_data)
    below_ventricle = np.zeros_like(csf_data)

    above_ventricle[:, :, :lower_bound] = csf_data[:, :, :lower_bound]
    below_ventricle[:, :, upper_bound + 1:] = csf_data[:, :, upper_bound + 1:]

    # Save the above and below ventricle masks
    os.makedirs(output_dir, exist_ok=True)
    save_nifti(above_ventricle, affine, header, os.path.join(output_dir, "above_ventricle.nii.gz"))
    save_nifti(below_ventricle, affine, header, os.path.join(output_dir, "below_ventricle.nii.gz"))

    # Create ventricle inside and outside bounding box masks
    ventricle_inside = np.zeros_like(ventricle_data)
    ventricle_outside = np.zeros_like(csf_data)

    ventricle_inside[ventricle_data > 0] = csf_data[ventricle_data > 0]  # Inside ventricle box
    ventricle_outside[(csf_data > 0) & (ventricle_data == 0)] = csf_data[(csf_data > 0) & (ventricle_data == 0)]  # Outside box

    # Save ventricle zone masks
    save_nifti(ventricle_inside, affine, header, os.path.join(output_dir, "ventricle_inside.nii.gz"))
    save_nifti(ventricle_outside, affine, header, os.path.join(output_dir, "ventricle_outside.nii.gz"))

    print("Processing complete. Masks saved in:", output_dir)

# # Example usage
# csf_mask_path = "csf_mask.nii.gz"  # Replace with actual path
# ventricle_mask_path = "ventricle_mask.nii.gz"  # Replace with actual path
# output_dir = "output_masks"
#
# process_masks(csf_mask_path, ventricle_mask_path, output_dir)

