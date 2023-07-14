# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:26:44 2019

@author: atul
"""

import ants,os,sys,subprocess
import SimpleITK as sitk
import math
import numpy as np
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts')
from utilities_master import * 
from scipy.io import loadmat
##############
import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from vtk import * 
import sys
import six
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import glob
from scipy import ndimage as ndi
import np_obb
from sympy import *
from skimage import exposure
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.util import invert
from image_features import image_features 
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/imagedisplay')
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/csfbased')
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts')
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/HEMORRHAGE/SOFTWARE/pyScripts')
from utilities_master import *
from vtk_python_functions import * 
from csf_bounding_box import *
from matplotlib.patches import Circle
import subprocess
import cv2
from savenumpymatrix import *
# identify the slice which contains the
import vtk
import itk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import svd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import sys, subprocess
from volume_rendering_vtk import * 
from vtk import * 
from vtk import (
    vtkJPEGReader, vtkImageCanvasSource2D, vtkImageActor, vtkPolyDataMapper,
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkSuperquadricSource,
    vtkActor, VTK_MAJOR_VERSION
)
from myfunctions import * 
import re
colors = vtk.vtkNamedColors()
renderer = vtk.vtkRenderer()
###############
## Apply BET:
project_folder="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment"


#MATFILE_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/CTP_MASK/"
#CTP_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/CTP/"
NECT_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT"
NECT_BET_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/BET"
RESULT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/RESULTS"
SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/RESULTS/IMAGES"
latexfilename=os.path.join(os.path.dirname(SLICE_OUTPUT_DIRECTORY),os.path.basename(project_folder)+".tex")
grayscalefilextension="_levelset.nii.gz"

allgrayscalefolder = os.listdir(NECT_directory_name_parent)
file1=latex_start(latexfilename)
latex_start(latexfilename)
latex_begin_document(latexfilename)
dict_for_csv=[]
dict_for_csv_perslice=[]
count =0
features_template_upper = image_features(['/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/SOFTWARES/upper_template_new.jpg'])  #template_upper
features_template_lower = image_features(['/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/SOFTWARES/template_lower.jpg']) 
#for dirname in os.listdir(NECT_directory_name_parent): 
#    if count <1 and  os.path.isdir(os.path.join(NECT_directory_name_parent,dirname)): # and dirname=="WUSTL_664"    
#        command="mkdir -p  " + os.path.join(RESULT_DIRECTORY,dirname)
#        subprocess.call(command,shell=True)
#        
#        NECT_directory_name=os.path.join(NECT_directory_name_parent,dirname)  
#        NECT_filenames=glob.glob(NECT_directory_name+"/*" + grayscalefilextension) 
#        for   NECT_filename in NECT_filenames:
#            if count <1 :
#                nect_file_basename=os.path.basename(NECT_filename)
#                nect_file_basename_forimagename=nect_file_basename.split('.')[0]
#                NECT_HET_filename=os.path.join(NECT_BET_directory_name_parent,dirname,nect_file_basename[:-16]+"_levelset_bet.nii.gz" )
#                CSF_Mask_filename=os.path.join(NECT_directory_name_parent,dirname,nect_file_basename[:-16]+"_final_seg.nii.gz" )
#                CSF_Mask_filename_data=nib.load(CSF_Mask_filename)
#                CSF_Mask_filename_data_np=np.uint8(CSF_Mask_filename_data.get_fdata()*255) #exposure.rescale_intensity( CSF_Mask_filename_data.get_fdata() , in_range=(0.5, 1200)) *255
#                
#                print(NECT_HET_filename)
#    
#                file_gray_bet=NECT_HET_filename #NECT_HET_filename_gray
#                filter_type='_Gabor'
#                RESULT_DIR= os.path.join(os.path.join(RESULT_DIRECTORY,dirname),'midlines_gabor')
#                gray_image_filepath=NECT_filename
#                gray_bet_image_filepath=file_gray_bet
#                slicefirst=1
#                sliceend=15
#                DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
#                DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
#            #    img_nii=nib.load(gray_image_filepath) #filesindir[file])
#                print('gray_image_filepath')
#                print(gray_image_filepath)
#                file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
#                img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
#                img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
#                
#                #        img = cv2.imread(filesindir[file]) # img_nii.get_fdata() #cv2.imread('0.jpg')
#                img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
#                img=img*normalizeimage0to1(img_gray_bet.get_fdata())
#        #        count= count+1
#                min_diff=9999999
#                matched_image_id_upper=0
#                diff_array=[]
#    #        cv2.imwrite('template_lower.jpg',img[:,:,24]*255)
#    #        
#    # 
#    #        
#    ##        cv2.imshow(" ", xxx)
#    ##        cv2.waitKey()
#    ##       ## upper slice 
#                for x in range(img.shape[2]):
#                    if x > img.shape[2]*0.5:
#                        cv2.imwrite('test.jpg',img[:,:,x]*255)
#                        features_test = image_features(['test.jpg'])
#                        feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
#                        diff_array.append(feature_diff)
#                        if min_diff> feature_diff:
#                            min_diff=feature_diff
#                            matched_image_id_upper=x
#        #                    cv2.imwrite(dirname+'upper.jpg',img[:,:,x]*255)
#        #        ## lower slice  
#                min_diff=9999999
#                matched_image_id_lower=0
#                diff_array_lower=[]          
#                for x in range(img.shape[2]):
#                    if x< img.shape[2]*0.5:
#                        cv2.imwrite('test.jpg',img[:,:,x]*255)
#                        features_test = image_features(['test.jpg'])
#                        feature_diff=np.sqrt(np.sum((features_template_lower-features_test)*(features_template_lower-features_test))/features_template_lower.shape[1])
#                        diff_array_lower.append(feature_diff)
#                        if min_diff> feature_diff:
#                            min_diff=feature_diff
#                            matched_image_id_lower=x
#        #                    cv2.imwrite(dirname+'lower.jpg',img[:,:,x]*255)
#                file_gray=gray_image_filepath
#                slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_v1(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_upper) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
##                show_slice_withaline(np.uint8(img[:,:,matched_image_id_upper]*255),np.array([pointA_upper,pointB_upper]))
#                slice_3_layer_lower, filtered_img_lower, binary_image_copy_lower,score_diff_from_1_lower,slope_of_lines_lower, pointA_lower, pointB_lower= find_falxline_v1(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_lower) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
#                pointA_lower,pointB_lower=lower_slice_midline_refinement_v1_1(np.uint8(img_gray_bet.get_fdata()[:,:,matched_image_id_lower]*255),np.uint8(img[:,:,matched_image_id_lower]*255),pointA_lower,pointB_lower,anglethreshold=10)
##                show_slice_withaline(np.uint8(img[:,:,matched_image_id_lower]*255),np.array([pointA_lower,pointB_lower]))
#                slice_3_layer= np.zeros([img[:,:,matched_image_id_lower].shape[0],img[:,:,matched_image_id_lower].shape[1],3])
#                slice_3_layer[:,:,0]=np.uint8(img[:,:,matched_image_id_lower]*255)
#                slice_3_layer[:,:,1]=np.uint8(img[:,:,matched_image_id_lower]*255)
#                slice_3_layer[:,:,2]=np.uint8(img[:,:,matched_image_id_lower]*255)
#        #        slice_3_layer=np.uint8(slice_3_layer)
#                binary_image_copy=cv2.line(np.uint8(img[:,:,matched_image_id_lower]*255), ( int(pointA_lower[0]),int(pointA_lower[1])),(int(pointB_lower[0]),int(pointB_lower[1])), (255,255,0), 2)
#                cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(matched_image_id_lower)+'.jpg'),binary_image_copy)
#                
#                pair1=slope_of_lines_upper[0] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
#                pair2=slope_of_lines_lower[0] #(pair2_1[1],pair2_1[4],pair2_1[5],pair2_1[0])
#                infarct_pixel_intensity=[]
#                noninfarct_pixel_intensity=[]
#                if np.sum(np.array(pair1[1]))>0 and np.sum(np.array(pair2[1]))>0  and np.sum(np.array(pair1[2]))>0 and np.sum(np.array(pair2[2]))>0 :
#                #    p_actor=draw_plane_1(np.array([pair1[1][1], pair1[1][0], pair1[3]]),np.array([pair2[1][1], pair2[1][0], pair2[3]]),np.array([pair1[2][1], pair1[2][0], pair1[3]]))
#                #    renderer.AddActor(p_actor)
#                ##    slice_3_layer, filtered_img, binary_image_copy, pair1, pair2 =find_four_points_for_plane_nosave(file_gray,RESULT_DIR,filter_type,slicefirst=0,sliceend=15)
#                ##    seg_explicit_thresholds_numpy=sitk.GetArrayFromImage(seg_explicit_thresholds)
#                ##    # set  image in a plane and get the planesource
#                    print('find_four_points_for_plane_nosave')
#                    print('len(pair1)')
#                    print(len(pair1[1]))
#                    print(len(pair1[2]))
#                    filename_gray_bet_data_np=nib.load(file_gray_bet).get_fdata() #file_gray).get_fdata()
#                    filename_gray_data_np=nib.load(file_gray).get_fdata()
#    
#        #            filename_gray_data_np[filename_gray_bet_data_np<255]=0
#    #                filename_mask_data_np=nib.load(os.path.join(RESULT_DIRECTORY,'moving_mask.nii.gz')).get_fdata()
#    #                filename_gray_data_np_copy=np.copy(filename_gray_data_np)
#                #    print(pair1[3])
#                #    draw_imageplanes_vtk(normalizeimage0to1(filename_gray_data_np)*255,renderer,pair1[3],pair1[3]+1)
#                    planeSource = vtk.vtkPlaneSource()
#                
#                    planeSource.SetPoint1(pair1[1][1], pair1[1][0], pair1[3])
#                    planeSource.SetPoint2(pair1[2][1], pair1[2][0], pair1[3])
#                #    planeSource.SetPoint2(pair2[1][1], pair2[1][0], pair2[3])
#                    planeSource.SetOrigin((pair2[1][1]+pair2[2][1])/2, (pair2[1][0]+pair2[2][0])/2, pair2[3])
#                    planeSource.Update()
#                    normal_p=planeSource.GetNormal()
#                    center_p=planeSource.GetCenter() #[(pair1[1][1]+pair1[2][1])/2, (pair1[1][0]+pair1[2][0])/2, pair1[3]] #planeSource.GetCenter(
#                    plane_size=512
#                
#                
#                #    draw_perpendicular_plane_oneslice2(center_p,normal_p,"X",renderer,plane_size)
#                    transformPD0, actor0= draw_plane_2((pair1[1][1], pair1[1][0], pair1[3]),(pair2[1][1], pair2[1][0], pair2[3]),(pair1[2][1], pair1[2][0], pair1[3]),renderer,scale_factor=5, N="Z")
#                    filename_gray_data_np=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
#                    filename_gray_data_np_1=np.uint8(filename_gray_data_np*255)
#                    numpy_image=normalizeimage0to1(filename_gray_data_np)*255
#    #                numpy_image_mask=normalizeimage0to1(filename_mask_data_np)*255 
#                    ## left and right number of pixels:
#                    left_pixels_num=0
#                    right_pixels_num=0
#                    this_slice_left_volume=0
#                    this_slice_right_volume=0
#                    for img_idx in range(numpy_image.shape[2]):
#                #    img_idx=0
#                        print(np.sum(numpy_image[:,:,img_idx]))
#                        if img_idx>0 and img_idx < numpy_image.shape[2]:
#                            print("img_idx working")
#                            transformPD1,actor1= image_plane_vtk_getplane(numpy_image[:,:,img_idx],img_idx, rgbFrame=None)
#                            renderer.AddActor(actor1)
#                            act,intersection_line= cutter_polydata_v1(center_p,normal_p,transformPD1)
#                            print(intersection_line.GetOutput().GetNumberOfPoints())
#                            points=np.array([[intersection_line.GetOutput().GetPoint(0)[1],intersection_line.GetOutput().GetPoint(0)[0]],[intersection_line.GetOutput().GetPoint(1)[1],intersection_line.GetOutput().GetPoint(1)[0]]])
#                            points_copy=np.copy(points)
#                            ##  calculate the left and right pixels:
#                            img_with_line=CSF_Mask_filename_data_np[:,:,img_idx]
#                            img_with_line_nonzero_id = np.transpose(np.nonzero(img_with_line)) 
#                            thisimage=filename_gray_data_np_1[:,:,img_idx]
#                            current_left_num=0
#                            current_right_num=0
#                            slice_3_layer= np.zeros([img_with_line.shape[0],img_with_line.shape[1],3])
#                            slice_3_layer[:,:,0]= thisimage #imgray1
#                            slice_3_layer[:,:,1]= thisimage #imgray1
#                            slice_3_layer[:,:,2]= thisimage# imgray1
#                            for non_zero_pixel in img_with_line_nonzero_id:
#                                xx=whichsideofline((points_copy[0][1],points_copy[0][0]) ,(points_copy[1][1],points_copy[1][0]),non_zero_pixel)
#                                if xx>0: ## LEFT
#                                    slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                                    slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=255
#                                    slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=0
#                                    current_left_num = current_left_num + 1
#                                if xx<0: ## RIGHT
#                                    slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                                    slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=0
#                                    slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=255
#                                    current_right_num = current_right_num + 1
#                                    
#                            
#                            lineThickness = 2
#    #                        left_pixels_num=left_pixels_num+current_left_num
#    #         #np.int8(CSF_Mask_filename_data_np[:,:,img_idx])  #np.int8(numpy_image[:,:,img_idx])
#    #                                thisimage=filename_gray_data_np_1[:,:,img_idx]
#    ##                                thisimage[img_with_line>0]=255
#    
#    #                        out_ind = geek.transpose(geek.nonzero(arr)) 
#                            this_slice_left_volume = current_left_num*np.prod(np.array(nib.load(file_gray).header["pixdim"][1:4])) 
#                            this_slice_right_volume = current_right_num*np.prod(np.array(nib.load(file_gray).header["pixdim"][1:4])) 
#                            
#                            img_with_line1=cv2.line(slice_3_layer, (int(intersection_line.GetOutput().GetPoint(0)[1]),int(intersection_line.GetOutput().GetPoint(0)[0])), (int(intersection_line.GetOutput().GetPoint(1)[1]),int(intersection_line.GetOutput().GetPoint(1)[0])), (0,255,0), lineThickness)
#                            cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,nect_file_basename_forimagename+"_" +str(img_idx))+".png",img_with_line1)
#                            
#                            
#                            ## get the mask image:
#                            this_slice_left_volume=this_slice_left_volume/1000
#                            this_slice_right_volume=this_slice_right_volume/1000
#                            latex_start_table1c(latexfilename)
#                            latex_insertimage_table1c(latexfilename,image1=os.path.join(SLICE_OUTPUT_DIRECTORY,nect_file_basename_forimagename+"_" +str(img_idx))+".png",caption= dirname+"_" +str(img_idx),imagescale=0.3)
#                            latex_end_table2c(latexfilename)
#                            latex_insert_line(latexfilename,"\\texttt{\\detokenize{" + nect_file_basename_forimagename+"_" + str(img_idx) + "}}")
#                            latex_insert_line(latexfilename,"\\texttt{ LEFT VOLUME     :     " + str(this_slice_left_volume) + "}")
#                            latex_insert_line(latexfilename,"\\texttt{ RIGHT VOLUME     :     " + str(this_slice_right_volume) + "}")
#                            this_dict1={"FileName": nect_file_basename_forimagename +"_" + str(img_idx) ,"LEFT_VOLUME":this_slice_left_volume, "RIGHT_VOLUME":this_slice_right_volume} #,"Ventricles_Vol":ventricle_vol,"Sulci_VolL":leftcountsul,"Sulci_VolR":rightcountsul,"Ventricles_VolL":leftcountven,"Ventricles_VolR":rightcountven,"sulci_vol_above_vent": sulci_vol_above_vent,"sulci_vol_below_vent" :sulci_vol_below_vent,"sulci_vol_at_vent":sulci_vol_at_vent}
#                            dict_for_csv_perslice.append(this_dict1)
#                        left_pixels_num=left_pixels_num+this_slice_left_volume
#                        right_pixels_num=right_pixels_num+this_slice_right_volume 
#                        latex_insert_line(latexfilename,"\\texttt{\\detokenize{" + nect_file_basename_forimagename + "}}")
#                        latex_insert_line(latexfilename,"\\texttt{ LEFT VOLUME     :     " + str(left_pixels_num) + "}")
#                        latex_insert_line(latexfilename,"\\texttt{ RIGHT VOLUME     :     " + str(right_pixels_num) + "}")
#                        this_dict={"FileName": nect_file_basename_forimagename ,"LEFT_VOLUME":left_pixels_num, "RIGHT_VOLUME":right_pixels_num} #,"Ventricles_Vol":ventricle_vol,"Sulci_VolL":leftcountsul,"Sulci_VolR":rightcountsul,"Ventricles_VolL":leftcountven,"Ventricles_VolR":rightcountven,"sulci_vol_above_vent": sulci_vol_above_vent,"sulci_vol_below_vent" :sulci_vol_below_vent,"sulci_vol_at_vent":sulci_vol_at_vent}
#                        dict_for_csv.append(this_dict)
##                        count = count +1     

latex_end(latexfilename)
latex_file_build(latexfilename)
#csv_filename=os.path.join(os.path.dirname(SLICE_OUTPUT_DIRECTORY),os.path.basename(project_folder))
#csvfile_with_vol=csv_filename+'.csv'
#csv_columns=['FileName','LEFT_VOLUME','RIGHT_VOLUME'] #,'Ventricles_Vol','Sulci_VolL','Sulci_VolR','Ventricles_VolL','Ventricles_VolR','sulci_vol_above_vent','sulci_vol_below_vent','sulci_vol_at_vent']
#write_csv(csvfile_with_vol,csv_columns,dict_for_csv)

#csvfile_with_vol=csv_filename+'_EACHSLICE.csv'
#csv_columns=['FileName','LEFT_VOLUME','RIGHT_VOLUME'] #,'Ventricles_Vol','Sulci_VolL','Sulci_VolR','Ventricles_VolL','Ventricles_VolR','sulci_vol_above_vent','sulci_vol_below_vent','sulci_vol_at_vent']
#write_csv(csvfile_with_vol,csv_columns,dict_for_csv_perslice)

#
##    #                    show_slice_withaline(np.copy(numpy_image[:,:,img_idx]),points)
##    #                    show_slice_withaline(np.copy(numpy_image_mask[:,:,img_idx]),points)
##                        v1=np.array([512,0]) ## point from the image
##                        v2_1=np.array([intersection_line.GetOutput().GetPoint(0)[1],intersection_line.GetOutput().GetPoint(0)[0]]) ## point 1 from the midline
##                        v2_2=np.array([intersection_line.GetOutput().GetPoint(1)[1],intersection_line.GetOutput().GetPoint(1)[0]]) ## point 1 from the midline
##                        v2=v2_2-v2_1
##                        print(v2)
##                        print(angle_bet_two_vector(v1,v2))
##                        angle=  angle_bet_two_vector(v1,v2) 
##                        angleRad=angle_bet_two_vectorRad(v1,v2)
##                        ## translation:
##                        mid_point_line=np.mean(points,axis=0)
##                        # delta translation:
##                        image_midpoint=np.array([255,255])
##                        translation_delta=image_midpoint-mid_point_line
##                        M = np.float32([[1,0,translation_delta[0]],[0,1,translation_delta[1]]])
##                        I_t_gray = cv2.warpAffine(np.copy(numpy_image[:,:,img_idx]),M,(512,512), flags= cv2.INTER_NEAREST)
##                        I_t_mask = cv2.warpAffine(np.copy(numpy_image_mask[:,:,img_idx]),M,(512,512) , flags= cv2.INTER_NEAREST)
##    #                    show_slice_withaline(np.copy(numpy_image[:,:,img_idx]),points)
##                        #show_slice_withaline(I_t_gray,points)
##                        translate_points= points+translation_delta
##    #                    show_slice_withaline(I_t_mask,translate_points)
##                        points=translate_points
##                        ## translation matrix
##                        p1x,p1y= rotate_around_point_highperf(np.array([points[0][0],points[0][1]]), angleRad, origin=(255,255))
##                        p2x,p2y= rotate_around_point_highperf(np.array([points[1][0],points[1][1]]), angleRad, origin=(255,255))
##                        points1=np.array([[p1x,p1y],[p2x,p2y]])
##    ##                    print('np.copy(numpy_image_mask[:,:,img_idx])')
##    ##                    print(np.max(np.copy(numpy_image_mask[:,:,img_idx])))
##    #        #            I1=rotate_image(I_t,(255,255),angle)
##                        I_t_r_gray=rotate_image(I_t_gray,(255,255),angle)
##    #                    show_slice_withaline(I_t_r_gray,points1)
##                        I_t_r_mask=rotate_image(I_t_mask,(255,255),angle)
##    #                    show_slice_withaline(I_t_r_mask,points1)
##    #        #            #show_slice_withaline(I1,points1)
##    #        ###            cv2.imshow("", img_with_line)
##                        I_t_r_f_gray=cv2.flip(I_t_r_gray,0)
##                        I_t_r_f_mask=cv2.flip(I_t_r_mask,0)
##                        #show_slice_withaline(I_t_r_f_gray,points1)
##    #                    show_slice_withaline(I_t_r_f_mask,points1)
##    #        ###            show_slice_colored_withpoint(I2,points)
##                        I_t_r_f_rinv_gray=rotate_image(I_t_r_f_gray,(256,256),-angle)
##                        I_t_r_f_rinv_mask=rotate_image(I_t_r_f_mask,(256,256),-angle)
##                        p1x,p1y= rotate_around_point_highperf(np.array([points1[0][0],points1[0][1]]), -angleRad, origin=(255,255))
##                        p2x,p2y= rotate_around_point_highperf(np.array([points1[1][0],points1[1][1]]), -angleRad, origin=(255,255))
##                        points1=np.array([[p1x,p1y],[p2x,p2y]])
##                        #show_slice_withaline(I_t_r_f_rinv_gray,points1)
##    #                    show_slice_withaline(I_t_r_f_rinv_mask,points1)
##                        M = np.float32([[1,0,-translation_delta[0]],[0,1,-translation_delta[1]]])
##                        I_t_r_f_rinv_tinv_gray = cv2.warpAffine(I_t_r_f_rinv_gray,M,(512,512) , flags= cv2.INTER_NEAREST)
##                        I_t_r_f_rinv_tinv_mask = cv2.warpAffine(I_t_r_f_rinv_mask,M,(512,512), flags= cv2.INTER_NEAREST )
##                        points1=points1-translation_delta
##                        #show_slice_withaline(I_t_r_f_rinv_tinv_gray,points1)
##    #                    show_slice_withaline(I_t_r_f_rinv_tinv_mask,points1)
##                        I4=np.copy(numpy_image[:,:,img_idx])
##                        I4[I_t_r_f_rinv_tinv_mask>0]=255
##                        I4[np.copy(numpy_image_mask[:,:,img_idx])>0]=255
##                        I5=np.copy(filename_gray_data_np_copy[:,:,img_idx])
##                        if np.sum(I_t_r_f_rinv_tinv_mask)>0 :
##                            infarct_pixels=I5[numpy_image_mask[:,:,img_idx]>0]
##                            infarct_pixels_gt20 = infarct_pixels[infarct_pixels>=20]
##                            infarct_pixels_gt20_lt80 = infarct_pixels_gt20[infarct_pixels_gt20<=80]
##                            non_infarct_pixels=I5[I_t_r_f_rinv_tinv_mask>0]
##                            non_infarct_pixels_gt20 = non_infarct_pixels[non_infarct_pixels>=20]
##                            non_infarct_pixels_gt20_lt80 = non_infarct_pixels_gt20[non_infarct_pixels_gt20<=80]
##                            mean_slice_infarct_pixels_gt20_lt80= np.mean(infarct_pixels_gt20_lt80)
##                            if math.isnan(mean_slice_infarct_pixels_gt20_lt80):
##                                mean_slice_infarct_pixels_gt20_lt80=0
##                            mean_slice_non_infarct_pixels_gt20_lt80=np.mean(non_infarct_pixels_gt20_lt80)
##                            if math.isnan(mean_slice_non_infarct_pixels_gt20_lt80):
##                                mean_slice_non_infarct_pixels_gt20_lt80=0
##                            infarct_pixel_intensity.append(mean_slice_infarct_pixels_gt20_lt80) #I5[numpy_image_mask[:,:,img_idx]>0]))
##                            noninfarct_pixel_intensity.append(mean_slice_non_infarct_pixels_gt20_lt80)#I5[I_t_r_f_rinv_tinv_mask>0]))
##                            img_with_line1=cv2.line(I4, (int(points1[0][0]),int(points1[0][1])), (int(points1[1][0]),int(points1[1][1])), (0,255,0), lineThickness)
##    
##                            cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,dirname+"_" +str(img_idx))+".png",img_with_line1)
##    #                        latex_start_table1c(latexfilename)
##    #                        latex_insertimage_table1c(latexfilename,image1=os.path.join(SLICE_OUTPUT_DIRECTORY,dirname+"_" +str(img_idx))+".png",caption= dirname+"_" +str(img_idx),imagescale=0.3)
##    #                        latex_end_table2c(latexfilename)
##    #                        latex_insert_line(latexfilename,"\\texttt{\\detokenize{" +dirname+"_" +str(img_idx) + "}}")
##                            NWU_slice= (1- ((np.mean(infarct_pixels_gt20_lt80))/(np.mean(non_infarct_pixels_gt20_lt80)))) * 100
##                            this_dict={"Slice": dirname+"_" +str(img_idx),"NWU":NWU_slice} #,"Ventricles_Vol":ventricle_vol,"Sulci_VolL":leftcountsul,"Sulci_VolR":rightcountsul,"Ventricles_VolL":leftcountven,"Ventricles_VolR":rightcountven,"sulci_vol_above_vent": sulci_vol_above_vent,"sulci_vol_below_vent" :sulci_vol_below_vent,"sulci_vol_at_vent":sulci_vol_at_vent}
##                            dict_for_csv.append(this_dict)
##                            
##            #            print('img_idx')
##            #            print(img_idx)
##    #                    show_slice_withaline(I4,points1) #show_slice(I4)
##    #                    print('points diff')
##    #                    print(points1-points_copy)
##    #        
###            NWU= (1- (((np.mean(np.array(infarct_pixel_intensity)))/(np.mean(np.array(noninfarct_pixel_intensity))))) ) * 100
###            print(NWU)
###    #        count = count+1
###            this_dict={"Slice":dirname+ "_OVERALL","NWU":NWU} 
###            dict_for_csv.append(this_dict)
#latex_end(latexfilename)
#latex_file_build(latexfilename)
###csv_filename=latexfilename=os.path.join(os.path.dirname(SLICE_OUTPUT_DIRECTORY),os.path.basename(project_folder))
###csvfile_with_vol=csv_filename+'.csv'
###csv_columns=['Slice','NWU'] #,'Ventricles_Vol','Sulci_VolL','Sulci_VolR','Ventricles_VolL','Ventricles_VolR','sulci_vol_above_vent','sulci_vol_below_vent','sulci_vol_at_vent']
###write_csv(csvfile_with_vol,csv_columns,dict_for_csv)
###                    
###                    