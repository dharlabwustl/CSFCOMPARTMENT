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
from  Segmentation_Ventricle_Sulcus_CSF_1_Dec15_2019 import * 
colors = vtk.vtkNamedColors()
renderer = vtk.vtkRenderer()
###############
## Apply BET:
ANAYLYSIS_TYPE="CSF_COMPARTMENT_VEN_SUL_AB"
project_folder="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment"


#MATFILE_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/CTP_MASK/"
#CTP_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/CTP/"
NECT_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT"
NECT_BET_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/BET"
RESULT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/RESULTS"
SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/RESULTS/IMAGES"

grayscalefilextension="_levelset.nii.gz"

#allgrayscalefolder = os.listdir(NECT_directory_name_parent)
#file1=latex_start(latexfilename)
#latex_start(latexfilename)
#latex_begin_document(latexfilename)
dict_for_csv=[]
dict_for_csv_perslice=[]
count =0
#features_template_upper = image_features(['/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/SOFTWARES/upper_template_new.jpg'])  #template_upper
#features_template_lower = image_features(['/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/SOFTWARES/template_lower.jpg']) 

grayscale_suffix="_levelset"
masksuffix="_final_seg" #
betsuffix="_levelset_bet"
dirname=sys.argv[1]
if "Helsinki2000_1" != dirname:
    partcount=1
    partcount_forname=1
    latexpart="_"+str(partcount)
    latexfilename=os.path.join(os.path.dirname(SLICE_OUTPUT_DIRECTORY),ANAYLYSIS_TYPE+dirname+latexpart+".tex")
    #file1=latex_start(latexfilename)
    latex_start_1(latexfilename)
    #latex_begin_document(latexfilename)
    
    #for dirname in os.listdir(NECT_directory_name_parent): 
    #    if count <1 and  os.path.isdir(os.path.join(NECT_directory_name_parent,dirname)): # and dirname=="WUSTL_664"        
    if count <1 and  os.path.isdir(os.path.join(NECT_directory_name_parent,dirname)):
    #        
    #        command="mkdir -p  " + os.path.join(RESULT_DIRECTORY,dirname)
    #        subprocess.call(command,shell=True)        
        NECT_directory_name=os.path.join(NECT_directory_name_parent,dirname) 
        NECT_filenames=glob.glob(NECT_directory_name+"/*" + grayscalefilextension) 
        for   NECT_filename in NECT_filenames:
            if count <1 :
                if partcount %10==0:
                    latex_end(latexfilename)
                    latex_file_build(latexfilename)
                    partcount_forname=partcount_forname+1
                    latexpart="_"+str(partcount_forname)
                    latexfilename=os.path.join(os.path.dirname(SLICE_OUTPUT_DIRECTORY),ANAYLYSIS_TYPE+dirname+latexpart+".tex")
                    #file1=latex_start(latexfilename)
                    latex_start_1(latexfilename)
                    #latex_begin_document(latexfilename)
                    partcount=partcount+1
                    
                
                nect_file_basename=os.path.basename(NECT_filename)
                nect_file_basename_forimagename=nect_file_basename.split('.')[0]
                NECT_HET_filename=os.path.join(NECT_BET_directory_name_parent,dirname,nect_file_basename[:-16]+"_levelset_bet.nii.gz" )
                CSF_Mask_filename=os.path.join(NECT_directory_name_parent,dirname,nect_file_basename[:-16]+"_final_seg.nii.gz" )
        #                CSF_Mask_filename_data=nib.load(CSF_Mask_filename)
        #                CSF_Mask_filename_data_np=np.uint8(CSF_Mask_filename_data.get_fdata()*255) #exposure.rescale_intensity( CSF_Mask_filename_data.get_fdata() , in_range=(0.5, 1200)) *255
                RAW_DATA_FOLDER=NECT_directory_name
        #                call_for_all_files(RAW_DATA_FOLDER,grayscale_suffix,masksuffix,betsuffix)
                each_unique_names_file_pattern=dirname
                filename_gray = NECT_filename
                filename_mask = CSF_Mask_filename
                filename_bet = NECT_HET_filename
                print("filename_gray")
                print(filename_gray)
                sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_v1(latexfilename,SLICE_OUTPUT_DIRECTORY,filename_gray,filename_mask,filename_bet)
                
                latex_start_table2c(latexfilename)
                latex_inserttext_table2c(latexfilename,text1='SulciVol:', text2=str(sulci_vol))
                latex_insert_line(latexfilename,text='\\\\')
                latex_inserttext_table2c(latexfilename,text1='VentricleVol:', text2=str(ventricle_vol))
                latex_insert_line(latexfilename,text="\\\\")
                latex_inserttext_table2c(latexfilename,text1='SulciVolAboveVent:', text2=str(sulci_vol_above_vent))
                latex_insert_line(latexfilename,text="\\\\")
                latex_inserttext_table2c(latexfilename,text1='SulciVolBelowVent:', text2=str(sulci_vol_below_vent))
                latex_insert_line(latexfilename,text="\\\\")
                latex_inserttext_table2c(latexfilename,text1='SulciVolAtVent:', text2=str(sulci_vol_at_vent))                        
                latex_end_table2c(latexfilename)
                partcount=partcount+1
                this_dict={"Subject": nect_file_basename[:-7],"Ventricles_Vol":ventricle_vol,"sulci_vol_above_vent": sulci_vol_above_vent,"sulci_vol_at_vent":sulci_vol_at_vent}
                dict_for_csv.append(this_dict)
    
                #print(dict_for_csv)
        csv_filename=os.path.join(RESULT_DIRECTORY,ANAYLYSIS_TYPE)
        csvfile_with_vol=csv_filename+dirname+'.csv'
        csv_columns=['Subject','Ventricles_Vol','sulci_vol_above_vent','sulci_vol_at_vent']
        write_csv(csvfile_with_vol,csv_columns,dict_for_csv)
        latex_end(latexfilename)
        latex_file_build(latexfilename)
