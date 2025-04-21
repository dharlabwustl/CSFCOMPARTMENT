#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:04:42 2020

@author: atul
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:26:44 2019

@author: atul
"""
from skimage import exposure
import webcolors
import argparse,inspect
import glob,os,csv,sys
import nibabel as nib
import numpy as np
import cv2 , re,subprocess,time,math
sys.path.append("/software")
import traceback
#sys.path.append("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/DOCKERIZE/DOCKERIZEPYTHON/docker_fsl/docker/fsl/fsl-v5.0")
from fillmaster_session_list import *
from download_with_session_ID import *
from utilities_simple_trimmed import *

from github import Github
#############################################################
from dateutil.parser import parse
import pandas as pd
# g = Github()
# repo = g.get_repo("dharlabwustl/EDEMA_MARKERS")
# contents = repo.get_contents("module_NWU_CSFCompartment_Calculations.py")
# dt = parse(contents.last_modified)

Version_Date="_VersionDate-" +'08102023' # dt.strftime("%m%d%Y")

#############################################################


now=time.localtime()

def insert_one_col_with_colname_colidx(csvfilename,csvfilename_output,colname,colvalue):
    returnvalue=0
    try:
        df=pd.read_csv(csvfilename)
        df[colname]=colvalue
        df.to_csv(csvfilename_output,index=False)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'insert_one_col_with_colname_colidx')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'insert_one_col_with_colname_colidx')
        subprocess.call(command,shell=True)

    return returnvalue
def call_begin_csvfile_with_session_name(args):
    csvfilename=args.stuff[1]
    session_id=args.stuff[2]
    scan_name=args.stuff[3]
    begin_csvfile_with_session_name(csvfilename,str(session_id),str(scan_name))
def begin_csvfile_with_session_name(csvfilename,session_id,scan_name):
    try:
        session_name=get_session_label(session_id)
        command="echo success at :: {}::{}::{}:{}>> /software/error.txt".format(inspect.stack()[0][3],session_name,csvfilename,scan_name)
        subprocess.call(command,shell=True)
        scanname_df=pd.DataFrame([session_id,session_name,scan_name]).T
        scanname_df.columns=['SESSION_ID','snipr_session','scan_name']
        scanname_df.to_csv(csvfilename,index=False)
    except:
        command="echo failed at :: {}>> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)


def call_insert_one_col_with_colname_colidx(args):
    csvfilename=args.stuff[1]
    csvfilename_output=args.stuff[2]
    colname=args.stuff[3]
    colvalue=args.stuff[4]
    insert_one_col_with_colname_colidx(csvfilename,csvfilename_output,colname,colvalue)
def combine_csv_horizontally(list_df_files,combined_csv_filename,niftifilename):
    list_df=[]
    for x in list_df_files:
        if os.path.exists(x):
            list_df.append(pd.read_csv(x))

    list_df_combined = pd.concat(list_df, axis=1)
    # list_df_combined['FILENAME']=niftifilename
    # column_to_move = pd.DataFrame(list_df_combined.pop('FILENAME'))
    # list_df_combined.insert(1, 'FILENAME', column_to_move)
    list_df_combined.to_csv(combined_csv_filename,index=False)
def call_combine_csv_horizontally(args):
    returnvalue=0

    try:
        combined_csv_filename=args.stuff[2]
        niftifilename=args.stuff[1]
        list_df_files=args.stuff[3:]
        combine_csv_horizontally(list_df_files,combined_csv_filename,niftifilename)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_combine_csv_horizontally')
        subprocess.call(command,shell=True)
        returnvalue=1
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return returnvalue

# def color_to_BGRvalues():
def draw_midline_on_a_slice(grayscale_filename,method_name,npyfiledirectory,slice_3_layer,slice_number):
    returnvalue=0
    try:
        filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(grayscale_filename).split(".nii")[0])
        this_npyfile_v2=os.path.join(npyfiledirectory,filename_tosave+method_name+"_"+str(slice_number)+  "_V2.npy")
        this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")

        if os.path.exists(this_npyfile_v2):
            this_npyfile=this_npyfile_v2
        if os.path.exists(this_npyfile):
            command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
            subprocess.call(command,shell=True)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
        subprocess.call(command,shell=True)
        if os.path.exists(this_npyfile):
            command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'I exist')
            subprocess.call(command,shell=True)
            calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
            x_points2=calculated_midline_points.item().get('x_axis')
            y_points2=calculated_midline_points.item().get('y_axis')
            x_points2=x_points2[:,0]
            y_points2=y_points2[:,0]
            slice_3_layer=cv2.line(slice_3_layer, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,255), 2)
            command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'draw_midline_on_a_slice')
            subprocess.call(command,shell=True)
            return slice_3_layer
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    print(returnvalue)
    return slice_3_layer


def masks_on_grayscale_colored_v1(args):
    grayscale_filename=args.stuff[1]
    contrast_limits=(int(args.stuff[2].split('_')[0]),int(args.stuff[2].split('_')[1]))
    mask_color_list=args.stuff[5].split('_')
    outputfile_dir=args.stuff[3]
    outputfile_suffix=args.stuff[4]
    npyfiledirectory=args.stuff[6]
    mask_filename_list=args.stuff[7:]
    returnvalue=0
    command="echo successful at :: {}::{}::{}::{}::{}::{}::{}::{} >> /software/error.txt".format(inspect.stack()[0][3],grayscale_filename,contrast_limits,mask_color_list,outputfile_dir,outputfile_suffix,npyfiledirectory,mask_filename_list)
    subprocess.call(command,shell=True)

    try:
        grayscale_filename_np=nib.load(grayscale_filename).get_fdata()
        grayscale_filename_np=exposure.rescale_intensity( grayscale_filename_np , in_range=(contrast_limits[0], contrast_limits[1]))*255
        slice_3_layer= np.zeros([grayscale_filename_np.shape[0],grayscale_filename_np.shape[1],3])
        method_name="REGIS"
        # npyfiledirectory="/input"

        for i in range(grayscale_filename_np.shape[2]):
            slice_3_layer[:,:,0]= grayscale_filename_np[:,:,i] #imgray1
            slice_3_layer[:,:,1]= grayscale_filename_np[:,:,i] #imgray1
            slice_3_layer[:,:,2]= grayscale_filename_np[:,:,i]# imgray1

            for mask_filename_list_id in range(len(mask_filename_list)):
                mask_filename_np=nib.load(mask_filename_list[mask_filename_list_id]).get_fdata()
                slice_3_layer[:,:,0][mask_filename_np[:,:,i]>0]=webcolors.name_to_rgb(mask_color_list[mask_filename_list_id])[2]
                slice_3_layer[:,:,1][mask_filename_np[:,:,i]>0]=webcolors.name_to_rgb(mask_color_list[mask_filename_list_id])[1]
                slice_3_layer[:,:,2][mask_filename_np[:,:,i]>0]=webcolors.name_to_rgb(mask_color_list[mask_filename_list_id])[0]
            slice_number="{0:0=3d}".format(i)
            # # filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(grayscale_filename).split(".nii")[0])
            # # this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+"_"+str(slice_number)+  "_V2.npy")
            # if os.path.exists(npyfiledirectory):
            #     try:
            slice_3_layer=draw_midline_on_a_slice(grayscale_filename,method_name,npyfiledirectory,slice_3_layer,slice_number)
                # except:
                #     pass
            # command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
            # subprocess.call(command,shell=True)
            # if os.path.exists(this_npyfile):
            #     calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
            #     x_points2=calculated_midline_points.item().get('x_axis')
            #     y_points2=calculated_midline_points.item().get('y_axis')
            #     x_points2=x_points2[:,0]
            #     y_points2=y_points2[:,0]
            #     slice_3_layer=cv2.line(slice_3_layer, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,0), 2)
            #     command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
            #     subprocess.call(command,shell=True)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (0, 0, 255)

            # Line thickness of 2 px
            thickness = 2
            slice_3_layer = cv2.putText(slice_3_layer,str(slice_number) , org, font,  fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(outputfile_dir, os.path.basename(grayscale_filename).split('.nii')[0] + "_" + outputfile_suffix+'_'+ slice_number+".jpg"),slice_3_layer)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'masks_on_grayscale_colored')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)

    print(returnvalue)
    return  returnvalue

def masks_on_grayscale_colored(grayscale_filename,contrast_limits,mask_filename_list,mask_color_list,outputfile_dir,outputfile_suffix,npyfiledirectory=""):
    returnvalue=0

    try:
        command="echo successful at :: {}::grayscale_filename::{} >> /software/error.txt".format(inspect.stack()[0][3],grayscale_filename)
        subprocess.call(command,shell=True)
        command="echo successful at :: {}::contrast_limits::{} >> /software/error.txt".format(inspect.stack()[0][3],contrast_limits)
        subprocess.call(command,shell=True)
        command="echo successful at :: {}::mask_filename_list::{} >> /software/error.txt".format(inspect.stack()[0][3],mask_filename_list)
        subprocess.call(command,shell=True)
        command="echo successful at :: {}::mask_color_list::{} >> /software/error.txt".format(inspect.stack()[0][3],mask_color_list)
        subprocess.call(command,shell=True)
        command="echo successful at :: {}::outputfile_dir::{} >> /software/error.txt".format(inspect.stack()[0][3],outputfile_dir)
        subprocess.call(command,shell=True)
        command="echo successful at :: {}::outputfile_suffix::{} >> /software/error.txt".format(inspect.stack()[0][3],outputfile_suffix)
        subprocess.call(command,shell=True)
        command="echo successful at :: {}::npyfiledirectory::{} >> /software/error.txt".format(inspect.stack()[0][3],npyfiledirectory)
        subprocess.call(command,shell=True)
        grayscale_filename_np=nib.load(grayscale_filename).get_fdata()
        grayscale_filename_np=exposure.rescale_intensity( grayscale_filename_np , in_range=(contrast_limits[0], contrast_limits[1]))*255
        slice_3_layer= np.zeros([grayscale_filename_np.shape[0],grayscale_filename_np.shape[1],3])
        method_name="REGIS"
        # npyfiledirectory="/input"

        for i in range(grayscale_filename_np.shape[2]):
            slice_3_layer[:,:,0]= grayscale_filename_np[:,:,i] #imgray1
            slice_3_layer[:,:,1]= grayscale_filename_np[:,:,i] #imgray1
            slice_3_layer[:,:,2]= grayscale_filename_np[:,:,i]# imgray1

            for mask_filename_list_id in range(len(mask_filename_list)):
                mask_filename_np=nib.load(mask_filename_list[mask_filename_list_id]).get_fdata()
                command="echo successful at :: {}::sizemask_filename_np::{} >> /software/error.txt".format(inspect.stack()[0][3],mask_filename_np.shape[1])
                subprocess.call(command,shell=True)
                command="echo successful at :: {}::sizegrayscale_filename_np::{} >> /software/error.txt".format(inspect.stack()[0][3],grayscale_filename_np.shape[1])
                subprocess.call(command,shell=True)
                slice_3_layer[:,:,0][mask_filename_np[:,:,i]>0]=webcolors.name_to_rgb(mask_color_list[mask_filename_list_id])[2]
                slice_3_layer[:,:,1][mask_filename_np[:,:,i]>0]=webcolors.name_to_rgb(mask_color_list[mask_filename_list_id])[1]
                slice_3_layer[:,:,2][mask_filename_np[:,:,i]>0]=webcolors.name_to_rgb(mask_color_list[mask_filename_list_id])[0]
            slice_number="{0:0=3d}".format(i)
            # filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(grayscale_filename).split(".nii")[0])
            # this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+"_"+str(slice_number)+  "_V2.npy")
            # if os.path.exists(npyfiledirectory):
            #     try:
            slice_3_layer=draw_midline_on_a_slice(grayscale_filename,method_name,npyfiledirectory,slice_3_layer,slice_number)
                # except:
                #     pass
            # command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
            # subprocess.call(command,shell=True)
            # if os.path.exists(this_npyfile):
            #     calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
            #     x_points2=calculated_midline_points.item().get('x_axis')
            #     y_points2=calculated_midline_points.item().get('y_axis')
            #     x_points2=x_points2[:,0]
            #     y_points2=y_points2[:,0]
            #     slice_3_layer=cv2.line(slice_3_layer, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,0), 2)
            #     command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
            #     subprocess.call(command,shell=True)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)

                # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (0, 0, 255)

            # Line thickness of 2 px
            thickness = 2
            slice_3_layer = cv2.putText(slice_3_layer,str(slice_number) , org, font,  fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(outputfile_dir, os.path.basename(grayscale_filename).split('.nii')[0] + "_" + outputfile_suffix+'_'+ slice_number+".jpg"),slice_3_layer)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'masks_on_grayscale_colored')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)

    print(returnvalue)
    return  returnvalue
def call_masks_on_grayscale_colored(args):
    returnvalue=0
    try:
        grayscale_filename=args.stuff[1]
        contrast_limits=(int(args.stuff[2].split('_')[0]),int(args.stuff[2].split('_')[1]))
        mask_color_list=args.stuff[5].split('_')
        outputfile_dir=args.stuff[3]
        outputfile_suffix=args.stuff[4]
        npyfiledirectory=args.stuff[6]
        mask_filename_list=args.stuff[7:]
        masks_on_grayscale_colored(grayscale_filename,contrast_limits,mask_filename_list,mask_color_list,outputfile_dir,outputfile_suffix,npyfiledirectory)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_masks_on_grayscale_colored')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue

def masks_subtraction(mask_donor,mask_tobe_subtracted,output_mask_file):
    returnvalue=0
    try:
        mask_donor_np=nib.load(mask_donor).get_fdata()
        mask_donor_np[mask_donor_np>=0.5]=1
        mask_donor_np[mask_donor_np<1]=0
        mask_tobe_subtracted_np=nib.load(mask_tobe_subtracted).get_fdata()
        mask_tobe_subtracted_np[mask_tobe_subtracted_np>=0.5]=1
        mask_tobe_subtracted_np[mask_tobe_subtracted_np<1]=0
        mask_donor_np[mask_tobe_subtracted_np>0]=0
        array_img = nib.Nifti1Image(mask_donor_np,affine=nib.load(mask_donor).affine, header=nib.load(mask_donor).header)
        nib.save(array_img, output_mask_file)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'masks_subtraction')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue

def call_masks_subtraction(args):
    returnvalue=0
    try:
        mask_donor=args.stuff[1]
        mask_tobe_subtracted=args.stuff[2]
        output_mask_file=args.stuff[3]
        masks_subtraction(mask_donor,mask_tobe_subtracted,output_mask_file)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_masks_subtraction')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue
def call_calculate_volume_mask_from_yasheng(args):

    returnvalue=0

    try:
        mask_np=nib.load(args.stuff[1]).get_fdata()
        single_voxel_volume=np.prod(np.array(nib.load(args.stuff[2]).header["pixdim"][1:4]))
        # righthalf_np=nib.load(args.stuff[2]).get_fdata()
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_calculate_volume')
        subprocess.call(command,shell=True)
        column_name=args.stuff[3]
        column_name=column_name.replace('levelset_','')
        column_name=column_name.replace('unet_','')
        column_name=column_name.replace('_unet','')
        column_name=column_name.upper()
        column_name=column_name.replace('4DL','SAH')
        filename_to_write=args.stuff[4]
        returnvalue=calculate_volume(mask_np,single_voxel_volume=single_voxel_volume,column_name=column_name,filename_to_write=filename_to_write)
        # returnvalue=1

        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_calculate_volume')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue


def calculate_volume(mask_np,single_voxel_volume=1,column_name='test',filename_to_write="test.csv"):
    returnvalue="NONE"


    try:
        mask_np[mask_np>=0.5]=1
        mask_np[mask_np<1]=0
        volume=single_voxel_volume*np.sum(mask_np)
        # infarct_total_voxels_volume =np.sum(mask_np)*
        volume=volume/1000
        volume=round(volume,2)
        # righthalf_np[righthalf_np>0]=1
        # righthalf_np[righthalf_np<1]=0
        # left_right_ratio=np.sum(lefthalf_np)/np.sum(righthalf_np)
        # if np.sum(lefthalf_np) > np.sum(righthalf_np) :
        #     left_right_ratio=np.sum(righthalf_np)/np.sum(lefthalf_np)
        returnvalue=volume
        left_right_ratio_df=pd.DataFrame([volume])
        left_right_ratio_df.columns=[column_name]
        left_right_ratio_df.to_csv(filename_to_write,index=False)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],"calculate_volume")
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print('left_right_ratio'+str(returnvalue))
    return  returnvalue
def call_calculate_volume(args):

    returnvalue=0
    try:
        mask_np=nib.load(args.stuff[1]).get_fdata()
        single_voxel_volume=np.prod(np.array(nib.load(args.stuff[1]).header["pixdim"][1:4]))
        # righthalf_np=nib.load(args.stuff[2]).get_fdata()
        column_name=args.stuff[2]
        column_name=column_name.replace('levelset_','')
        column_name=column_name.replace('unet_','')
        column_name=column_name.upper()
        column_name=column_name.replace('4DL','SAH')
        filename_to_write=args.stuff[3]
        returnvalue=calculate_volume(mask_np,single_voxel_volume=single_voxel_volume,column_name=column_name,filename_to_write=filename_to_write)
        # returnvalue=1

        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_calculate_volume')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue
def numberlist_to_csv(value_col,name_col,file_to_save):

    returnvalue=0
    try:
        value_col_df=pd.DataFrame(value_col)
        value_col_df.columns=[name_col]
        value_col_df.to_csv(file_to_save,index=False)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_calculate_volume')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue
def call_slice_num_to_csv(args):
    returnvalue=0
    try:
        value_col=nib.load(args.stuff[1]).get_fdata().shape[2]
        name_col=args.stuff[2]
        file_to_save=args.stuff[3]
        numberlist_to_csv([value_col],name_col,file_to_save)
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_calculate_volume')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue

def ratio_left_right(lefthalf_np,righthalf_np,column_name='test',filename_to_write="test.csv"):
    returnvalue="NONE"


    try:
        lefthalf_np[lefthalf_np>=0.5]=1
        lefthalf_np[lefthalf_np<1]=0
        righthalf_np[righthalf_np>=0.5]=1
        righthalf_np[righthalf_np<1]=0
        if np.sum(lefthalf_np) > 0 and np.sum(righthalf_np) > 0:
            left_right_ratio=np.sum(lefthalf_np)/np.sum(righthalf_np)
            if np.sum(lefthalf_np) > np.sum(righthalf_np) :
                left_right_ratio=np.sum(righthalf_np)/np.sum(lefthalf_np)
            left_right_ratio=round(left_right_ratio,2)
            returnvalue=left_right_ratio
            left_right_ratio_df=pd.DataFrame([left_right_ratio])
            left_right_ratio_df.columns=[column_name]
            left_right_ratio_df.to_csv(filename_to_write,index=False)
            command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'ratio_left_right')
            subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    print('left_right_ratio'+str(returnvalue))
    return  returnvalue
def call_ratio_left_right(args):
    returnvalue=0
    try:
        lefthalf_np=nib.load(args.stuff[1]).get_fdata()
        righthalf_np=nib.load(args.stuff[2]).get_fdata()
        column_name=args.stuff[3]
        column_name=column_name.replace('levelset_','')
        column_name=column_name.replace('unet_','')
        column_name=column_name.upper()
        column_name=column_name.replace('4DL','SAH')
        filename_to_write=args.stuff[4]
        returnvalue=ratio_left_right(lefthalf_np,righthalf_np,column_name=column_name,filename_to_write=filename_to_write)
        returnvalue=1

        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'call_ratio_left_right')
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    print(returnvalue)
    return  returnvalue
def original_to_512x512(niftifilename,niftifilenameoutput):
    try:
        filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata())
        empty_header = nib.Nifti1Header()
        array_img= nib.Nifti1Image(filename_gray_data_np,affine=None,header=empty_header)
        nib.save(array_img, niftifilenameoutput)
    except:
        command="echo failed at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],niftifilename)
        subprocess.call(command,shell=True)
def call_original_to_512x512(args):
    niftifilename=args.stuff[1]
    niftifilenameoutput=args.stuff[2]
    original_to_512x512(niftifilename,niftifilenameoutput)


def divide_a_mask_into_left_right_submasks_v1(niftifilename,Mask_filename,npyfiledirectory,OUTPUT_DIRECTORY) :
    returnvalue=0
    try:
        # command="echo successful at :: {}::divide_a_mask_into_left_right_submasks_v1::{} >> /software/error.txt".format(inspect.stack()[0][3],Mask_filename)
        # subprocess.call(command,shell=True)
        Mask_filename_fdata=nib.load(Mask_filename).get_fdata()
        Mask_filename_fdata_June21_2023=nib.load(Mask_filename).get_fdata()
        Mask_filename_fdata_June21_2023_np=resizeinto_512by512(Mask_filename_fdata_June21_2023)
        Mask_filename_data_np=resizeinto_512by512(Mask_filename_fdata)
        Mask_filename_data_np[Mask_filename_data_np>1]=0
        Mask_filename_data_np[Mask_filename_data_np>=0.5]=1
        Mask_filename_data_np[Mask_filename_data_np<1]=0
        filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata())
        numpy_image=filename_gray_data_np
        left_half_filename=Mask_filename.split('.nii')[0]+'_left_half_originalRF.nii.gz'
        right_half_filename=Mask_filename.split('.nii')[0]+'_right_half_originalRF.nii.gz'
        lefthalf_mask_np_3d= np.zeros([numpy_image.shape[0],numpy_image.shape[1],numpy_image.shape[2]])
        righthalf_mask_np_3d= np.zeros([numpy_image.shape[0],numpy_image.shape[1],numpy_image.shape[2]])
        for img_idx in range(numpy_image.shape[2]):
            if img_idx>0 and img_idx < numpy_image.shape[2]:
                method_name="REGIS"
                slice_number="{0:0=3d}".format(img_idx)
                # filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
                # this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
                filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
                this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+ "_"+str(slice_number)+ "_V2"+ ".npy")
                if os.path.exists(this_npyfile):
                    command="echo successful at :: {}::this_npyfile::{} >> /software/error.txt".format(inspect.stack()[0][3],this_npyfile)
                    subprocess.call(command,shell=True)
                    calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
                    x_points2=calculated_midline_points.item().get('x_axis')
                    y_points2=calculated_midline_points.item().get('y_axis')
                    x_points2=x_points2[:,0]
                    y_points2=y_points2[:,0]

                    img_with_line=Mask_filename_data_np[:,:,img_idx]
                    # img_with_line[img_with_line>=0.5]=1
                    # img_with_line[img_with_line<1]=0
                    img_with_line_nonzero_id = np.transpose(np.nonzero(img_with_line))
                    # Mask_filename_data_np_idx=Mask_filename_fdata_June21_2023_np[:,:,img_idx]
                    # Mask_filename_data_np_idx[Mask_filename_data_np_idx>=0.5]=1
                    # Mask_filename_data_np_idx[Mask_filename_data_np_idx<1]=0
                    for non_zero_pixel in img_with_line_nonzero_id:
                        xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
                        if xx>0: ## RIGHT
                            righthalf_mask_np_3d[:,:,img_idx][non_zero_pixel[0],non_zero_pixel[1]]=1
                        if xx<0: ## LEFT
                            lefthalf_mask_np_3d[:,:,img_idx][non_zero_pixel[0],non_zero_pixel[1]]=1
        # if numpy_image.shape[1] == 512 :
        if nib.load(niftifilename).get_fdata().shape[1]==512:
            whenOFsize512x512_new_flip_np(lefthalf_mask_np_3d,niftifilename,left_half_filename,OUTPUT_DIRECTORY)
            whenOFsize512x512_new_flip_np(righthalf_mask_np_3d,niftifilename,right_half_filename,OUTPUT_DIRECTORY)
        else:
            whenOFsize512x5xx_new_flip_np(niftifilename,lefthalf_mask_np_3d,left_half_filename,OUTPUT_DIRECTORY)
            whenOFsize512x5xx_new_flip_np(niftifilename,righthalf_mask_np_3d,right_half_filename,OUTPUT_DIRECTORY)
        # array_img_left = nib.Nifti1Image(lefthalf_mask_np_3d,affine=target_nii.affine, header=new_header)
        # nib.save(array_img, os.path.join(output_directoryname,target_save))
        # levelset2originalRF_new_flip_with_params(original_file,levelset_file,OUTPUT_DIRECTORY)

        returnvalue=1
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],Mask_filename)
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  returnvalue
# import numpy as np
# from scipy.ndimage import label, center_of_mass
#
# import numpy as np
from scipy.ndimage import label, center_of_mass

def get_centroids_and_sizes(binary_slice):
    """
    Given a binary 2D image, returns:
    1. An image with centroids of connected components marked as 1.
    2. A list of tuples: ((y, x), count), where (y, x) is the centroid and count is number of pixels in the cluster.

    Args:
        binary_slice (np.ndarray): 2D binary image.

    Returns:
        centroid_image (np.ndarray): Binary image with centroid pixels = 1.
        centroid_info_list (List[Tuple[Tuple[int, int], int]]): List of centroid coordinates and pixel counts.
    """
    if binary_slice.ndim != 2:
        raise ValueError("Input must be a 2D binary image.")

    # Label the connected components
    labeled_array, num_features = label(binary_slice)

    # Prepare the return image
    centroid_image = np.zeros_like(binary_slice, dtype=np.uint8)
    centroid_info_list = []

    # Process each cluster
    for label_idx in range(1, num_features + 1):
        # Mask of current cluster
        cluster_mask = (labeled_array == label_idx)

        # Count of pixels in the cluster
        count = np.count_nonzero(cluster_mask)

        # Compute centroid
        centroid = center_of_mass(cluster_mask)
        y, x = np.round(centroid).astype(int)

        if 0 <= y < centroid_image.shape[0] and 0 <= x < centroid_image.shape[1]:
            centroid_image[y, x] = 1
            centroid_info_list.append(((y, x), count))

    return centroid_image, centroid_info_list
import numpy as np
from scipy.ndimage import label, center_of_mass

def distance_point_to_line_1(A, B, P):
    """
    Calculate signed distance from point P to line defined by A and B
    """
    return abs((P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])) / np.linalg.norm(np.array(B) - np.array(A))

def filter_clusters_by_distance_and_size(binary_slice, line_ptA, line_ptB, distance_thresh=100, size_thresh=200):
    """
    Removes clusters from a binary slice if their centroid is within `distance_thresh`
    from a line AND their size is below `size_thresh`.

    Returns the filtered binary image.
    """
    if binary_slice.ndim != 2:
        raise ValueError("Input must be a 2D binary image.")

    # Label connected components
    labeled_array, num_features = label(binary_slice)

    # Copy to modify
    filtered_array = np.copy(labeled_array)

    for label_idx in range(1, num_features + 1):
        cluster_mask = (labeled_array == label_idx)
        count = np.count_nonzero(cluster_mask)

        centroid = center_of_mass(cluster_mask)
        y, x = np.round(centroid).astype(int)

        if 0 <= y < binary_slice.shape[0] and 0 <= x < binary_slice.shape[1]:
            distance = distance_point_to_line(line_ptA, line_ptB, (y, x))

            if distance < distance_thresh and count < size_thresh:
                filtered_array[cluster_mask] = 0  # Remove this cluster

    # Convert back to binary image
    return (filtered_array > 0).astype(np.uint8)


def distance_mask_point_from_midline(niftifilename,Mask_filename_data_np_1,npyfiledirectory) :
    returnvalue=0
    try:
        # Mask_filename_fdata=nib.load(Mask_filename).get_fdata()
        # Mask_filename_fdata_June21_2023=nib.load(Mask_filename).get_fdata()
        # Mask_filename_fdata_June21_2023_np=resizeinto_512by512(Mask_filename_fdata_June21_2023)
        # Mask_filename_data_np=resizeinto_512by512(Mask_filename_fdata)
        # Mask_filename_data_np This is already 512x512 from the code where this function is called.
        Mask_filename_data_np=Mask_filename_data_np_1.copy()
        Mask_filename_data_np[Mask_filename_data_np>1]=0
        Mask_filename_data_np[Mask_filename_data_np>=0.5]=1
        Mask_filename_data_np[Mask_filename_data_np<1]=0
        filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata())
        numpy_image=filename_gray_data_np
        # left_half_filename=Mask_filename.split('.nii')[0]+'_left_half_originalRF.nii.gz'
        # right_half_filename=Mask_filename.split('.nii')[0]+'_right_half_originalRF.nii.gz'
        # lefthalf_mask_np_3d= np.zeros([numpy_image.shape[0],numpy_image.shape[1],numpy_image.shape[2]])
        # righthalf_mask_np_3d= np.zeros([numpy_image.shape[0],numpy_image.shape[1],numpy_image.shape[2]])
        for img_idx in range(numpy_image.shape[2]):
            if img_idx>0 and img_idx < numpy_image.shape[2]:
                method_name="REGIS"
                slice_number="{0:0=3d}".format(img_idx)
                filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
                this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
                if os.path.exists(this_npyfile):
                    calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
                    x_points2=calculated_midline_points.item().get('x_axis')
                    y_points2=calculated_midline_points.item().get('y_axis')
                    x_points2=x_points2[:,0]
                    y_points2=y_points2[:,0]
                    filtered_slice=filter_clusters_by_distance_and_size(Mask_filename_data_np[:,:,img_idx], (int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])), distance_thresh=5000, size_thresh=25)
                    if img_idx > Mask_filename_data_np.shape[2]/2:
                        filtered_slice=filter_clusters_by_distance_and_size(filtered_slice, (int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])), distance_thresh=50, size_thresh=100)
                    Mask_filename_data_np[:,:,img_idx]=filtered_slice
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],niftifilename)
        subprocess.call(command,shell=True)
        return Mask_filename_data_np
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  Mask_filename_data_np


def divide_a_mask_into_left_right_submasks(niftifilename,Mask_filename,npyfiledirectory,OUTPUT_DIRECTORY) :
    returnvalue=0
    try:
        Mask_filename_fdata=nib.load(Mask_filename).get_fdata()
        Mask_filename_fdata_June21_2023=nib.load(Mask_filename).get_fdata()
        Mask_filename_fdata_June21_2023_np=resizeinto_512by512(Mask_filename_fdata_June21_2023)
        Mask_filename_data_np=resizeinto_512by512(Mask_filename_fdata)
        Mask_filename_data_np[Mask_filename_data_np>1]=0
        Mask_filename_data_np[Mask_filename_data_np>=0.5]=1
        Mask_filename_data_np[Mask_filename_data_np<1]=0
        filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata())
        numpy_image=filename_gray_data_np
        left_half_filename=Mask_filename.split('.nii')[0]+'_left_half_originalRF.nii.gz'
        right_half_filename=Mask_filename.split('.nii')[0]+'_right_half_originalRF.nii.gz'
        lefthalf_mask_np_3d= np.zeros([numpy_image.shape[0],numpy_image.shape[1],numpy_image.shape[2]])
        righthalf_mask_np_3d= np.zeros([numpy_image.shape[0],numpy_image.shape[1],numpy_image.shape[2]])
        for img_idx in range(numpy_image.shape[2]):
            if img_idx>0 and img_idx < numpy_image.shape[2]:
                method_name="REGIS"
                slice_number="{0:0=3d}".format(img_idx)
                filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
                this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
                if os.path.exists(this_npyfile):
                    calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
                    x_points2=calculated_midline_points.item().get('x_axis')
                    y_points2=calculated_midline_points.item().get('y_axis')
                    x_points2=x_points2[:,0]
                    y_points2=y_points2[:,0]

                    img_with_line=Mask_filename_data_np[:,:,img_idx]
                    # img_with_line[img_with_line>=0.5]=1
                    # img_with_line[img_with_line<1]=0
                    img_with_line_nonzero_id = np.transpose(np.nonzero(img_with_line))
                    # Mask_filename_data_np_idx=Mask_filename_fdata_June21_2023_np[:,:,img_idx]
                    # Mask_filename_data_np_idx[Mask_filename_data_np_idx>=0.5]=1
                    # Mask_filename_data_np_idx[Mask_filename_data_np_idx<1]=0
                    for non_zero_pixel in img_with_line_nonzero_id:
                        xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
                        if xx>0: ## RIGHT
                            righthalf_mask_np_3d[:,:,img_idx][non_zero_pixel[0],non_zero_pixel[1]]=1
                        if xx<0: ## LEFT
                            lefthalf_mask_np_3d[:,:,img_idx][non_zero_pixel[0],non_zero_pixel[1]]=1
        # if numpy_image.shape[1] == 512 :
        if nib.load(niftifilename).get_fdata().shape[1]==512:
            whenOFsize512x512_new_flip_np(lefthalf_mask_np_3d,niftifilename,left_half_filename,OUTPUT_DIRECTORY)
            whenOFsize512x512_new_flip_np(righthalf_mask_np_3d,niftifilename,right_half_filename,OUTPUT_DIRECTORY)
        else:
            whenOFsize512x5xx_new_flip_np(niftifilename,lefthalf_mask_np_3d,left_half_filename,OUTPUT_DIRECTORY)
            whenOFsize512x5xx_new_flip_np(niftifilename,righthalf_mask_np_3d,right_half_filename,OUTPUT_DIRECTORY)
        # array_img_left = nib.Nifti1Image(lefthalf_mask_np_3d,affine=target_nii.affine, header=new_header)
        # nib.save(array_img, os.path.join(output_directoryname,target_save))
        # levelset2originalRF_new_flip_with_params(original_file,levelset_file,OUTPUT_DIRECTORY)

        returnvalue=1
        command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],Mask_filename)
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  returnvalue

# def divide_a_mask_into_left_right_submasks(maskfilename):
#     returnvalue=0
#     try:
#         returnvalue=1
#         command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],maskfilename)
#         subprocess.call(command,shell=True)
#     except:
#         command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
#         subprocess.call(command,shell=True)
#     return  returnvalue
def call_divide_a_mask_into_left_right_submasks(args):
    returnvalue=0
    try:
        niftifilename=args.stuff[1]
        Mask_filename=args.stuff[2]
        npyfiledirectory=args.stuff[3]
        OUTPUT_DIRECTORY=args.stuff[4]
        divide_a_mask_into_left_right_submasks(niftifilename,Mask_filename,npyfiledirectory,OUTPUT_DIRECTORY)
        returnvalue=1
        command="echo successful at :: {} >> /software/error.txt".format(inspect.stack()[0][3]) #                subprocess.call("echo " + "scanId1type::{}  >> /workingoutput/error.txt".format(scanId) ,shell=True )
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  returnvalue
def call_divide_a_mask_into_left_right_submasks_v1(args):
    returnvalue=0
    try:
        niftifilename=args.stuff[1]
        Mask_filename=args.stuff[2]
        npyfiledirectory=args.stuff[3]
        OUTPUT_DIRECTORY=args.stuff[4]
        divide_a_mask_into_left_right_submasks_v1(niftifilename,Mask_filename,npyfiledirectory,OUTPUT_DIRECTORY)
        returnvalue=1
        command="echo successful at :: {} >> /software/error.txt".format(inspect.stack()[0][3]) #                subprocess.call("echo " + "scanId1type::{}  >> /workingoutput/error.txt".format(scanId) ,shell=True )
        subprocess.call(command,shell=True)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
    return  returnvalue
# def remove_few_columns(csvfilename,columnstoremove):
#     csvfilename_df=pd.read_csv(csvfilename)
#     csvfilename_df.drop(columnstoremove, axis=1, inplace=True)
#     csvfilename_df.to_csv(csvfilename.split('.csv')[0]+'columndropped.csv',index=False)
# def determine_infarct_side(numpy_image,filename_gray_data_np_copy,niftifilename,npyfiledirectory,csf_seg_np,numpy_image_mask):
#     infarct_side='NONE'
#     left_ids=[]
#     right_ids=[]
#     left_side_ones=0
#     right_side_ones=0
#     for img_idx in range(numpy_image.shape[2]):
#         #             print("I AM HERE 4")
#         if img_idx>0 and img_idx < numpy_image.shape[2] and  filename_gray_data_np_copy.shape==csf_seg_np.shape:
#             method_name="REGIS"
#
#             slice_number="{0:0=3d}".format(img_idx)
#             filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
#             this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
#
#             #                this_npyfile=os.path.join(npyfiledirectory,os.path.basename(niftifilename).split(".nii")[0]+str(img_idx)+npyfileextension)
#             if os.path.exists(this_npyfile):
#                 calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
#                 x_points2=calculated_midline_points.item().get('x_axis') #,y_points2=points_on_line(extremepoints)
#                 y_points2=calculated_midline_points.item().get('y_axis')
#                 #                            slice_3_layer= np.zeros([numpy_image.shape[0],numpy_image.shape[1],3])
#                 x_points2=x_points2[:,0]
#                 y_points2=y_points2[:,0]
#
#                 #################################################
#                 ######################################################################
#                 img_with_line_nonzero_id = np.transpose(np.nonzero(np.copy(numpy_image_mask[:,:,img_idx])))
#                 #                    thisimage=filename_gray_data_np_1[:,:,img_idx]
#                 #                    current_left_num=0
#                 #                    current_right_num=0
#                 #                    slice_3_layer= np.zeros([I_t_gray.shape[0],I_t_gray.shape[1],3])
#                 #                    slice_3_layer[:,:,0]= I_t_gray #imgray1
#                 #                    slice_3_layer[:,:,1]= I_t_gray #imgray1
#                 #                    slice_3_layer[:,:,2]= I_t_gray# imgray1
#
#                 #                     infarct_side="NO INFARCT"
#
#                 for non_zero_pixel in img_with_line_nonzero_id:
#                     xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
#                     if xx>0: ## RIGHT
#                         #                            slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                         #                            slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=100
#                         #                            slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=200
#                         right_side_ones = right_side_ones + 1 # I_t_mask[non_zero_pixel[0],non_zero_pixel[1]]
#                         right_ids.append([non_zero_pixel[0],non_zero_pixel[1],img_idx])
#                     #                        print()
#                     if xx<0: ## LEFT
#                         #                            slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=100
#                         #                            slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=0
#                         #                            slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=100
#                         left_side_ones = left_side_ones + 1 #I_t_mask[non_zero_pixel[0],non_zero_pixel[1]]
#                         left_ids.append([non_zero_pixel[0],non_zero_pixel[1],img_idx])
#
#     if (left_side_ones > right_side_ones):
#         infarct_side="LEFT"
#         for right_id in right_ids:
#             #             print("I am Left")
#             numpy_image_mask[right_id[0],right_id[1],right_id[2]]=np.min(numpy_image_mask)
#
#
#     if (right_side_ones > left_side_ones):
#         infarct_side="RIGHT"
#         for left_id in left_ids:
#             #             print("I am Right")
#             numpy_image_mask[left_id[0],left_id[1],left_id[2]]=np.min(numpy_image_mask)
#     return infarct_side,numpy_image_mask


# def call_nwu_csfcompartment():
#     measure_NWU_after_subt_csf_Oct_5_2020()
#     measure_compartments_with_reg_round5_one_file_sh_v1()

def whichsideofline(line_pointA,line_pointB,point_todecide):

    return (point_todecide[0]-line_pointA[0])*(line_pointB[1]-line_pointA[1])  -  (point_todecide[1]-line_pointA[1])*(line_pointB[0]-line_pointA[0])

import numpy as np

def distance_point_to_line(line_pointA, line_pointB, point):
    """
    Compute the perpendicular distance from a point to a line defined by two points in 2D.
    """
    x0, y0 = point
    x1, y1 = line_pointA
    x2, y2 = line_pointB

    # Vector cross product magnitude divided by line length
    numerator = abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1))
    denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if denominator == 0:
        raise ValueError("Line points must not be identical")

    return numerator / denominator

def mirror_a_mask(args):
    try:
        niftifilename=args.stuff[1]
        Mask_filename=args.stuff[2]
        npyfiledirectory=args.stuff[3]
        mask_flipped_filename=args.stuff[4]
        Mask_filename_data_nib=nib.load(Mask_filename)
        Mask_filename_data_np=Mask_filename_data_nib.get_fdata() ##*255 #resizeinto_512by512(nib.load(Mask_filename).get_fdata()) #nib.load(Mask_filename).get_fdata()
        # filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata()) #nib.load(niftifilename).get_fdata() #
        # filename_gray_data_np=contrast_stretch_np(filename_gray_data_np,1) #exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
        # numpy_image=normalizeimage0to1(filename_gray_data_np)*255 #filename_gray_data_np #
        numpy_image_mask=Mask_filename_data_np
        numpy_image_mask_copy=(nib.load(Mask_filename).get_fdata())*0
        for img_idx in range(numpy_image_mask.shape[2]):
            if img_idx>0 and img_idx < numpy_image_mask.shape[2]: ## and  filename_gray_data_np_copy.shape==csf_seg_np.shape:
                method_name="REGIS"
                slice_number="{0:0=3d}".format(img_idx)
                filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
                this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
                # filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
                # this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+ "_"+str(slice_number)+ "_V2"+ ".npy")
                print(this_npyfile)
                command="echo passed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
                subprocess.call(command,shell=True)
                command="echo npyfilename :: {} >> /software/error.txt".format(this_npyfile)
                subprocess.call(command,shell=True)
                if os.path.exists(this_npyfile):
                    calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
                    x_points2=calculated_midline_points.item().get('x_axis')
                    y_points2=calculated_midline_points.item().get('y_axis')
                    x_points2=x_points2[:,0]
                    y_points2=y_points2[:,0]
                    v1=np.array([512,0]) ## point from the image
                    v2_1=np.array([x_points2[0],y_points2[0]]) ## point 1 from the midline
                    v2_2=np.array([x_points2[1],y_points2[1]]) ## point 2 from the midline
                    v2=v2_2-v2_1
                    angle=  angle_bet_two_vector(v1,v2)
                    angleRad=angle_bet_two_vectorRad(v1,v2)
                    ## translation:
                    points=np.array([[x_points2[0],y_points2[0]],[x_points2[511],y_points2[511]]])
                    mid_point_line=np.mean(points,axis=0)
                    # delta translation:
                    image_midpoint=np.array([int(Mask_filename_data_np[:,:,img_idx].shape[0]/2),int(Mask_filename_data_np[:,:,img_idx].shape[1]/2)]) #np.array([255,255])
                    translation_delta=image_midpoint-mid_point_line
                    M = np.float32([[1,0,translation_delta[0]],[0,1,translation_delta[1]]])
                    I_t_mask =cv2.warpAffine(np.copy(numpy_image_mask[:,:,img_idx]),M,(Mask_filename_data_np[:,:,img_idx].shape[0],Mask_filename_data_np[:,:,img_idx].shape[1]) , flags= cv2.INTER_NEAREST) # cv2.warpAffine(np.copy(numpy_image_mask[:,:,img_idx]),M,(512,512) , flags= cv2.INTER_NEAREST)

                    #########################################################################
                    translate_points= points+translation_delta
                    points=translate_points
                    ## translation matrix
                    p1x,p1y= rotate_around_point_highperf(np.array([points[0][0],points[0][1]]), angleRad, origin=(255,255))
                    p2x,p2y= rotate_around_point_highperf(np.array([points[1][0],points[1][1]]), angleRad, origin=(255,255))
                    points1=np.array([[p1x,p1y],[p2x,p2y]])
                    I_t_r_mask=rotate_image(I_t_mask,(255,255),angle)
                    I_t_r_f_mask=cv2.flip(I_t_r_mask,0)
                    I_t_r_f_rinv_mask=rotate_image(I_t_r_f_mask,(256,256),-angle)
                    p1x,p1y= rotate_around_point_highperf(np.array([points1[0][0],points1[0][1]]), -angleRad, origin=(255,255))
                    p2x,p2y= rotate_around_point_highperf(np.array([points1[1][0],points1[1][1]]), -angleRad, origin=(255,255))
                    points1=np.array([[p1x,p1y],[p2x,p2y]])
                    M = np.float32([[1,0,-translation_delta[0]],[0,1,-translation_delta[1]]])
                    I_t_r_f_rinv_tinv_mask = cv2.warpAffine(I_t_r_f_rinv_mask,M,(512,512), flags= cv2.INTER_NEAREST )
                    numpy_image_mask_copy[:,:,img_idx]=I_t_r_f_rinv_tinv_mask  #* 255

        numpy_image_mask_flipped = nib.Nifti1Image(numpy_image_mask_copy,affine=Mask_filename_data_nib.affine,header=Mask_filename_data_nib.header)
        nib.save(numpy_image_mask_flipped, mask_flipped_filename)

    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
def call_infarct_and_reflectedinfarct_related_parameters(args):
    infarct_and_reflectedinfarct_related_parameters(args)
    return 1

def remove_voxels_from_mask_with_threshold_range(args):
    mask_nib=nib.load(args.stuff[1])
    mask=mask_nib.get_fdata()
    grayscale_image=nib.load(args.stuff[2]).get_fdata()
    threshold_lower_limit=int(args.stuff[3])
    threshold_upper_limit=int(args.stuff[4])
    mask[grayscale_image>=threshold_lower_limit]=np.min(mask)
    mask[grayscale_image<=threshold_upper_limit]=np.min(mask)
    array_img= nib.Nifti1Image(mask,affine=mask_nib.affine,header=mask_nib.header)
    mask_nib_outputfilename=args.stuff[1].split('.nii')[0]+'_range_'+str(threshold_lower_limit)+'_to_'+str(threshold_upper_limit)+'.nii.gz'
    nib.save(array_img, mask_nib_outputfilename)

def infarct_and_reflectedinfarct_related_parameters(args):
    numerator_count=""
    numerator_mean=""
    numerator_volume=""
    nwu_like_ratio=''
    denominator_count=''
    denominator_mean=''
    denominator_volume = ''
    numerator_mask=nib.load(args.stuff[1]).get_fdata()
    denominator_mask=nib.load(args.stuff[2]).get_fdata()
    grayscale_image=resizeinto_512by512(nib.load(args.stuff[3]).get_fdata())
    threshold_lower_limit_inf=int(args.stuff[4])
    threshold_upper_limit_inf=int(args.stuff[5])
    threshold_lower_limit_norm=int(args.stuff[6])
    threshold_upper_limit_norm=int(args.stuff[7])
    session_ID=args.stuff[8]
    csvfilename=args.stuff[9]
    maskname_other_than_infarct=''
    if len(args.stuff)>10:
        maskname_other_than_infarct=args.stuff[10]
    try:
        # columnname="INFARCT VOX_NUMBERS"
        # infarctmask_count=count_voxels_mask_binary(args.stuff[1])
        # fill_datapoint_each_sessionn(session_ID,columnname,infarctmask_count,csvfilename)
        # columnname="NON INFARCT VOX_NUMBERS"
        # infarct_mirror_mask_count=count_voxels_mask_binary(args.stuff[2])
        # fill_datapoint_each_sessionn(session_ID,columnname,infarct_mirror_mask_count,csvfilename)
        # columns_name=["SCAN_NAME" , "NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "INFARCT THRESH RANGE","NORMAL THRESH RANGE"]

        numerator=grayscale_image[numerator_mask>0] #.flatten()
        numerator=numerator[numerator>=threshold_lower_limit_inf]
        numerator=numerator[numerator<=threshold_upper_limit_inf]
        numerator=numerator.flatten()
        numerator_count=len(numerator)

        columnname="INFARCT VOX_NUMBERS"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_VOX_NUMBERS"
        fill_datapoint_each_sessionn(session_ID,columnname,numerator_count,csvfilename)
        numerator_mean=np.mean(numerator)
        columnname= "INFARCT DENSITY"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_DENSITY"
        fill_datapoint_each_sessionn(session_ID,columnname,numerator_mean,csvfilename)
        numerator_volume = numerator_count*np.prod(np.array(nib.load(args.stuff[3]).header["pixdim"][1:4]))/1000
        columnname="INFARCT VOLUME"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_VOLUME"
        fill_datapoint_each_sessionn(session_ID,columnname,numerator_volume,csvfilename)
        denominator=grayscale_image[denominator_mask>0] #.flatten()
        denominator=denominator[denominator>=threshold_lower_limit_norm]
        denominator=denominator[denominator<=threshold_upper_limit_norm]
        denominator_count=len(denominator)
        columnname="NON INFARCT VOX_NUMBERS"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_REFLECTION_VOX_NUMBERS"
        fill_datapoint_each_sessionn(session_ID,columnname,denominator_count,csvfilename)
        denominator_mean=np.mean(denominator)
        columnname= "NON INFARCT DENSITY"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_REFLECTION_DENSITY"
        fill_datapoint_each_sessionn(session_ID,columnname,denominator_mean,csvfilename)
        denominator_volume = denominator_count*np.prod(np.array(nib.load(args.stuff[3]).header["pixdim"][1:4]))/1000
        columnname="INFARCT REFLECTION VOLUME"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_REFLECTION_VOLUME"
        fill_datapoint_each_sessionn(session_ID,columnname,denominator_volume,csvfilename)
        if denominator_mean>0:
            nwu_like_ratio=(1 - (numerator_mean/denominator_mean))*100
            nwu_like_ratio=round(nwu_like_ratio,2)
        columnname="NWU"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_NWU"
        fill_datapoint_each_sessionn(session_ID,columnname,nwu_like_ratio,csvfilename)
        columnname="INFARCT THRESH RANGE"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct + "_THRESH_RANGE"
        fill_datapoint_each_sessionn(session_ID,columnname,str(threshold_lower_limit_inf)+'to'+str(threshold_upper_limit_inf),csvfilename)
        columnname="NORMAL THRESH RANGE"
        if len(maskname_other_than_infarct) >2:
            columnname=maskname_other_than_infarct+"_REFLECTION_THRESH_RANGE"
        fill_datapoint_each_sessionn(session_ID,columnname,str(threshold_lower_limit_norm)+'to'+str(threshold_upper_limit_norm),csvfilename)

        # scan_name=os.path.basename(args.stuff[3]).split('.nii')[0]  ##+ "_TOTAL"
        # values=[scan_name,nwu_like_ratio,numerator_count,round(numerator_mean,2),denominator_count,round(denominator_mean,2),round(numerator_volume,2),round(denominator_volume,2),str(threshold_lower_limit_inf)+'to'+str(threshold_upper_limit_inf),str(threshold_lower_limit_norm)+'to'+str(threshold_upper_limit_norm)]
        # nwu_like_ratio_df=pd.DataFrame(values) #[scan_name,"","","","",nwu_like_ratio,numerator_count,numerator_mean,denominator_count,denominator_mean,numerator_volume,denominator_volume,"","","","",str(threshold_lower_limit_inf)+'to'+str(threshold_upper_limit_inf),str(threshold_lower_limit_norm)+'to'+str(threshold_upper_limit_norm)])
        # nwu_like_ratio_df=nwu_like_ratio_df.T
        # columns_name=["SCAN_NAME" , "NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "INFARCT THRESH RANGE","NORMAL THRESH RANGE"]
        # nwu_like_ratio_df.columns=columns_name #["FileName_slice" , "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME", "INFARCT SIDE","NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "BET VOLUME","CSF RATIO","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF","INFARCT THRESH RANGE","NORMAL THRESH RANGE"]
        # nwu_like_ratio_df.to_csv(args.stuff[3].split('.nii')[0]+"_FROM_INFARCT.csv",index=False)
        # # columns_df=pd.DataFrame(["FileName_slice" , "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME", "INFARCT SIDE","NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "BET VOLUME","CSF RATIO","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF","INFARCT THRESH RANGE","NORMAL THRESH RANGE"])
        # # columns_df.to_csv(args.stuff[3].split('.nii')[0]+"_NWU_cols.csv",index=False)
        # #  , "BET VOLUME","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF",
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
def call_side_of_lesion(args):
    lefthalf=args.stuff[1]
    righthalf=args.stuff[2]
    session_id=args.stuff[3]
    csvfilename=args.stuff[4]
    side_of_lesion(lefthalf,righthalf,session_id,csvfilename)
def side_of_lesion(lefthalf,righthalf,session_ID,csvfilename='NONE.csv'):
    lesion_side=''
    try:
        righthalf_count=count_voxels_mask_binary(righthalf) #,grayscale_image)
        lefthalf_count=count_voxels_mask_binary(lefthalf) #,grayscale_image)
        if righthalf_count>lefthalf_count:
            lesion_side='right'
        elif righthalf_count<lefthalf_count:
            lesion_side='left'
        columnname='LESION_SIDE'
        columnvalue=lesion_side
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass


def call_csf_related_parameters(args):
    lefthalf=args.stuff[1]
    righthalf=args.stuff[2]
    complete_csf=args.stuff[3]
    grayscale_image=args.stuff[4]
    session_ID=args.stuff[5]
    csvfilename=args.stuff[6]
    csf_related_parameters(lefthalf,righthalf,complete_csf,grayscale_image,session_ID,csvfilename=csvfilename)
def count_voxels_mask_binary(maskfilename):
    mask_np=nib.load(maskfilename).get_fdata()
    mask_np[mask_np>=0.5]=1
    mask_np[mask_np<1]=0
    voxel_count=np.sum(mask_np)
    return voxel_count
def volume_voxels_mask_binary(maskfilename,niftigrayfile):
    grayscale_image_nib=nib.load(niftigrayfile)
    voxel_count=count_voxels_mask_binary(maskfilename)
    mask_volume=voxel_count*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
    return mask_volume
def ratio_left_right_mask_binary(left_maskfile,right_maskfile):
    returnvalue=''
    right_mask_np_count=count_voxels_mask_binary(right_maskfile)
    left_mask_np_count=count_voxels_mask_binary(left_maskfile)
    if left_mask_np_count > 0 and right_mask_np_count > 0:
        left_right_ratio=left_mask_np_count/right_mask_np_count
        if left_right_ratio > right_mask_np_count:
            left_right_ratio=right_mask_np_count/left_right_ratio
        left_right_ratio=round(left_right_ratio,2)
        returnvalue=left_right_ratio
    return returnvalue
def csf_related_parameters(lefthalf,righthalf,complete_csf,grayscale_image,session_ID,csvfilename="NONE.csv"): #column_name='test',filename_to_write="test.csv"):
    returnvalue="NONE"
    # 'SCAN_NAME', "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME","CSF RATIO",
    righthalf_volume=''
    lefthalf_volume=''
    total_csf_volume=''
    left_right_ratio=''
    scan_name=''
    try:
        scan_name=os.path.basename(grayscale_image).split('.nii')[0]
        righthalf_volume=volume_voxels_mask_binary(righthalf,grayscale_image)
        columnname='csf_right'
        columnvalue=righthalf_volume
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
        # columns_name=["SCAN_NAME", "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME","CSF RATIO"]
        lefthalf_volume=volume_voxels_mask_binary(lefthalf,grayscale_image)
        columnname='csf_left'
        columnvalue=lefthalf_volume
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
        total_csf_volume=volume_voxels_mask_binary(complete_csf,grayscale_image)
        columnname='csf_total'
        columnvalue=total_csf_volume
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
        left_right_ratio=ratio_left_right_mask_binary(lefthalf,righthalf)
        columnname='csf_ratio'
        columnvalue=left_right_ratio
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)

        # lefthalf_np=nib.load(lefthalf).get_fdata()
        # righthalf_np=nib.load(righthalf).get_fdata()
        # complete_csf_np=nib.load(complete_csf).get_fdata()
        # grayscale_image_nib=nib.load(grayscale_image)
        # lefthalf_np[lefthalf_np>=0.5]=1
        # lefthalf_np[lefthalf_np<1]=0
        # righthalf_np[righthalf_np>=0.5]=1
        # righthalf_np[righthalf_np<1]=0
        # complete_csf_np[complete_csf_np>=0.5]=1
        # complete_csf_np[complete_csf_np<1]=0
        # righthalf_volume = np.sum(righthalf_np)*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
        # righthalf_volume=round(righthalf_volume,2)
        # lefthalf_volume = np.sum(lefthalf_np)*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
        # lefthalf_volume=round(lefthalf_volume,2)
        # complete_csf_np[complete_csf_np>=0.5]=1
        # complete_csf_np[complete_csf_np<1]=0
        # total_csf_volume=np.sum(complete_csf_np)*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
        # total_csf_volume=round(total_csf_volume,2)
        # if np.sum(lefthalf_np) > 0 and np.sum(righthalf_np) > 0:
        #     left_right_ratio=np.sum(lefthalf_np)/np.sum(righthalf_np)
        #     if np.sum(lefthalf_np) > np.sum(righthalf_np) :
        #         left_right_ratio=np.sum(righthalf_np)/np.sum(lefthalf_np)
        #     left_right_ratio=round(left_right_ratio,2)
        #     returnvalue=left_right_ratio
        #     # left_right_ratio_df=pd.DataFrame([left_right_ratio])
        #     # left_right_ratio_df.columns=[column_name]
        #     # left_right_ratio_df.to_csv(filename_to_write,index=False)
        #     command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'ratio_left_right')
        #     subprocess.call(command,shell=True)
        # # 'SCAN_NAME', "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME","CSF RATIO",
        # values=[scan_name,lefthalf_volume,righthalf_volume,total_csf_volume,left_right_ratio]
        # values_from_csf_df=pd.DataFrame(values) #[scan_name,"","","","",nwu_like_ratio,numerator_count,numerator_mean,denominator_count,denominator_mean,numerator_volume,denominator_volume,"","","","",str(threshold_lower_limit_inf)+'to'+str(threshold_upper_limit_inf),str(threshold_lower_limit_norm)+'to'+str(threshold_upper_limit_norm)])
        # values_from_csf_df=values_from_csf_df.T
        # columns_name=["SCAN_NAME", "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME","CSF RATIO"]
        # values_from_csf_df.columns=columns_name #["FileName_slice" , "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME", "INFARCT SIDE","NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "BET VOLUME","CSF RATIO","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF","INFARCT THRESH RANGE","NORMAL THRESH RANGE"]
        # values_from_csf_df.to_csv(os.path.join(os.path.dirname(grayscale_image),scan_name+"_FROM_CSF.csv"),index=False)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    print('left_right_ratio'+str(returnvalue))
    return  returnvalue
def call_bet_related_parameters(args):
    lefthalf=args.stuff[1]
    righthalf=args.stuff[2]
    complete_bet=args.stuff[3]
    grayscale_image=args.stuff[4]
    session_ID=args.stuff[5]
    csvfilename=args.stuff[6]
    bet_related_parameters(lefthalf,righthalf,complete_bet,grayscale_image,session_ID,csvfilename=csvfilename)
def number_of_slices(args):
    niftifilename=args.stuff[1]
    session_ID=args.stuff[2]
    if len(args.stuff)>3:
        csvfilename=args.stuff[3]
        columnname="slices"
        columnvalue=nib.load(niftifilename).shape[2]
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)

def bet_related_parameters(lefthalf,righthalf,complete_bet,grayscale_image,session_ID,csvfilename="NONE.csv"): #column_name='test',filename_to_write="test.csv"):
    returnvalue="NONE"
    # ,
    righthalf_volume=''
    lefthalf_volume=''
    total_csf_volume=''
    left_right_ratio=''
    scan_name=''
    try:
        righthalf_volume=volume_voxels_mask_binary(righthalf,grayscale_image)
        columnname="RIGHT BRAIN VOLUME without CSF"
        columnvalue=righthalf_volume
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
        lefthalf_volume=volume_voxels_mask_binary(lefthalf,grayscale_image)
        columnname="LEFT BRAIN VOLUME without CSF"
        columnvalue=lefthalf_volume
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
        complete_bet_volume=volume_voxels_mask_binary(complete_bet,grayscale_image)
        columnname="BET VOLUME"
        columnvalue=complete_bet_volume
        fill_datapoint_each_sessionn(session_ID,columnname,columnvalue,csvfilename)
        #
        # # columns_name=["SCAN_NAME", "BET VOLUME","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF"]
        # scan_name=os.path.basename(grayscale_image).split('.nii')[0]
        # lefthalf_np=nib.load(lefthalf).get_fdata()
        # righthalf_np=nib.load(righthalf).get_fdata()
        # # complete_csf_np=nib.load(complete_csf).get_fdata()
        # complete_bet_np=nib.load(complete_bet).get_fdata()
        # grayscale_image_nib=nib.load(grayscale_image)
        # lefthalf_np[lefthalf_np>=0.5]=1
        # lefthalf_np[lefthalf_np<1]=0
        # righthalf_np[righthalf_np>=0.5]=1
        # righthalf_np[righthalf_np<1]=0
        # complete_bet_np[complete_bet_np>=0.5]=1
        # complete_bet_np[complete_bet_np<1]=0
        # # complete_csf_np[complete_csf_np>=0.5]=1
        # # complete_csf_np[complete_csf_np<1]=0
        # ##################
        # # complete_bet_np[complete_csf_np>0]=0 #np.min(complete_bet_np)
        # # lefthalf_np[complete_csf_np>0]=0 #np.min(lefthalf_np)
        # # righthalf_np[complete_csf_np>0]=0 #np.min(righthalf_np)
        # ######################
        # righthalf_volume = np.sum(righthalf_np)*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
        # righthalf_volume=round(righthalf_volume,2)
        # lefthalf_volume = np.sum(lefthalf_np)*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
        # lefthalf_volume=round(lefthalf_volume,2)
        # total_bet_volume=np.sum(complete_bet_np)*np.prod(np.array(grayscale_image_nib.header["pixdim"][1:4]))/1000
        # total_bet_volume=round(total_bet_volume,2)
        # if np.sum(lefthalf_np) > 0 and np.sum(righthalf_np) > 0:
        #     left_right_ratio=np.sum(lefthalf_np)/np.sum(righthalf_np)
        #     if np.sum(lefthalf_np) > np.sum(righthalf_np) :
        #         left_right_ratio=np.sum(righthalf_np)/np.sum(lefthalf_np)
        #     left_right_ratio=round(left_right_ratio,2)
        #     returnvalue=left_right_ratio
        #     # left_right_ratio_df=pd.DataFrame([left_right_ratio])
        #     # left_right_ratio_df.columns=[column_name]
        #     # left_right_ratio_df.to_csv(filename_to_write,index=False)
        #     command="echo successful at :: {}::maskfilename::{} >> /software/error.txt".format(inspect.stack()[0][3],'ratio_left_right')
        #     subprocess.call(command,shell=True)
        # # 'SCAN_NAME', "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME","CSF RATIO",
        # values=[scan_name,total_bet_volume,lefthalf_volume,righthalf_volume]
        # values_from_csf_df=pd.DataFrame(values) #[scan_name,"","","","",nwu_like_ratio,numerator_count,numerator_mean,denominator_count,denominator_mean,numerator_volume,denominator_volume,"","","","",str(threshold_lower_limit_inf)+'to'+str(threshold_upper_limit_inf),str(threshold_lower_limit_norm)+'to'+str(threshold_upper_limit_norm)])
        # values_from_csf_df=values_from_csf_df.T
        # columns_name=["SCAN_NAME", "BET VOLUME","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF"]
        # values_from_csf_df.columns=columns_name #["FileName_slice" , "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME", "INFARCT SIDE","NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "BET VOLUME","CSF RATIO","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF","INFARCT THRESH RANGE","NORMAL THRESH RANGE"]
        # values_from_csf_df.to_csv(os.path.join(os.path.dirname(grayscale_image),scan_name+"_FROM_BET.csv"),index=False)
    except:
        command="echo failed at :: {} >> /software/error.txt".format(inspect.stack()[0][3])
        subprocess.call(command,shell=True)
        pass
    print('left_right_ratio'+str(returnvalue))
    return  returnvalue


# def measure_compartments_with_reg_round5_one_file_sh_v1() : #niftifilenamedir,npyfiledirectory,npyfileextension):
#     # $grayimage $betimage  $csfmaskimage ${infarctmaskimage}  $npyfiledirectory     $output_directory  $lower_threshold $upper_threshold
#     print(" I am in measure_compartments_with_reg_round5_one_file_sh_v1() ")
#     print("code added on July 15 2022")
#     niftifilename=sys.argv[1] ## THis is the  gray file:
#
#     niftifilenamedir=os.path.dirname(niftifilename) #sys.argv[3] #"/inputdirectory" #sys.argv[1] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/FU_CTs_Masks/CTs" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/MISSINGDATA1/" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/ALLCOHORTINONE/TILTED"
#
#     npyfiledirectory=sys.argv[5] #"/processedniftifiles" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/FU_CTs_Masks/SMOOTH_IML" # /media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly/"
#
#
#     print(sys.argv)
#     SLICE_OUTPUT_DIRECTORY=sys.argv[6] #"/outputdirectory" #sys.argv[4] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/FU_CTs_Masks/CSF_RL_VOL_OUTPUT" #sys.argv[4] ####"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/shellscript/RegistrationMethod/test"
#
#
#     lower_thresh="NA" #int(float(sys.argv[7]))
#     upper_thresh="NA" #int(float(sys.argv[8]))
#     lower_thresh_normal="NA"
#     upper_thresh_normal="NA"
#     print(niftifilename)
#     print("sys.argv[2]")
#     print(sys.argv[2])
#     infarct_side="NA"
#     NWU="NA"
#     infarct_pixels_number="NA"
#     infarct_pixels_density="NA"
#     nonfarct_pixels_number="NA"
#     noninfarct_pixels_density="NA"
#     overall_infarct_vol="NA"
#     overall_non_infarct_vol="NA"
#     infarct_total_voxels_volume="NA"
#     infarct_side="NONE"
#     left_brain_volume=0
#     right_brain_volume=0
#     gray_image_data=nib.load(sys.argv[1]).get_fdata()
#     bet_image_data=nib.load(sys.argv[2]).get_fdata()
#     csf_image_data=nib.load(sys.argv[3]).get_fdata()
#     Infarct_Mask_filename_June20=sys.argv[4]
#     Infarct_Mask_filename_June20_data=nib.load(Infarct_Mask_filename_June20).get_fdata()
#     # csf_image_data[Infarct_Mask_filename_June20_data>0]=np.min(csf_image_data)
#     Infarct_Mask_filename_June20_data_512=resizeinto_512by512(Infarct_Mask_filename_June20_data)
#     if gray_image_data.shape[0] == bet_image_data.shape[0] == csf_image_data.shape[0]  and gray_image_data.shape[1] == bet_image_data.shape[1] == csf_image_data.shape[1]  and  gray_image_data.shape[2] == bet_image_data.shape[2] == csf_image_data.shape[2]:
#
#         if os.path.exists(sys.argv[4]):
#             infarct_image_data=nib.load(sys.argv[4]).get_fdata()
#             print(sys.argv[4])
#
#             if  gray_image_data.shape[1]  ==  infarct_image_data.shape[1] and  gray_image_data.shape[0] == infarct_image_data.shape[0]    and gray_image_data.shape[2] == infarct_image_data.shape[2]:
#                 lower_thresh=int(float(sys.argv[7]))
#                 upper_thresh=int(float(sys.argv[8]))
#                 ## check if infarct file exists: sys.argv[4]
#                 lower_thresh,upper_thresh,lower_thresh_normal,upper_thresh_normal, infarct_total_voxels_volume,infarct_side,NWU,infarct_pixels_number,infarct_pixels_density,nonfarct_pixels_number,noninfarct_pixels_density, overall_infarct_vol,overall_non_infarct_vol= measure_NWU_after_subt_csf_Oct_5_2020()
#
#
#         niftifilename_basename_split_nii=os.path.basename(niftifilename).split(".nii")[0] #.split("_")
#         # bet_filename=niftifilename_basename_split_nii+BET_file_extension
#         bet_filename_path=sys.argv[2] #os.path.join(BET_OUTPUT_DIRECTORY,bet_filename)
#         # now=time.localtime()
#         date_time = time.strftime("_%m_%d_%Y",now)
#         grayfilename=niftifilename #os.path.join(niftifilenamedir,grayfilename)
#         thisfilebasename=os.path.basename(grayfilename).split("_resaved")[0]
#         # csvfile_with_vol_total=os.path.join(SLICE_OUTPUT_DIRECTORY,os.path.basename(grayfilename).split(".nii")[0] + "_threshold"+ str(lower_thresh) + "_" + str(upper_thresh) + "TOTAL.csv")
#         csvfile_with_vol_total=os.path.join(SLICE_OUTPUT_DIRECTORY,thisfilebasename + "_threshold"+ str(lower_thresh) + "_" + str(upper_thresh) + "TOTAL" +Version_Date+date_time + ".csv")
#
#
#
#         latexfilename=os.path.join(SLICE_OUTPUT_DIRECTORY,thisfilebasename+"_thresh_"+str(lower_thresh) + "_" +str(upper_thresh) + Version_Date + date_time+".tex")
#         # latexfilename=os.path.join(SLICE_OUTPUT_DIRECTORY,os.path.basename(grayfilename).split(".nii")[0]+"_thresh_"+str(lower_thresh) + "_" +str(upper_thresh) +".tex")
#
#
#         latex_start(latexfilename)
#         latex_begin_document(latexfilename)
#         row = ["FileName_slice" , "LEFT CSF VOLUME", "RIGHT CSF VOLUME","TOTAL CSF VOLUME", "INFARCT SIDE","NWU", "INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS", "NON INFARCT DENSITY","INFARCT VOLUME","INFARCT REFLECTION VOLUME", "BET VOLUME","CSF RATIO","LEFT BRAIN VOLUME without CSF" ,"RIGHT BRAIN VOLUME without CSF","INFARCT THRESH RANGE","NORMAL THRESH RANGE"]
#         col_names=np.copy(np.array(row))
#
#
#         with open(csvfile_with_vol_total, 'w') as f:
#             writer = csv.writer(f)
#             writer.writerow(row)
#
#         npyfileextension="REGISMethodOriginalRF_midline.npy"
#
#         CSF_Mask_filename=sys.argv[3] #os.path.join(csf_mask_directory,mask_basename)
#         print('CSF_Mask_filename')
#         print(CSF_Mask_filename)
#         print('niftifilename')
#         print(niftifilename)
#
#         left_pixels_num=0
#         right_pixels_num=0
#
#         infarct_mask_basename_path=sys.argv[4] #infarct_mask_basename_path_list[0] #os.path.join(niftifilenamedir,"Masks",mask_basename)
#
#
#
#
#         if os.path.exists(CSF_Mask_filename) and os.path.exists(niftifilename): # and os.path.exists(infarct_mask_basename_path) :
#             print("BOTH FILE EXISTS")
#
#             print('CSF_Mask_filename')
#             print(CSF_Mask_filename)
#             print('niftifilename')
#             print(niftifilename)
#
#
#             CSF_Mask_filename_fdata=nib.load(CSF_Mask_filename).get_fdata()
#             CSF_Mask_filename_fdata_June21_2023=nib.load(CSF_Mask_filename).get_fdata()
#             CSF_Mask_filename_fdata[Infarct_Mask_filename_June20_data>0]=0 #np.min(CSF_Mask_filename_fdata)
#             CSF_Mask_filename_fdata_June21_2023_np=resizeinto_512by512(CSF_Mask_filename_fdata_June21_2023)
#             CSF_Mask_filename_data_np=resizeinto_512by512(CSF_Mask_filename_fdata) ##nib.load(CSF_Mask_filename).get_fdata()) #nib.load(CSF_Mask_filename).get_fdata() #
#             CSF_Mask_filename_data_np[CSF_Mask_filename_data_np>1]=0
#             CSF_Mask_filename_fdata_June21_2023_np[CSF_Mask_filename_fdata_June21_2023_np>1]=0
#
#             ######################### added on July 15 2022 ##################################
#             #             print("56code added on July 15 2022")
#             if os.path.exists(sys.argv[4]):
#                 infarct_image_data_1=resizeinto_512by512(nib.load(sys.argv[4]).get_fdata())
#                 #                 print('np.max(infarct_image_data_1):{}'.format(np.max(infarct_image_data_1)))
#                 print('Filename:{}'.format(os.path.basename(niftifilename)))
#                 print('Number of voxels in CSF mask before infarct subtraction:{}'.format(len(CSF_Mask_filename_data_np[CSF_Mask_filename_data_np>0])))
#
#                 CSF_Mask_filename_data_np[infarct_image_data_1>0]=0
#                 print("code for subtraction:{}".format('CSF_Mask_data[infarct_data>0]=0'))
#
#                 print('Number of voxels in CSF mask after infarct subtraction:{}'.format(len(CSF_Mask_filename_data_np[CSF_Mask_filename_data_np>0])))
#             #                 print("58code added on July 15 2022")
#             ##################################################################################
#
#             #                print(np.max(CSF_Mask_filename_data_np))
#             filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata()) #nib.load(niftifilename).get_fdata() #
#             filename_bet_gray_data_np=contrast_stretch_np(resizeinto_512by512(nib.load(bet_filename_path).get_fdata()),1) #contrast_stretch_np(nib.load(bet_filename_path).get_fdata(),1) #
#             filename_gray_data_np=contrast_stretch_np(filename_gray_data_np,1) #exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
#             filename_gray_data_np_1=contrast_stretch_np(resizeinto_512by512(nib.load(grayfilename).get_fdata()),1)*255  #contrast_stretch_np(nib.load(grayfilename).get_fdata(),1)*255 ##np.uint8(filename_gray_data_np*255)
#             numpy_image=filename_gray_data_np #normalizeimage0to1(filename_gray_data_np)*255
#             filename_brain_data_np_minus_CSF=np.copy(filename_bet_gray_data_np)*255
#             #             filename_brain_data_np_minus_CSF[filename_bet_gray_data_np<np.max(filename_bet_gray_data_np)]=np.min(filename_brain_data_np_minus_CSF)
#             filename_brain_data_np_minus_CSF[CSF_Mask_filename_data_np>=np.max(CSF_Mask_filename_data_np)]=np.min(filename_brain_data_np_minus_CSF)
#             upper_slice_num=0
#             lower_slice_num=0
#             found_lower_slice=0
#             for slice_num_csf in range(CSF_Mask_filename_data_np.shape[2]):
#
#                 if found_lower_slice==0 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0:
#                     lower_slice_num=slice_num_csf
#                     found_lower_slice=1
#                 if found_lower_slice==1 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0 :
#                     upper_slice_num=slice_num_csf
#             this_slice_left_volume=0
#             this_slice_right_volume=0
#             for img_idx in range(numpy_image.shape[2]):
#                 if img_idx>0 and img_idx < numpy_image.shape[2]:
#
#                     method_name="REGIS"
#                     slice_number="{0:0=3d}".format(img_idx)
#                     filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
#                     this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
#                     #                        this_npyfile=os.path.join(npyfiledirectory,os.path.basename(niftifilename).split(".nii")[0]+str(img_idx)+npyfileextension)
#                     #                        print(this_npyfile)
#                     if os.path.exists(this_npyfile):
#                         print("YES FOUND BOTH FILES")
#                         print('latexfilename')
#                         print(latexfilename)
#                         calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
#                         x_points2=calculated_midline_points.item().get('x_axis') #,y_points2=points_on_line(extremepoints)
#                         # print(x_points2)
#                         y_points2=calculated_midline_points.item().get('y_axis')
#                         slice_3_layer= np.zeros([numpy_image.shape[0],numpy_image.shape[1],3])
#                         x_points2=x_points2[:,0]
#                         y_points2=y_points2[:,0]
#                         # print(this_npyfile)
#
#                         img_with_line=CSF_Mask_filename_data_np[:,:,img_idx]
#                         # print(np.max(img_with_line))
#                         img_with_line_nonzero_id = np.transpose(np.nonzero(img_with_line))
#                         thisimage=filename_gray_data_np_1[:,:,img_idx]
#                         current_left_num=0
#                         current_right_num=0
#                         slice_3_layer= np.zeros([img_with_line.shape[0],img_with_line.shape[1],3])
#                         slice_3_layer[:,:,0]= thisimage #imgray1
#                         slice_3_layer[:,:,1]= thisimage #imgray1
#                         slice_3_layer[:,:,2]= thisimage# imgray1
#
#                         Infarct_Mask_filename_June20_data_512_idx  = Infarct_Mask_filename_June20_data_512[:,:,img_idx]
#                         Infarct_Mask_filename_June20_data_512_idx[Infarct_Mask_filename_June20_data_512_idx>0]=1
#                         Infarct_Mask_filename_June20_data_512_idx[Infarct_Mask_filename_June20_data_512_idx<1]=0
#                         CSF_Mask_filename_data_np_idx=CSF_Mask_filename_fdata_June21_2023_np[:,:,img_idx]
#                         CSF_Mask_filename_data_np_idx[CSF_Mask_filename_data_np_idx>0]=1
#                         CSF_Mask_filename_data_np_idx[CSF_Mask_filename_data_np_idx<1]=0
#                         Infarct_Mask_filename_June20_data_512_idx=CSF_Mask_filename_data_np_idx*Infarct_Mask_filename_June20_data_512_idx
#                         # Infarct_Mask_filename_June20_data_512_idx[CSF_Mask_filename_data_np[:,:,img_idx]>0]=255
#                         # Infarct_Mask_filename_June20_data_512_idx[Infarct_Mask_filename_June20_data_512_idx[:,:,img_idx]<0.5 ]=255
#                         # Infarct_Mask_filename_June20_data_512_idx[(Infarct_Mask_filename_June20_data_512_idx[:,:,img_idx]<255) ]=0
#                         print("np.unique(CSF_Mask_filename_data_np)::{}".format(np.unique(CSF_Mask_filename_data_np)))
#                         # Infarct_Mask_filename_June20_data_512_idx[CSF_Mask_filename_data_np[:,:,img_idx]==np.min(CSF_Mask_filename_data_np)]=np.min(Infarct_Mask_filename_June20_data_512)
#                         Infarct_Mask_filename_June20_data_512_idx_flatten=Infarct_Mask_filename_June20_data_512_idx.flatten()
#                         # INFARCT_VOXELS_IN_CSF=INFARCT_VOXELS_IN_CSF+np.count_nonzero(Infarct_Mask_filename_June20_data_512_idx_flatten)
#
#                         slice_3_layer_brain= np.zeros([img_with_line.shape[0],img_with_line.shape[1],3])
#                         slice_3_layer_brain[:,:,0]= filename_brain_data_np_minus_CSF[:,:,img_idx] #imgray1
#                         slice_3_layer_brain[:,:,1]= filename_brain_data_np_minus_CSF[:,:,img_idx] #imgray1
#                         slice_3_layer_brain[:,:,2]= filename_brain_data_np_minus_CSF[:,:,img_idx] # imgray1
#                         # font
#                         font = cv2.FONT_HERSHEY_SIMPLEX
#
#                         # org
#                         org = (50, 50)
#
#                         # fontScale
#                         fontScale = 1
#
#                         # Blue color in BGR
#                         color = (0, 0, 255)
#
#                         # Line thickness of 2 px
#                         thickness = 2
#
#                         imagefilename_gray=os.path.basename(niftifilename).split(".nii")[0].replace(".","_")+"_" +str(slice_number)+"gray"
#                         slice_3_layer = cv2.putText(slice_3_layer,str(slice_number) , org, font,  fontScale, color, thickness, cv2.LINE_AA)
#                         #                         slice_3_layer_brain = cv2.putText(slice_3_layer,str(slice_number) , org, font,  fontScale, color, thickness, cv2.LINE_AA)
#                         cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename_gray +".png"),slice_3_layer)
#                         for non_zero_pixel in img_with_line_nonzero_id:
#                             xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
#                             if xx>0: ## RIGHT
#                                 slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                                 slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=255
#                                 slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=0
#                                 current_right_num = current_right_num + 1
#                             if xx<0: ## LEFT
#                                 slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                                 slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=0
#                                 slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=255
#                                 current_left_num = current_left_num + 1
#                         brainarea_with_nonzero_id = np.transpose(np.nonzero(filename_brain_data_np_minus_CSF[:,:,img_idx]))
#
#                         left_brain_voxel_count=0
#                         right_brain_voxel_count=0
#                         for non_zero_pixel in brainarea_with_nonzero_id:
#                             xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
#                             if xx>0: ## RIGHT
#                                 slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                                 slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],1]=255
#                                 slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],2]=0
#                                 right_brain_voxel_count = right_brain_voxel_count + 1
#                             if xx<0: ## LEFT
#                                 slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                                 slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],1]=0
#                                 slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],2]=255
#                                 left_brain_voxel_count = left_brain_voxel_count + 1
#
#
#                         lineThickness = 2
#
#                         this_slice_left_volume = current_left_num*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#                         this_slice_right_volume = current_right_num*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#
#                         this_slice_gray_left_volume = left_brain_voxel_count*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#                         this_slice_gray_right_volume = right_brain_voxel_count*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#                         this_slice_gray_left_volume=this_slice_gray_left_volume/1000
#                         this_slice_gray_right_volume=this_slice_gray_right_volume/1000
#
#                         #######################
#                         slice_3_layer[:,:,0][Infarct_Mask_filename_June20_data_512_idx>0]=255
#                         print("values of pixels")
#                         slice_3_layer[:,:,1][Infarct_Mask_filename_June20_data_512_idx>0]=0
#                         slice_3_layer[:,:,2][Infarct_Mask_filename_June20_data_512_idx>0]=0
#
#                         img_with_line1=cv2.line(slice_3_layer, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,0), 2)
#
#                         img_hemibrain_line1=cv2.line(slice_3_layer_brain, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,0), 2)
#                         slice_number="{0:0=3d}".format(img_idx)
#
#
#                         imagefilename=os.path.basename(niftifilename).split(".nii")[0].replace(".","_")+"_" +str(slice_number)
#                         imagefilename_infarct=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_infarct.png")
#                         image_infarct_details=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_infarct_details.png")
#                         image_infarct_noninfarct_histogram=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_infarct_noninfarct_histogram.png")
#                         image_left_right_brain=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_left_right_brain.png")
#
#
#                         cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"),img_with_line1)
#                         cv2.imwrite(image_left_right_brain,slice_3_layer_brain)
#                         nect_file_basename_forimagename=imagefilename
#
#                         ## get the mask image:
#                         this_slice_left_volume=this_slice_left_volume/1000
#                         this_slice_right_volume=this_slice_right_volume/1000
#
#                         ################################
#                         # upper_slice_num=0
#                         # lower_slice_num=0
#                         # found_lower_slice=0
#                         # for slice_num_csf in range(CSF_Mask_filename_data_np.shape[2]):
#
#                         #     if found_lower_slice==0 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0:
#                         #         lower_slice_num=slice_num_csf
#                         #         found_lower_slice=1
#                         #     if found_lower_slice==1 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0 :
#                         #         upper_slice_num=slice_num_csf
#                         ##########################
#
#                         image_list=[]
#                         print("lower_slice_num:{} and upper_slice_num:{}".format(lower_slice_num,upper_slice_num))
#                         if os.path.exists(sys.argv[4])  and int(slice_number) >=int(lower_slice_num) and int(slice_number)<=int(upper_slice_num) :
#                             latex_start_tableNc_noboundary(latexfilename,5)
#
#                             image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename_gray +".png"))
#                             image_list.append(image_left_right_brain)
#                             image_list.append(imagefilename_infarct)
#                             # image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,image_infarct_details))
#                             image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"))
#                             latex_insertimage_tableNc(latexfilename,image_list,len(image_list), caption="",imagescale=0.25, angle=90,space=0.51)
#                             latex_end_table2c(latexfilename)
#
#                         elif int(slice_number) >=int(lower_slice_num) and int(slice_number)<=int(upper_slice_num) :
#                             latex_start_tableNc_noboundary(latexfilename,2)
#
#                             image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename_gray +".png"))
#                             # image_list.append(imagefilename_infarct)
#                             # image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,image_infarct_details))
#                             image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"))
#                             latex_insertimage_tableNc(latexfilename,image_list,2, caption="",imagescale=0.2, angle=90,space=0.51)
#
#                             latex_end_table2c(latexfilename)
#
#                         left_pixels_num=left_pixels_num+this_slice_left_volume
#                         right_pixels_num=right_pixels_num+this_slice_right_volume
#
#                         left_brain_volume=left_brain_volume + this_slice_gray_left_volume
#                         right_brain_volume=right_brain_volume + this_slice_gray_right_volume
#             image_array=np.asarray(filename_bet_gray_data_np)
#             print("image_array MINIMUM")
#             print(np.min(image_array))
#             BET_VOLUME = (image_array > 0).sum()*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4])) / 1000
#             # CSF_RATIO=left_pixels_num/right_pixels_num
#             # if left_pixels_num > right_pixels_num :
#             #     CSF_RATIO=right_pixels_num/left_pixels_num
#                 # thisfilebasename=os.path.basename(grayfilename).split("_levelset")[0]
#             # row2 = [os.path.basename(niftifilename).split(".nii")[0] , str(left_pixels_num), str(right_pixels_num),str(left_pixels_num+right_pixels_num), infarct_side,NWU, infarct_pixels_number, infarct_pixels_density, nonfarct_pixels_number,noninfarct_pixels_density,overall_infarct_vol,overall_non_infarct_vol,str(BET_VOLUME),str(CSF_RATIO),str(left_brain_volume),str(right_brain_volume),str(lower_thresh)+"to"+ str(upper_thresh),str(lower_thresh_normal) +"to" +str(upper_thresh_normal)]
#             left_pixels_num_after_edema_subt,right_pixels_num_after_edema_subt,CSF_RATIO_after_edema_subt=csf_ratio_after_subtractionof_edema(niftifilename,bet_filename_path,grayfilename,Infarct_Mask_filename_June20_data,CSF_Mask_filename_data_np,npyfiledirectory,latexfilename,SLICE_OUTPUT_DIRECTORY)
#             left_pixels_num=left_pixels_num_after_edema_subt
#             right_pixels_num=right_pixels_num_after_edema_subt
#             CSF_RATIO=CSF_RATIO_after_edema_subt
#             row2 = [thisfilebasename , str(left_pixels_num), str(right_pixels_num),str(left_pixels_num+right_pixels_num), infarct_side,NWU, infarct_pixels_number, infarct_pixels_density, nonfarct_pixels_number,noninfarct_pixels_density,overall_infarct_vol,overall_non_infarct_vol,str(BET_VOLUME),str(CSF_RATIO),str(left_brain_volume),str(right_brain_volume),str(lower_thresh)+"to"+ str(upper_thresh),str(lower_thresh_normal) +"to" +str(upper_thresh_normal)]
#
#             values_in_col=np.array(row2)
#
#
#             with open(csvfile_with_vol_total, 'a') as f1:
#                 writer = csv.writer(f1)
#                 writer.writerow(row2)
#             this_nii_filename_list=[]
#             # this_nii_filename_list.append(os.path.basename(niftifilename).split(".nii")[0]) #thisfilebasename
#             this_nii_filename_list.append(thisfilebasename)
#             this_nii_filename_df=pd.DataFrame(this_nii_filename_list)
#             this_nii_filename_df.columns=['FILENAME']
#
#             latex_start_tableNc_noboundary(latexfilename,1)
#             latex_insert_line_nodek(latexfilename,text=this_nii_filename_df.to_latex(index=False))
#             latex_end_table2c(latexfilename)
#
#             #             latex_insert_line_nodek(latexfilename,"\\newpage")
#             #             latex_insert_line_nodate(latexfilename,"\\texttt{\\detokenize{" + os.path.basename(niftifilename).split(".nii")[0] + "}}")
#
#             values_in_table=[]
#
#             #             text1=[]
#             #             text1.append(" Regions ")
#             #             text1.append("Volume  (ml)")
#             #             latex_start_tableNc(latexfilename,2)
#             #             latex_inserttext_tableNc(latexfilename,text1,2,space=-1.4)
#
#             for x in range(0,col_names.shape[0]):
#                 #                 text1=[]
#                 values_in_table.append([(str(col_names[x])).replace("_"," "),(str(values_in_col[x])).replace("_","")])
#             #                 text1.append((str(col_names[x])).replace("_"," "))
#             #                 text1.append((str(values_in_col[x])).replace("_",""))
#
#             #                 latex_inserttext_tableNc(latexfilename,text1,2,space=-1.4)
#             #             latex_end_table2c(latexfilename)
#             values_in_table.pop(0)
#             values_in_table_df=pd.DataFrame(values_in_table)
#             values_in_table_df.columns=[" Regions ","Volume  (ml)"]
#             latex_start_tableNc_noboundary(latexfilename,1)
#             latex_insert_line_nodek(latexfilename,text=values_in_table_df.to_latex(index=False))
#             latex_end_table2c(latexfilename)
#         latex_end(latexfilename)
#         remove_few_columns(csvfile_with_vol_total,["INFARCT VOX_NUMBERS", "INFARCT DENSITY", "NON INFARCT VOX_NUMBERS"])

# def csf_ratio_after_subtractionof_edema(niftifilename,bet_filename_path,grayfilename,Infarct_Mask_filename_June20_data,CSF_Mask_filename_data_np,npyfiledirectory,latexfilename,SLICE_OUTPUT_DIRECTORY):
#     EDEMA_VOXELS_IN_CSF=0
#     left_pixels_num=0
#     right_pixels_num=0
#     filename_gray_data_np=resizeinto_512by512(nib.load(niftifilename).get_fdata()) #nib.load(niftifilename).get_fdata() #
#     filename_bet_gray_data_np=contrast_stretch_np(resizeinto_512by512(nib.load(bet_filename_path).get_fdata()),1) #contrast_stretch_np(nib.load(bet_filename_path).get_fdata(),1) #
#     filename_gray_data_np=contrast_stretch_np(filename_gray_data_np,1) #exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200))
#     filename_gray_data_np_1=contrast_stretch_np(resizeinto_512by512(nib.load(grayfilename).get_fdata()),1)*255  #contrast_stretch_np(nib.load(grayfilename).get_fdata(),1)*255 ##np.uint8(filename_gray_data_np*255)
#     Infarct_Mask_filename_June20_data_512=resizeinto_512by512(Infarct_Mask_filename_June20_data)
#     # filename_gray_data_np[Infarct_Mask_filename_June20_data_512>np.min(Infarct_Mask_filename_June20_data_512)]=np.min(filename_gray_data_np)
#     # filename_gray_data_np_1[Infarct_Mask_filename_June20_data_512>np.min(Infarct_Mask_filename_June20_data_512)]=np.min(filename_gray_data_np_1)
#     CSF_Mask_filename_data_np[Infarct_Mask_filename_June20_data_512>np.min(Infarct_Mask_filename_June20_data_512)]=np.min(CSF_Mask_filename_data_np)
#     numpy_image=filename_gray_data_np #normalizeimage0to1(filename_gray_data_np)*255
#     filename_brain_data_np_minus_CSF=np.copy(filename_bet_gray_data_np)*255
#     #             filename_brain_data_np_minus_CSF[filename_bet_gray_data_np<np.max(filename_bet_gray_data_np)]=np.min(filename_brain_data_np_minus_CSF)
#     filename_brain_data_np_minus_CSF[CSF_Mask_filename_data_np>=np.max(CSF_Mask_filename_data_np)]=np.min(filename_brain_data_np_minus_CSF)
#     upper_slice_num=0
#     lower_slice_num=0
#     found_lower_slice=0
#     for slice_num_csf in range(CSF_Mask_filename_data_np.shape[2]):
#
#         if found_lower_slice==0 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0:
#             lower_slice_num=slice_num_csf
#             found_lower_slice=1
#         if found_lower_slice==1 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0 :
#             upper_slice_num=slice_num_csf
#     this_slice_left_volume=0
#     this_slice_right_volume=0
#     for img_idx in range(numpy_image.shape[2]):
#         if img_idx>0 and img_idx < numpy_image.shape[2]:
#
#             method_name="REGIS"
#             slice_number="{0:0=3d}".format(img_idx)
#             filename_tosave=re.sub('[^a-zA-Z0-9 \n\_]', '', os.path.basename(niftifilename).split(".nii")[0])
#             this_npyfile=os.path.join(npyfiledirectory,filename_tosave+method_name+str(slice_number)+  ".npy")
#             #                        this_npyfile=os.path.join(npyfiledirectory,os.path.basename(niftifilename).split(".nii")[0]+str(img_idx)+npyfileextension)
#             #                        print(this_npyfile)
#             if os.path.exists(this_npyfile):
#                 print("YES FOUND BOTH FILES")
#                 print('latexfilename')
#                 print(latexfilename)
#                 calculated_midline_points=np.load(this_npyfile,allow_pickle=True)
#                 x_points2=calculated_midline_points.item().get('x_axis') #,y_points2=points_on_line(extremepoints)
#                 # print(x_points2)
#                 y_points2=calculated_midline_points.item().get('y_axis')
#                 slice_3_layer= np.zeros([numpy_image.shape[0],numpy_image.shape[1],3])
#                 x_points2=x_points2[:,0]
#                 y_points2=y_points2[:,0]
#                 # print(this_npyfile)
#
#                 img_with_line=CSF_Mask_filename_data_np[:,:,img_idx]
#                 # print(np.max(img_with_line))
#                 img_with_line_nonzero_id = np.transpose(np.nonzero(img_with_line))
#                 thisimage=filename_gray_data_np_1[:,:,img_idx]
#                 current_left_num=0
#                 current_right_num=0
#                 slice_3_layer= np.zeros([img_with_line.shape[0],img_with_line.shape[1],3])
#                 slice_3_layer[:,:,0]= thisimage #imgray1
#                 slice_3_layer[:,:,1]= thisimage #imgray1
#                 slice_3_layer[:,:,2]= thisimage  # imgray1
#                 Infarct_Mask_filename_June20_data_512_idx  = Infarct_Mask_filename_June20_data_512[:,:,img_idx]
#                 print("np.unique(CSF_Mask_filename_data_np)::{}".format(np.unique(CSF_Mask_filename_data_np)))
#                 Infarct_Mask_filename_June20_data_512_idx[CSF_Mask_filename_data_np[:,:,img_idx]==np.min(CSF_Mask_filename_data_np)]=np.min(Infarct_Mask_filename_June20_data_512)
#                 Infarct_Mask_filename_June20_data_512_idx_flatten=Infarct_Mask_filename_June20_data_512_idx.flatten()
#                 EDEMA_VOXELS_IN_CSF=EDEMA_VOXELS_IN_CSF+np.count_nonzero(Infarct_Mask_filename_June20_data_512_idx_flatten)
#
#
#                 slice_3_layer_brain= np.zeros([img_with_line.shape[0],img_with_line.shape[1],3])
#                 slice_3_layer_brain[:,:,0]= filename_brain_data_np_minus_CSF[:,:,img_idx] #imgray1
#                 slice_3_layer_brain[:,:,1]= filename_brain_data_np_minus_CSF[:,:,img_idx] #imgray1
#                 slice_3_layer_brain[:,:,2]= filename_brain_data_np_minus_CSF[:,:,img_idx] # imgray1
#                 # font
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#
#                 # org
#                 org = (50, 50)
#
#                 # fontScale
#                 fontScale = 1
#
#                 # Blue color in BGR
#                 color = (0, 0, 255)
#
#                 # Line thickness of 2 px
#                 thickness = 2
#
#                 imagefilename_gray=os.path.basename(niftifilename).split(".nii")[0].replace(".","_")+"_" +str(slice_number)+"gray"
#                 slice_3_layer = cv2.putText(slice_3_layer,str(slice_number) , org, font,  fontScale, color, thickness, cv2.LINE_AA)
#                 #                         slice_3_layer_brain = cv2.putText(slice_3_layer,str(slice_number) , org, font,  fontScale, color, thickness, cv2.LINE_AA)
#                 # cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename_gray +".png"),slice_3_layer)
#                 for non_zero_pixel in img_with_line_nonzero_id:
#                     xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
#                     if xx>0: ## RIGHT
#                         slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                         slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=255
#                         slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=0
#                         current_right_num = current_right_num + 1
#                     if xx<0: ## LEFT
#                         slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                         slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],1]=0
#                         slice_3_layer[non_zero_pixel[0],non_zero_pixel[1],2]=255
#                         current_left_num = current_left_num + 1
#                 brainarea_with_nonzero_id = np.transpose(np.nonzero(filename_brain_data_np_minus_CSF[:,:,img_idx]))
#
#                 left_brain_voxel_count=0
#                 right_brain_voxel_count=0
#                 for non_zero_pixel in brainarea_with_nonzero_id:
#                     xx=whichsideofline((int(y_points2[511]),int(x_points2[511])),(int(y_points2[0]),int(x_points2[0])) ,non_zero_pixel)
#                     if xx>0: ## RIGHT
#                         slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                         slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],1]=255
#                         slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],2]=0
#                         right_brain_voxel_count = right_brain_voxel_count + 1
#                     if xx<0: ## LEFT
#                         slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],0]=0
#                         slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],1]=0
#                         slice_3_layer_brain[non_zero_pixel[0],non_zero_pixel[1],2]=255
#                         left_brain_voxel_count = left_brain_voxel_count + 1
#
#
#                 lineThickness = 2
#
#                 this_slice_left_volume = current_left_num*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#                 this_slice_right_volume = current_right_num*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#
#                 this_slice_gray_left_volume = left_brain_voxel_count*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#                 this_slice_gray_right_volume = right_brain_voxel_count*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4]))
#                 this_slice_gray_left_volume=this_slice_gray_left_volume/1000
#                 this_slice_gray_right_volume=this_slice_gray_right_volume/1000
#
#                 img_with_line1=cv2.line(slice_3_layer, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,0), 2)
#
#                 img_hemibrain_line1=cv2.line(slice_3_layer_brain, ( int(x_points2[0]),int(y_points2[0])),(int(x_points2[511]),int(y_points2[511])), (0,255,0), 2)
#                 slice_number="{0:0=3d}".format(img_idx)
#
#                 slice_3_layer[:,:,0][Infarct_Mask_filename_June20_data_512_idx>0]=255
#                 slice_3_layer[:,:,1][Infarct_Mask_filename_June20_data_512_idx>0]=0
#                 slice_3_layer[:,:,2][Infarct_Mask_filename_June20_data_512_idx>0]=0
#                 imagefilename=os.path.basename(niftifilename).split(".nii")[0].replace(".","_")+"_" +str(slice_number)
#
#                 # imagefilename_ICH=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_ICH.png")
#                 # imagename_class2=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_class2.png")
#                 # imagename_class1=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_class1.png")
#                 # image_ICH_details=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_ICH_details.png")
#                 # image_ICH_nonICH_histogram=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_ICH_nonICH_histogram.png")
#                 # image_left_right_brain=os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +"_left_right_brain.png")
#
#
#                 # cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"),img_with_line1)
#                 # cv2.imwrite(image_left_right_brain,slice_3_layer_brain)
#                 # nect_file_basename_forimagename=imagefilename
#
#                 ## get the mask image:
#                 this_slice_left_volume=this_slice_left_volume/1000
#                 this_slice_right_volume=this_slice_right_volume/1000
#
#                 ################################
#                 # upper_slice_num=0
#                 # lower_slice_num=0
#                 # found_lower_slice=0
#                 # for slice_num_csf in range(CSF_Mask_filename_data_np.shape[2]):
#
#                 #     if found_lower_slice==0 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0:
#                 #         lower_slice_num=slice_num_csf
#                 #         found_lower_slice=1
#                 #     if found_lower_slice==1 and np.sum(CSF_Mask_filename_data_np[:,:,slice_num_csf]) >0 :
#                 #         upper_slice_num=slice_num_csf
#                 ##########################
#
#                 image_list=[]
#                 # print("lower_slice_num:{} and upper_slice_num:{}".format(lower_slice_num,upper_slice_num))
#                 # if os.path.exists(sys.argv[4])  and int(slice_number) >=int(lower_slice_num) and int(slice_number)<=int(upper_slice_num) :
#                 #     latex_start_tableNc_noboundary(latexfilename,6)
#                 #
#                 #     image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename_gray +".png"))
#                 #     image_list.append(image_left_right_brain)
#                 #     # image_list.append(imagefilename_ICH)
#                 #     image_list.append(imagename_class1)
#                 #     image_list.append(imagename_class2)
#                 #
#                 #     # image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,image_ICH_details))
#                 #     image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"))
#                 #     latex_insertimage_tableNc(latexfilename,image_list,len(image_list), caption="",imagescale=.2, angle=90,space=0.51)
#                 #     latex_end_table2c(latexfilename)
#                 #
#                 # elif int(slice_number) >=int(lower_slice_num) and int(slice_number)<=int(upper_slice_num) :
#                 #     latex_start_tableNc_noboundary(latexfilename,2)
#                 #
#                 #     image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename_gray +".png"))
#                 #     # image_list.append(imagefilename_ICH)
#                 #     # image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,image_ICH_details))
#                 #     image_list.append(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"))
#                 #     latex_insertimage_tableNc(latexfilename,image_list,2, caption="",imagescale=0.2, angle=90,space=0.51)
#                 #
#                 #     latex_end_table2c(latexfilename)
#
#                 left_pixels_num=left_pixels_num+this_slice_left_volume
#                 right_pixels_num=right_pixels_num+this_slice_right_volume
#
#                 # left_brain_volume=left_brain_volume + this_slice_gray_left_volume
#                 # right_brain_volume=right_brain_volume + this_slice_gray_right_volume
#     image_array=np.asarray(filename_bet_gray_data_np)
#     print("image_array MINIMUM")
#     print(np.min(image_array))
#     # EDEMA_VOXELS_IN_CSF_TOTAL_VOL=EDEMA_VOXELS_IN_CSF*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4])) / 1000
#     # BET_VOLUME = (image_array > 0).sum()*np.prod(np.array(nib.load(niftifilename).header["pixdim"][1:4])) / 1000
#     CSF_RATIO="NOT DEFINED"
#     if right_pixels_num != 0 and left_pixels_num !=0:
#         CSF_RATIO=left_pixels_num/right_pixels_num
#         if left_pixels_num > right_pixels_num :
#             CSF_RATIO=right_pixels_num/left_pixels_num
#     return left_pixels_num,right_pixels_num,CSF_RATIO

def call_mirror_a_mask(args):
    mirror_a_mask(args)
    return 1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stuff', nargs='+')
    args = parser.parse_args()
    name_of_the_function=args.stuff[0]
    return_value=0
    if name_of_the_function == "call_divide_a_mask_into_left_right_submasks":
        return_value=call_divide_a_mask_into_left_right_submasks(args)
    if name_of_the_function == "call_ratio_left_right":
        return_value=call_ratio_left_right(args)
    if name_of_the_function == "call_calculate_volume":
        return_value=call_calculate_volume(args)
    if name_of_the_function == "call_masks_subtraction":
        return_value=call_masks_subtraction(args)
    if name_of_the_function == "call_masks_on_grayscale_colored":
        return_value=call_masks_on_grayscale_colored(args)
    if name_of_the_function == "call_combine_csv_horizontally":
        return_value=call_combine_csv_horizontally(args)
    if name_of_the_function == "call_calculate_volume_mask_from_yasheng":
        return_value=call_calculate_volume_mask_from_yasheng(args)
    if name_of_the_function == "call_slice_num_to_csv":
        return_value=call_slice_num_to_csv(args)
    if name_of_the_function == "call_insert_one_col_with_colname_colidx":
        return_value=call_insert_one_col_with_colname_colidx(args)
    if name_of_the_function == "call_mirror_a_mask":
        return_value=call_mirror_a_mask(args)
    if name_of_the_function == "call_infarct_and_reflectedinfarct_related_parameters":
        return_value=call_infarct_and_reflectedinfarct_related_parameters(args)
    if name_of_the_function == "call_csf_related_parameters":
        return_value=call_csf_related_parameters(args)
    if name_of_the_function == "call_divide_a_mask_into_left_right_submasks_v1":
        return_value=call_divide_a_mask_into_left_right_submasks_v1(args)

    if name_of_the_function == "call_bet_related_parameters":
        return_value=call_bet_related_parameters(args)
    if name_of_the_function == "call_side_of_lesion":
        return_value=call_side_of_lesion(args)
    if name_of_the_function == "call_begin_csvfile_with_session_name":
        return_value=call_begin_csvfile_with_session_name(args)
    if name_of_the_function == "call_original_to_512x512":
        return_value=call_original_to_512x512(args)
    if "call" not in name_of_the_function:
        globals()[args.stuff[0]](args)
    return return_value
if __name__ == '__main__':
    main()