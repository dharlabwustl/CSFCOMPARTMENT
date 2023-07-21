# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:26:44 2019

@author: atul
"""
import os.path
import sys,argparse,inspect

from utilities_simple_trimmed import *
#_compartments

from  Segmentation_Ventricle_Sulcus_CSF_1_Dec15_2019 import * 
colors = vtk.vtkNamedColors()
renderer = vtk.vtkRenderer()

ANAYLYSIS_TYPE="CSF_COMPARTMENT_VEN_SUL_AB"
project_folder="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment"


NECT_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/DATA/NECT"
NECT_BET_directory_name_parent="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/DATA/BET"
RESULT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/RESULTS"
SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/RESULTS/IMAGES"
grayscalefilextension="_levelset.nii.gz"


dict_for_csv=[]
dict_for_csv_perslice=[]
count =0


grayscale_suffix="_levelset"
masksuffix="_final_seg" #
betsuffix="_levelset_bet"
def csf_compartments(filename_gray,filename_mask,filename_bet):
    returnvalue=0
    try:

        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_v1(filename_gray,filename_mask,filename_bet)

        print("I SUCCEED AT ::{}".format(inspect.stack()[0][3]))
        returnvalue=1
    except:
        print("I FAILED AT ::{}".format(inspect.stack()[0][3]))
        pass
    return returnvalue
def call_csf_compartments(args):
    returnvalue=0

    return returnvalue

for dirname in os.listdir(NECT_directory_name_parent): 
    if count <1 and  os.path.isdir(os.path.join(NECT_directory_name_parent,dirname)): # and dirname=="WUSTL_664"        

        NECT_directory_name=os.path.join(NECT_directory_name_parent,dirname) 
        NECT_filenames=glob.glob(NECT_directory_name+"/*" + grayscalefilextension) 
        for   NECT_filename in NECT_filenames:
            if count <1 and "Krak" in os.path.basename(NECT_filename) :
                nect_file_basename=os.path.basename(NECT_filename)
                nect_file_basename_forimagename=nect_file_basename.split('.')[0]
                NECT_HET_filename=os.path.join(NECT_BET_directory_name_parent,dirname,nect_file_basename[:-16]+"_levelset_bet.nii.gz" )
                CSF_Mask_filename=os.path.join(NECT_directory_name_parent,dirname,nect_file_basename[:-16]+"_final_seg.nii.gz" )
                RAW_DATA_FOLDER=NECT_directory_name
                # command= "cp  " + NECT_HET_filename+  "  /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/"
                # subprocess.call(command,shell=True)
                # command= "cp  " + CSF_Mask_filename+  "  /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/"
                # subprocess.call(command,shell=True)
                # command= "cp  " + NECT_filename+  "  /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/"
                # subprocess.call(command,shell=True)
                each_unique_names_file_pattern=dirname
                filename_gray = NECT_filename
                filename_mask = CSF_Mask_filename
                filename_bet = NECT_HET_filename
                csf_compartments(filename_gray,filename_mask,filename_bet)
                subprocess.call("echo " + "NECT_filename AT ::{}  >> error.txt".format(NECT_filename) ,shell=True )
                count=count+1
#
#                 print("filename_gray")
#                 print(filename_gray)
#                 sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones_v1(latexfilename,SLICE_OUTPUT_DIRECTORY,filename_gray,filename_mask,filename_bet)
#                 latex_start_table2c(latexfilename)
#                 latex_inserttext_table2c(latexfilename,text1='SulciVol:', text2=str(sulci_vol))
#                 latex_insert_line(latexfilename,text='\\\\')
#                 latex_inserttext_table2c(latexfilename,text1='VentricleVol:', text2=str(ventricle_vol))
#                 latex_insert_line(latexfilename,text="\\\\")
#                 latex_inserttext_table2c(latexfilename,text1='SulciVolAboveVent:', text2=str(sulci_vol_above_vent))
#                 latex_insert_line(latexfilename,text="\\\\")
#                 latex_inserttext_table2c(latexfilename,text1='SulciVolBelowVent:', text2=str(sulci_vol_below_vent))
#                 latex_insert_line(latexfilename,text="\\\\")
#                 latex_inserttext_table2c(latexfilename,text1='SulciVolAtVent:', text2=str(sulci_vol_at_vent))
#                 latex_end_table2c(latexfilename)
#
#                 this_dict={"Subject": nect_file_basename[:-7],"Sulci_VolTotal":sulci_vol,"Ventricles_Vol":ventricle_vol,"Sulci_VolL":leftcountsul,"Sulci_VolR":rightcountsul,"Ventricles_VolL":leftcountven,"Ventricles_VolR":rightcountven,"sulci_vol_above_vent": sulci_vol_above_vent,"sulci_vol_below_vent" :sulci_vol_below_vent,"sulci_vol_at_vent":sulci_vol_at_vent}
#                 dict_for_csv.append(this_dict)
#                 count=count+1
#
# csv_filename=os.path.join(RESULT_DIRECTORY,ANAYLYSIS_TYPE)
# csvfile_with_vol=csv_filename+'.csv'
# csv_columns=['Subject','Sulci_VolTotal','Ventricles_Vol','Sulci_VolL','Sulci_VolR','Ventricles_VolL','Ventricles_VolR','sulci_vol_above_vent','sulci_vol_below_vent','sulci_vol_at_vent']
#
# write_csv(csvfile_with_vol,csv_columns,dict_for_csv)
# latex_end(latexfilename)
# latex_file_build(latexfilename)
# print("latexfilename::{}".format(latexfilename))
def main():
    print("WO ZAI ::{}".format("main"))
    parser = argparse.ArgumentParser()
    parser.add_argument('stuff', nargs='+')
    args = parser.parse_args()
    name_of_the_function=args.stuff[0]
    return_value=0
    if name_of_the_function == "call_csf_compartments":
        return_value=call_csf_compartments(args)

if __name__ == '__main__':
    main()