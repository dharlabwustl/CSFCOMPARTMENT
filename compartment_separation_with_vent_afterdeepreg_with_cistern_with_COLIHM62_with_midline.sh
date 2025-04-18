#!/bin/bash
echo " I AM AT COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN.sh"
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
#zoneV_min_z=${5}
#zoneV_max_z=${6}
sessionID=${1}
working_dir=/workinginput
output_directory=/workingoutput

final_output_directory=/outputinsidedocker

run_IML() {
  this_filename=${1}
  this_betfilename=${2}
  #  this_csfmaskfilename=${3}
  #  this_infarctmaskfilename=${4}
  echo "BET USING LEVELSET MASK"

  /software/bet_withlevelset.sh $this_filename ${this_betfilename} #${output_directory} #Helsinki2000_1019_10132014_1048_Head_2.0_ax_Tilt_1_levelset # ${3} # Helsinki2000_702_12172013_2318_Head_2.0_ax_levelset.nii.gz #${3} # $6 $7 $8 $9 ${10}

  echo "bet_withlevelset successful" >${output_directory}/success.txt
  this_filename_brain=${this_filename%.nii*}_brain_f.nii.gz
  # cp ${this_filename_brain} ${output_directory}/ #  ${final_output_directory}/
  echo "LINEAR REGISTRATION TO TEMPLATE"
  mat_file_num=$(ls ${output_directory}/*.mat | wc -l)
  if [[ ${mat_file_num} -gt 1 ]]; then
    echo "MAT FILES PRESENT"
    #    /software/linear_rigid_registration_onlytrasnformwith_matfile.sh
    /software/linear_rigid_registration_onlytrasnformwith_matfile.sh ${this_filename_brain}
  else
    /software/linear_rigid_registration.sh ${this_filename_brain} #${templatefilename} #$3 ${6} WUSTL_233_11122015_0840__levelset_brain_f.nii.gz
    /software/linear_rigid_registration_onlytrasnformwith_matfile.sh ${this_filename_brain}
    echo "linear_rigid_registration successful" >>${output_directory}/success.txt
  fi

  echo "RUNNING IML FSL PART"
  /software/ideal_midline_fslpart.sh ${this_filename} # ${templatefilename} ${mask_on_template}  #$9 #${10} #$8
  echo "ideal_midline_fslpart successful" >>${output_directory}/success.txt

  echo "RUNNING IML PYTHON PART"

  /software/ideal_midline_pythonpart.sh ${this_filename} #${templatefilename}  #$3 #$8 $9 ${10}
  echo "ideal_midline_pythonpart successful" >>${output_directory}/success.txt
  /software/ideal_midline_pythonpart_V2.sh ${this_filename} #${templatefilename}  #$3 #$8 $9 ${10}
  #    echo "RUNNING NWU AND CSF VOLUME CALCULATION "
  #
  #  /software/nwu_csf_volume.sh ${this_filename} ${this_betfilename} ${this_csfmaskfilename} ${this_infarctmaskfilename} ${lower_threshold} ${upper_threshold}
  #  echo "nwu_csf_volume successful" >>${output_directory}/success.txt
  #  thisfile_basename=$(basename $this_filename)
  #  # for texfile in $(/usr/lib/fsl/5.0/remove_ext ${output_directory}/$thisfile_basename)*.tex ;
  #  for texfile in ${output_directory}/*.tex; do
  #    pdflatex -halt-on-error -interaction=nonstopmode -output-directory=${output_directory} $texfile ##${output_directory}/$(/usr/lib/fsl/5.0/remove_ext $this_filename)*.tex
  #    rm ${output_directory}/*.aux
  #    rm ${output_directory}/*.log
  #  done
  #
  #  for filetocopy in $(/usr/lib/fsl/5.0/remove_ext ${output_directory}/$thisfile_basename)*_brain_f.nii.gz; do
  #    cp ${filetocopy} ${final_output_directory}/
  #  done
  #

  #
  #  for filetocopy in ${output_directory}/*.pdf; do
  #    cp ${filetocopy} ${final_output_directory}/
  #  done
  #  for filetocopy in ${output_directory}/*.csv; do
  #    cp ${filetocopy} ${final_output_directory}/
  #  done

}



function midlineonly_each_scan() {
  local niftifilename_ext=${1}

  eachfile_basename_noext=''
  originalfile_basename=''
  original_ct_file=''
  #  for eachfile in ${working_dir}/*.nii*; do
  for eachfile in ${working_dir_1}/*.nii*; do
    if [[ ${eachfile} != *"levelset"* ]]; then
      # testmystring does not contain c0

      original_ct_file=${eachfile}
      eachfile_basename=$(basename ${eachfile})
      originalfile_basename=${eachfile_basename}
      eachfile_basename_noext=${eachfile_basename%.nii*}

      ############## files basename ##################################
      grayfilename=${eachfile_basename_noext}_resaved_levelset.nii
      if [[ "$eachfile_basename" == *".nii.gz"* ]]; then #"$STR" == *"$SUB"*
        grayfilename=${eachfile_basename_noext}_resaved_levelset.nii.gz
      fi
      betfilename=${eachfile_basename_noext}_resaved_levelset_bet.nii.gz
      #    csffilename=${eachfile_basename_noext}_resaved_csf_unet.nii.gz
      #    infarctfilename=${eachfile_basename_noext}_resaved_infarct_auto_removesmall.nii.gz
      ################################################
      ############## copy those files to the docker image ##################################
      cp ${working_dir}/${betfilename} ${output_directory}/
      #    cp ${working_dir}/${csffilename} ${output_directory}/
      #    cp ${working_dir}/${infarctfilename} ${output_directory}/
      ####################################################################################
      source /software/bash_functions_forhost.sh

      cp ${original_ct_file} ${output_directory}/${grayfilename}
      grayimage=${output_directory}/${grayfilename} #${gray_output_subdir}/${eachfile_basename_noext}_resaved_levelset.nii
      ###########################################################################

      #    #### originalfiel: .nii
      #    #### betfile: *bet.nii.gz
      #
      #    # original_ct_file=$original_CT_directory_names/
      #    levelset_infarct_mask_file=${output_directory}/${infarctfilename}
      #    echo "levelset_infarct_mask_file:${levelset_infarct_mask_file}"
      #    ## preprocessing infarct mask:
      #    python3 -c "
      #import sys ;
      #sys.path.append('/software/') ;
      #from utilities_simple_trimmed import * ;  levelset2originalRF_new_flip()" "${original_ct_file}" "${levelset_infarct_mask_file}" "${output_directory}"

      ## preprocessing bet mask:
      levelset_bet_mask_file=${output_directory}/${betfilename}
      echo "levelset_bet_mask_file:${levelset_bet_mask_file}"
      python3 -c "

import sys ;
sys.path.append('/software/') ;
from utilities_simple_trimmed import * ;  levelset2originalRF_new_flip()" "${original_ct_file}" "${levelset_bet_mask_file}" "${output_directory}"

      #    #### preprocessing csf mask:
      #    levelset_csf_mask_file=${output_directory}/${csffilename}
      #    echo "levelset_csf_mask_file:${levelset_csf_mask_file}"
      #    python3 -c "
      #import sys ;
      #sys.path.append('/software/') ;
      #from utilities_simple_trimmed import * ;   levelset2originalRF_new_flip()" "${original_ct_file}" "${levelset_csf_mask_file}" "${output_directory}"

      #    lower_threshold=0
      #    upper_threshold=20
      #    templatefilename=scct_strippedResampled1.nii.gz
      #    mask_on_template=midlinecssfResampled1.nii.gz

      x=$grayimage
      bet_mask_filename=${output_directory}/${betfilename}
      #    infarct_mask_filename=${output_directory}/${infarctfilename}
      #    csf_mask_filename=${output_directory}/${csffilename}
      run_IML $x ${bet_mask_filename} #${csf_mask_filename} ${infarct_mask_filename}
    fi
  done

  # for f in ${output_directory}/*; do
  #     # if [ -d "$f" ]; then
  #         # $f is a directory
  #         rm -r $f
  #     # fi
  # done

}

function call_get_resourcefiles_metadata_saveascsv_args() {

local resource_dir=${2}   #"NIFTI"
local output_csvfile=${4} #{array[1]}

local URI=${1} #{array[0]}
#  local file_ext=${5}
#  local output_csvfile=${output_csvfile%.*}${resource_dir}.csv

local final_output_directory=${3}
local call_download_files_in_a_resource_in_a_session_arguments=('call_get_resourcefiles_metadata_saveascsv_args' ${URI} ${resource_dir} ${final_output_directory} ${output_csvfile})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_files_in_a_resource_in_a_session_arguments[@]}")
echo " I AM AT call_get_resourcefiles_metadata_saveascsv_args"

}
echo " I AM RUNNING "
################ DOWNLOAD MASKS ###############################
## METADATA in the MASK directory
### FIND SESSION SCAN ID:


URI=/data/experiments/${sessionID}
resource_dir="NIFTI_LOCATION"
output_csvfile=${sessionID}_SCANSELECTION_METADATA.csv
call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}

while IFS="," read -ra array; do
  url=${array[6]}
  echo "${url}"
  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url} $(basename ${url} ) ${working_dir})
  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")

##    for niftifile_csvfilename in ${working_dir}/*NIFTILOCATION.csv; do
#niftifile_csvfilename=$(ls ${working_dir}/*NIFTILOCATION.csv)
#while IFS=',' read -ra array5; do
#scanID=${array5[2]}
#echo sessionId::${sessionID}
#echo scanId::${scanID}
#done < <(tail -n +2 "${niftifile_csvfilename}")
done < <(tail -n +2 "${working_dir}/${output_csvfile}")
#dir_to_save=${working_dir}
#greyfile="NONE" ##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset.nii.gz'
#betfile="NONE"  ##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset_bet.nii.gz'
#csffile="NONE"  ##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_final_seg.nii.gz'
#while IFS=',' read -ra array; do
##xx=0
##
###if [ ${array[1]} == "SNIPR01_E00894" ]  ; then
##  echo "${array[6]}"
#url=${array[6]}
#filename=$(basename ${url})
#
##def call_download_a_singlefile_with_URIString(args):
##    url=args.stuff[1]
##    filename=args.stuff[2]
##    dir_to_save=args.stuff[3]
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url} ${filename} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#
##call_download_files_in_a_resource_in_a_session_arguments=('call_download_files_in_a_resource_in_a_session' ${sessionID} "NIFTI_LOCATION" ${working_dir})
##outputfiles_present=$(python3 download_with_session_ID.py "${call_download_files_in_a_resource_in_a_session_arguments[@]}")
#
#while IFS=',' read -ra array1; do
##      echo "${array1[0]}"
#url1=${array1[0]}
#
#####################################################
##  resource_dir="MIDLINE_NPY"
##    output_csvfile_1_1=${sessionID}_MASK_METADATA.csv
##    call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_1_1}
##
##    while IFS=',' read -ra array2; do
##
##      url2=${array2[6]}
##      #################
##
##      if [[ ${url2} == *".npy"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
##        echo "It's there!"
##        echo "${array2[6]}"
##        filename2=$(basename ${url2})
##        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${output_directory})
##        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
##        greyfile=${dir_to_save}/${filename2}
##        echo "${url2}"
##      fi
##
##    done \
##      < <(tail -n +2 "${working_dir}/${output_csvfile_1_1}")
##    ################################################################
##    cp ${output_directory}/*_V2.npy ${working_dir_1}/
##
##
#
#
#####################################################
#
##      URI=/data/experiments/${sessionID}
#resource_dir="MASKS"
#
#output_csvfile_1=${sessionID}_MASK_METADATA.csv
#call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_1}
#
##    for niftifile_csvfilename in ${working_dir}/*NIFTILOCATION.csv; do
#niftifile_csvfilename=$(ls ${working_dir}/*NIFTILOCATION.csv)
#while IFS=',' read -ra array5; do
#scanID=${array5[2]}
#echo sessionId::${sessionID}
#echo scanId::${scanID}
#done < <(tail -n +2 "${niftifile_csvfilename}")
##      filename1=$(basename ${url1})
##  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
##  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#function_with_arguments=('call_delete_file_with_ext' ${sessionID} ${scanID} MASKS '_ventricle' ) ##'warped_1_mov_mri_region_' )
##    echo "outputfiles_present="'$(python3 utilities_simple_trimmed.py' "${function_with_arguments[@]}"
#outputfiles_present=$(python3 download_with_session_ID.py "${function_with_arguments[@]}")
#function_with_arguments=('call_delete_file_with_ext' ${sessionID} ${scanID} MASKS '_total' ) ##'warped_1_mov_mri_region_' )
##    echo "outputfiles_present="'$(python3 utilities_simple_trimmed.py' "${function_with_arguments[@]}"
#outputfiles_present=$(python3 download_with_session_ID.py "${function_with_arguments[@]}")
#while IFS=',' read -ra array2; do
#
#url2=${array2[6]}
#
#if [[ ${url2} == *"_vertical_bounding_box_512x512.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#greyfile=${dir_to_save}/${filename2}
#echo "${greyfile}"
#fi
#if [[ ${url2} == *"_levelset.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#greyfile=${dir_to_save}/${filename2}
#echo "${greyfile}"
#fi
#if [[ ${url2} == *"_levelset.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#greyfile=${dir_to_save}/${filename2}
#echo "${greyfile}"
#fi
#if [[ ${url2} == *"_levelset_bet.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#betfile=${dir_to_save}/${filename2}
#echo "${betfile}"
#fi
#if [[ ${url2} == *"_csf_unet.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#csffile=${dir_to_save}/${filename2}
#echo "${csffile}"
#fi
#done < <(tail -n +2 "${working_dir}/${output_csvfile_1}")
#
#
###############################################
#resource_dir="PREPROCESS_SEGM_3"
#output_csvfile_2=${sessionID}_PREPROCESS_SEGM_METADATA.csv
#call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_2}
##      filename1=$(basename ${url1})
##  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
##  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#echo "csffile:::::ATUL:::${csffile}"
#while IFS=',' read -ra array2; do
#url2=${array2[6]}
#
##          if [[ ${url2} == *"ventricle_bounds.csv"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
##            echo "It's there!"
##            echo "${array2[6]}"
##            filename2=$(basename ${url2})
##            call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
##            outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
##            ventricleboundfile=${dir_to_save}/${filename2}
##            echo "${ventricleboundfile}"
##          fi
#if [[ ${url2} == *"warped_1_mov_VENTRICLE_COLIHM62"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#venticle_only_mask=${dir_to_save}/${filename2}
#echo "${venticle_only_mask}"
#fi
#
#if [[ ${url2} == *"warped_1_mov_CISTERN_COLIHM62"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#cistern_only_mask=${dir_to_save}/${filename2}
#echo "${cistern_only_mask}"
#fi
#
##if [[ ${url2} == *"warped_1_mov_CISTERN_COLIHM62"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
##echo "It's there!"
##echo "${array2[6]}"
##filename2=$(basename ${url2})
##call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
##outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
##midline_only_mask=${dir_to_save}/${filename2}
##echo "${midline_only_mask}"
##fi
#
#done < <(tail -n +2 "${working_dir}/${output_csvfile_2}")
##########################################################
#
#
################################################################
#
##        venticle_only_mask=${betfile}
##        echo "${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
##        python3 findventriclemaskconvexhull10112024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
#ventricleboundfile=${dir_to_save}/'ventricle_bounds.csv'
#echo "python3 findventriclemaskobb_10102024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
#python3 findventriclemaskobb_10102024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
##
#echo "python3 findventriclemaskobb_03102025.py  ${cistern_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
#python3 findventriclemaskobb_03102025.py  ${cistern_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
#ventricle_obb_mask=${dir_to_save}/ventricle_obb_mask.nii
#ventricle_after_deepreg=${dir_to_save}/ventricle.nii
#cistern_after_deepreg=${dir_to_save}/cistern_after_deepreg.nii
#while IFS=',' read -ra array3; do
#echo "${array3[3]}::${array3[4]}"
#zoneV_min_z=${array3[3]}
#zoneV_max_z=${array3[4]}
#done < <(tail -n +2 "${ventricleboundfile}")
##############################################
#
#############
#
##    done
#
#
#################
#
##echo "call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )"
##call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )
##outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
###  echo ${outputfiles_present}
##fi
#echo "call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile} ${ventricle_after_deepreg} ${cistern_after_deepreg})"
##call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )
##exit 1
#call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile}  ${ventricle_after_deepreg} ${cistern_after_deepreg} )
#outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
#
#
##outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
#echo ${outputfiles_present}
#URI_1=${url2%/resource*}
#filename_prefix=$(basename ${url}) #${url2%/resource*} #filename=
#filename_prefix=${filename_prefix%_NIFTILOCATION*}
#resource_dirname="MASKS"
#this_data_basename=$(basename {greyfile})
#this_data_basename_noext=${this_data_basename%_resaved*}
#for file_name in ${dir_to_save}/${filename_prefix}*.nii.gz; do
#echo ${file_name}
#if [[ ${file_name} == *"${this_data_basename_noext}"* ]] || [[ ${file_name} == *"ventricle"* ]] || [[ ${file_name} == *"sulci"* ]]; then
#call_uploadsinglefile_with_URI_arguments=('call_uploadsinglefile_with_URI' ${URI_1} ${file_name} ${resource_dirname})
#outputfiles_present=$(python3 /software/download_with_session_ID.py "${call_uploadsinglefile_with_URI_arguments[@]}")
#echo ${outputfiles_present}
#fi
#done
#done < <(tail -n +2 "${dir_to_save}/${filename}")
#
#done \
#< \
#<(tail -n +2 "${working_dir}/${output_csvfile}")
#
### single filename NECT, its CSF mask and other relevant files
##rm /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/error.txt
##greyfile='/workinginput/SAH_1_01052014_2003_2_resaved_levelset.nii.gz'
#### '/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset.nii.gz'
##betfile='/workinginput/SAH_1_01052014_2003_2_resaved_levelset_bet.nii.gz'
###'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset_bet.nii.gz'
##csffile='/workinginput/SAH_1_01052014_2003_2_resaved_csf_unet.nii.gz'
###'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_final_seg.nii.gz'
##
##call_csf_compartments_arguments=('call_csf_compartments' ${greyfile} ${csffile} ${betfile} )
##outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}" )
##echo ${outputfiles_present}
####python /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/CSFCOMPARTMENT/CSF_COMPARTMENT_v1_part2_July18_2023.py
###
####pdflatex /media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/RESULTS/test.tex
