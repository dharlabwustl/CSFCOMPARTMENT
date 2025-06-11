#!/bin/bash

echo ">>> STARTING COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN.sh"

#----------------------------------------
# Environment Setup
#----------------------------------------
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
sessionID=${1}
scanID=""
working_dir=/workinginput
output_directory=/workingoutput
final_output_directory=/outputinsidedocker
working_dir_1=/input1
#----------------------------------------
# Function to get resource file metadata
#----------------------------------------
function call_get_resourcefiles_metadata_saveascsv_args() {
  local URI=${1}
  local resource_dir=${2}
  local final_output_directory=${3}
  local output_csvfile=${4}

  call_get_resourcefiles_metadata_saveascsv_args=('call_get_resourcefiles_metadata_saveascsv_args' ${URI} ${resource_dir} ${final_output_directory} ${output_csvfile})
  outputfiles_present=$(python3 download_with_session_ID.py "${call_get_resourcefiles_metadata_saveascsv_args[@]}")
  echo ">> Retrieved metadata for ${resource_dir}"
}

#----------------------------------------
# Function to extract scanID given a sessionID
#----------------------------------------

function get_scanID_from_sessionID() {
  local sessionID=$1
  local working_dir=$2
  local URI="/data/experiments/${sessionID}"
  local resource_dir="NIFTI_LOCATION"
  local output_csvfile="${sessionID}_SCANSELECTION_METADATA.csv"

  call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}

  local niftifile_csvfilename=$(ls ${working_dir}/*NIFTILOCATION.csv)
  scanID=$(tail -n +2 "${niftifile_csvfilename}" | cut -d',' -f3 | head -n 1)
  echo ${scanID}
}

#----------------------------------------
# Step 1: Download NIFTI_LOCATION Metadata
#----------------------------------------
URI="/data/experiments/${sessionID}"
resource_dir="NIFTI_LOCATION"
output_csvfile="${sessionID}_SCANSELECTION_METADATA.csv"
call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
#  call_download_a_singlefile_with_URIString_arguments=("call_get_resourcefiles_metadata_saveascsv_args" ${URI} ${resource_dir} ${working_dir} ${output_csvfile})
#  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#----------------------------------------
# Step 2: Download file(s) from metadata URLs
#----------------------------------------
while IFS=',' read -ra array; do
  url=${array[6]}
  filename=$(basename ${url})
  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url} ${filename} ${working_dir})
  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
done < <(tail -n +2 "${working_dir}/${output_csvfile}")

#----------------------------------------
# Step 3: Extract scanID from downloaded CSV
#----------------------------------------
get_scanID_from_sessionID ${sessionID} ${working_dir}
echo ${sessionID}::${scanID}
dir_to_save=${working_dir}
resource_dir="MIDLINE_NPY"
call_download_a_singlefile_with_URIString_arguments=('call_dowload_a_folder_as_zip' ${sessionID} ${scanID} ${resource_dir})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")

greyfile="NONE" ##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset.nii.gz'
betfile="NONE"  ##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset_bet.nii.gz'
csffile="NONE"  ##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_final_seg.nii.gz'
while IFS=',' read -ra array; do
#xx=0
#
##if [ ${array[1]} == "SNIPR01_E00894" ]  ; then
#  echo "${array[6]}"
url=${array[6]}
filename=$(basename ${url})

#def call_download_a_singlefile_with_URIString(args):
#    url=args.stuff[1]
#    filename=args.stuff[2]
#    dir_to_save=args.stuff[3]
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url} ${filename} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")

#call_download_files_in_a_resource_in_a_session_arguments=('call_download_files_in_a_resource_in_a_session' ${sessionID} "NIFTI_LOCATION" ${working_dir})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_files_in_a_resource_in_a_session_arguments[@]}")

while IFS=',' read -ra array1; do
#      echo "${array1[0]}"
url1=${array1[0]}
#      URI=/data/experiments/${sessionID}
resource_dir="MASKS"

output_csvfile_1=${sessionID}_MASK_METADATA.csv
call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_1}

##    for niftifile_csvfilename in ${working_dir}/*NIFTILOCATION.csv; do
#niftifile_csvfilename=$(ls ${working_dir}/*NIFTILOCATION.csv)
#while IFS=',' read -ra array5; do
#scanID=${array5[2]}
#echo sessionId::${sessionID}
#echo scanId::${scanID}
#done < <(tail -n +2 "${niftifile_csvfilename}")
#      filename1=$(basename ${url1})
#  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
#  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
function_with_arguments=('call_delete_file_with_ext' ${sessionID} ${scanID} MASKS '_ventricle' ) ##'warped_1_mov_mri_region_' )
    echo "outputfiles_present="'$(python3 utilities_simple_trimmed.py' "${function_with_arguments[@]}"
outputfiles_present=$(python3 download_with_session_ID.py "${function_with_arguments[@]}")
#function_with_arguments=('call_delete_file_with_ext' ${sessionID} ${scanID} MASKS '_total' ) ##'warped_1_mov_mri_region_' )
#    echo "outputfiles_present="'$(python3 utilities_simple_trimmed.py' "${function_with_arguments[@]}"
#outputfiles_present=$(python3 download_with_session_ID.py "${function_with_arguments[@]}")
while IFS=',' read -ra array2; do

url2=${array2[6]}

if [[ ${url2} == *"_vertical_bounding_box_512x512.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
greyfile=${dir_to_save}/${filename2}
echo "${greyfile}"
fi
if [[ ${url2} == *"_levelset.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
greyfile=${dir_to_save}/${filename2}
echo "${greyfile}"
fi
if [[ ${url2} == *"_levelset.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
greyfile=${dir_to_save}/${filename2}
echo "${greyfile}"
fi
if [[ ${url2} == *"_levelset_bet.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
betfile=${dir_to_save}/${filename2}
echo "${betfile}"
fi
if [[ ${url2} == *"_csf_unet.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
csffile=${dir_to_save}/${filename2}
echo "${csffile}"
fi
done < <(tail -n +2 "${working_dir}/${output_csvfile_1}")


##############################################
resource_dir="PREPROCESS_SEGM_3"
output_csvfile_2=${sessionID}_PREPROCESS_SEGM_METADATA.csv
call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_2}
#      filename1=$(basename ${url1})
#  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
#  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
echo "csffile:::::ATUL:::${csffile}"
while IFS=',' read -ra array2; do
url2=${array2[6]}

#          if [[ ${url2} == *"ventricle_bounds.csv"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#            echo "It's there!"
#            echo "${array2[6]}"
#            filename2=$(basename ${url2})
#            call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#            outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#            ventricleboundfile=${dir_to_save}/${filename2}
#            echo "${ventricleboundfile}"
#          fi
if [[ ${url2} == *"warped_1_mov_VENTRICLE_COLIHM62"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
venticle_only_mask=${dir_to_save}/${filename2}
echo "${venticle_only_mask}"
fi

if [[ ${url2} == *"warped_1_mov_CISTERN_COLIHM62"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
cistern_only_mask=${dir_to_save}/${filename2}
echo "${cistern_only_mask}"
fi

if [[ ${url2} == *"warped_1_mov_CISTERN_COLIHM62"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
echo "It's there!"
echo "${array2[6]}"
filename2=$(basename ${url2})
call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
midline_only_mask=${dir_to_save}/${filename2}
echo "${midline_only_mask}"
fi

done < <(tail -n +2 "${working_dir}/${output_csvfile_2}")

for each_npy in  $(find /ZIPFILEDIR/ -name '*.npy') ;  do  if [[ $each_npy  == *'V2'* ]] ; then  mv $each_npy ${working_dir_1} ; fi ; done
 for each_npy in  $(find /ZIPFILEDIR/ -name '*.npy') ;  do  if [[ $each_npy  == *'.npy'* ]] ; then  mv $each_npy ${output_directory} ; fi ; done
##################################################################################################################################
#################################################################################################################################

##############################################


#output_csvfile_2=${sessionID}_PREPROCESS_SEGM_METADATA.csv
#call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_2}
##      filename1=$(basename ${url1})
##  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
##  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#echo "csffile:::::ATUL:::${csffile}"
#while IFS=',' read -ra array2; do
#url2=${array2[6]}
#
#if [[ ${url2} == *".npy"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#echo "It's there!"
#echo "${array2[6]}"
#filename2=$(basename ${url2})
#call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${working_dir_1})
#outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#fi
#
#done < <(tail -n +2 "${working_dir}/${output_csvfile_2}")





#############################################################################################################################################
############################################################################################################################################


#        venticle_only_mask=${betfile}
#        echo "${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
#        python3 findventriclemaskconvexhull10112024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
ventricleboundfile=${dir_to_save}/'ventricle_bounds.csv'
echo "python3 findventriclemaskobb_10102024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
python3 findventriclemaskobb_10102024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
#
echo "python3 findventriclemaskobb_03102025.py  ${cistern_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
python3 findventriclemaskobb_03102025.py  ${cistern_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
ventricle_obb_mask=${dir_to_save}/ventricle_obb_mask.nii
ventricle_after_deepreg=${dir_to_save}/ventricle.nii
cistern_after_deepreg=${dir_to_save}/cistern_after_deepreg.nii
while IFS=',' read -ra array3; do
echo "${array3[3]}::${array3[4]}"
zoneV_min_z=${array3[3]}
zoneV_max_z=${array3[4]}
done < <(tail -n +2 "${ventricleboundfile}")
#############################################

############

#    done


################

#echo "call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )"
#call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )
#outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
##  echo ${outputfiles_present}
#fi
echo "call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile} ${ventricle_after_deepreg} ${cistern_after_deepreg} ${output_directory})"
#call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )
#exit 1
call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile}  ${ventricle_after_deepreg} ${cistern_after_deepreg}  ${output_directory})
outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")


#outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
echo ${outputfiles_present}
URI_1=${url2%/resource*}
filename_prefix=$(basename ${url}) #${url2%/resource*} #filename=
filename_prefix=${filename_prefix%_NIFTILOCATION*}
resource_dirname="MASKS"
this_data_basename=$(basename {greyfile})
this_data_basename_noext=${this_data_basename%_resaved*}
for file_name in ${dir_to_save}/${filename_prefix}*.nii.gz; do
echo ${file_name}
if [[ ${file_name} == *"${this_data_basename_noext}"* ]] || [[ ${file_name} == *"ventricle"* ]] || [[ ${file_name} == *"sulci"* ]]; then
call_uploadsinglefile_with_URI_arguments=('call_uploadsinglefile_with_URI' ${URI_1} ${file_name} ${resource_dirname})
outputfiles_present=$(python3 /software/download_with_session_ID.py "${call_uploadsinglefile_with_URI_arguments[@]}")
echo ${outputfiles_present}
fi
done
done < <(tail -n +2 "${dir_to_save}/${filename}")

done \
< \
<(tail -n +2 "${working_dir}/${output_csvfile}")