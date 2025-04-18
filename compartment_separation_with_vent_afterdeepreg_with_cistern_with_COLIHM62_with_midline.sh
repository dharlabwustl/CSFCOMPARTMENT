#!/bin/bash

echo ">>> STARTING COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN.sh"

#----------------------------------------
# Environment Setup
#----------------------------------------
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
sessionID=${1}

working_dir=/workinginput
output_directory=/workingoutput
final_output_directory=/outputinsidedocker

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
  local scanID=$(tail -n +2 "${niftifile_csvfilename}" | cut -d',' -f3 | head -n 1)
  echo ${scanID}
}

#----------------------------------------
# Step 1: Download NIFTI_LOCATION Metadata
#----------------------------------------
URI="/data/experiments/${sessionID}"
resource_dir="NIFTI_LOCATION"
output_csvfile="${sessionID}_SCANSELECTION_METADATA.csv"
#call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
  call_download_a_singlefile_with_URIString_arguments=("call_get_resourcefiles_metadata_saveascsv_args" ${URI} ${resource_dir} ${working_dir} ${output_csvfile})
  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
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
scanID=$(get_scanID_from_sessionID ${sessionID} ${working_dir})
echo ${sessionID}::${scanID}
#----------------------------------------
# Get scanID before main loop
#----------------------------------------

#----------------------------------------
# Download NIFTI_LOCATION Metadata
#----------------------------------------
#scanID=$(get_scanID_from_sessionID ${sessionID} ${working_dir})
#echo ${scanID}
#URI="/data/experiments/${sessionID}"
#resource_dir="NIFTI_LOCATION"
#output_csvfile="${sessionID}_SCANSELECTION_METADATA.csv"
#call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
#
#dir_to_save=${working_dir}
#greyfile="NONE"
#betfile="NONE"
#csffile="NONE"
#
##----------------------------------------
## Main Loop to Download and Process Files
##----------------------------------------
#while IFS=',' read -ra array; do
#  url=${array[6]}
#  filename=$(basename ${url})
#
#  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url} ${filename} ${dir_to_save})
#  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#
#  while IFS=',' read -ra array1; do
#    url1=${array1[0]}
#    output_csvfile_1="${sessionID}_MASK_METADATA.csv"
#    call_get_resourcefiles_metadata_saveascsv_args ${url1} MASKS ${working_dir} ${output_csvfile_1}
#
#    scanID=$(get_scanID_from_sessionID ${sessionID} ${working_dir})
#
#    function_with_arguments=('call_delete_file_with_ext' ${sessionID} ${scanID} MASKS '_ventricle')
#    outputfiles_present=$(python3 download_with_session_ID.py "${function_with_arguments[@]}")
#    function_with_arguments=('call_delete_file_with_ext' ${sessionID} ${scanID} MASKS '_total')
#    outputfiles_present=$(python3 download_with_session_ID.py "${function_with_arguments[@]}")
#
#    while IFS=',' read -ra array2; do
#      url2=${array2[6]}
#      filename2=$(basename ${url2})
#
#      if [[ ${url2} == *"_vertical_bounding_box_512x512.nii.gz"* ]]; then
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#        greyfile=${dir_to_save}/${filename2}
#      fi
#
#      if [[ ${url2} == *"_levelset.nii.gz"* ]]; then
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#        greyfile=${dir_to_save}/${filename2}
#      fi
#
#      if [[ ${url2} == *"_levelset_bet.nii.gz"* ]]; then
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#        betfile=${dir_to_save}/${filename2}
#      fi
#
#      if [[ ${url2} == *"_csf_unet.nii.gz"* ]]; then
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#        csffile=${dir_to_save}/${filename2}
#      fi
#    done < <(tail -n +2 "${working_dir}/${output_csvfile_1}")
#
#    # Download PREPROCESS_SEGM_3 masks
#    output_csvfile_2="${sessionID}_PREPROCESS_SEGM_METADATA.csv"
#    call_get_resourcefiles_metadata_saveascsv_args ${url1} PREPROCESS_SEGM_3 ${working_dir} ${output_csvfile_2}
#
#    while IFS=',' read -ra array2; do
#      url2=${array2[6]}
#      filename2=$(basename ${url2})
#
#      if [[ ${url2} == *"warped_1_mov_VENTRICLE_COLIHM62"* ]]; then
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#        venticle_only_mask=${dir_to_save}/${filename2}
#      fi
#
#      if [[ ${url2} == *"warped_1_mov_CISTERN_COLIHM62"* ]]; then
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#        cistern_only_mask=${dir_to_save}/${filename2}
#        midline_only_mask=${dir_to_save}/${filename2}
#      fi
#    done < <(tail -n +2 "${working_dir}/${output_csvfile_2}")
#
#    ventricleboundfile=${dir_to_save}/'ventricle_bounds.csv'
#    python3 findventriclemaskobb_10102024.py ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
#    python3 findventriclemaskobb_03102025.py ${cistern_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
#
#    while IFS=',' read -ra array3; do
#      zoneV_min_z=${array3[3]}
#      zoneV_max_z=${array3[4]}
#    done < <(tail -n +2 "${ventricleboundfile}")
#
#    ventricle_after_deepreg=${dir_to_save}/ventricle.nii
#    cistern_after_deepreg=${dir_to_save}/cistern_after_deepreg.nii
#
#    call_csf_compartments_arguments=('call_csf_compartments_ventbound_no_hem_with_cis_1' ${greyfile} ${csffile} ${ventricle_after_deepreg} ${cistern_after_deepreg})
#    outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
#
#    URI_1=${url2%/resource*}
#    filename_prefix=$(basename ${url})
#    filename_prefix=${filename_prefix%_NIFTILOCATION*}
#    this_data_basename_noext=$(basename ${greyfile})
#    this_data_basename_noext=${this_data_basename_noext%_resaved*}
#
#    for file_name in ${dir_to_save}/${filename_prefix}*.nii.gz; do
#      if [[ ${file_name} == *"${this_data_basename_noext}"* ]] || [[ ${file_name} == *"ventricle"* ]] || [[ ${file_name} == *"sulci"* ]]; then
#        call_uploadsinglefile_with_URI_arguments=('call_uploadsinglefile_with_URI' ${URI_1} ${file_name} MASKS)
#        outputfiles_present=$(python3 /software/download_with_session_ID.py "${call_uploadsinglefile_with_URI_arguments[@]}")
#      fi
#    done
#  done < <(tail -n +2 "${dir_to_save}/${filename}")
#done < <(tail -n +2 "${working_dir}/${output_csvfile}")
#
#echo ">>> DONE"
