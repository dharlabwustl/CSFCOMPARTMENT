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

  local args=('call_get_resourcefiles_metadata_saveascsv_args' "${URI}" "${resource_dir}" "${final_output_directory}" "${output_csvfile}")
  outputfiles_present=$(python3 download_with_session_ID.py "${args[@]}")
  echo ">> Retrieved metadata for ${resource_dir}"
}

#----------------------------------------
# Download NIFTI_LOCATION Metadata
#----------------------------------------
URI="/data/experiments/${sessionID}"
resource_dir="NIFTI_LOCATION"
output_csvfile="${sessionID}_SCANSELECTION_METADATA.csv"
call_get_resourcefiles_metadata_saveascsv_args "${URI}" "${resource_dir}" "${working_dir}" "${output_csvfile}"

dir_to_save=${working_dir}
greyfile="NONE"
betfile="NONE"
csffile="NONE"

#----------------------------------------
# Download individual scan files
#----------------------------------------
while IFS=',' read -ra array; do
  url=${array[6]}
  filename=$(basename "${url}")
  args=('call_download_a_singlefile_with_URIString' "${url}" "${filename}" "${dir_to_save}")
  python3 download_with_session_ID.py "${args[@]}"

  # Get MASKS metadata
  while IFS=',' read -ra array1; do
    url1=${array1[0]}
    output_csvfile_1="${sessionID}_MASK_METADATA.csv"
    call_get_resourcefiles_metadata_saveascsv_args "${url1}" "MASKS" "${working_dir}" "${output_csvfile_1}"

    # Get scanID from NIFTILOCATION CSV
    niftifile_csvfilename=$(ls "${working_dir}"/*NIFTILOCATION.csv)
    scanID=$(tail -n +2 "${niftifile_csvfilename}" | cut -d',' -f3)

    # Delete existing ventricle and total masks
    python3 download_with_session_ID.py "call_delete_file_with_ext" "${sessionID}" "${scanID}" "MASKS" "_ventricle"
    python3 download_with_session_ID.py "call_delete_file_with_ext" "${sessionID}" "${scanID}" "MASKS" "_total"

    # Process mask files
    while IFS=',' read -ra array2; do
      url2=${array2[6]}
      filename2=$(basename "${url2}")

      case "${url2}" in
        *"_vertical_bounding_box_512x512.nii.gz"*)
          greyfile="${dir_to_save}/${filename2}"
          ;;
        *"_levelset.nii.gz"*)
          greyfile="${dir_to_save}/${filename2}"
          ;;
        *"_levelset_bet.nii.gz"*)
          betfile="${dir_to_save}/${filename2}"
          ;;
        *"_csf_unet.nii.gz"*)
          csffile="${dir_to_save}/${filename2}"
          ;;
      esac

      if [[ -n ${filename2} ]]; then
        args=('call_download_a_singlefile_with_URIString' "${url2}" "${filename2}" "${dir_to_save}")
        python3 download_with_session_ID.py "${args[@]}"
        echo ">> Downloaded ${filename2}"
      fi
    done < <(tail -n +2 "${working_dir}/${output_csvfile_1}")

    #----------------------------------------
    # Process PREPROCESS_SEGM_3
    #----------------------------------------
    output_csvfile_2="${sessionID}_PREPROCESS_SEGM_METADATA.csv"
    call_get_resourcefiles_metadata_saveascsv_args "${url1}" "PREPROCESS_SEGM_3" "${working_dir}" "${output_csvfile_2}"

    while IFS=',' read -ra array2; do
      url2=${array2[6]}
      filename2=$(basename "${url2}")

      case "${url2}" in
        *"warped_1_mov_VENTRICLE_COLIHM62"*)
          venticle_only_mask="${dir_to_save}/${filename2}"
          ;;
        *"warped_1_mov_CISTERN_COLIHM62"*)
          cistern_only_mask="${dir_to_save}/${filename2}"
          ;;
      esac

      if [[ -n ${filename2} ]]; then
        args=('call_download_a_singlefile_with_URIString' "${url2}" "${filename2}" "${dir_to_save}")
        python3 download_with_session_ID.py "${args[@]}"
      fi
    done < <(tail -n +2 "${working_dir}/${output_csvfile_2}")

    #----------------------------------------
    # Run Ventricular Processing
    #----------------------------------------
    ventricleboundfile="${dir_to_save}/ventricle_bounds.csv"
    python3 findventriclemaskobb_10102024.py "${venticle_only_mask}" "${csffile}" "${dir_to_save}" "${greyfile}" "${betfile}"
    python3 findventriclemaskobb_03102025.py "${cistern_only_mask}" "${csffile}" "${dir_to_save}" "${greyfile}" "${betfile}"

    ventricle_after_deepreg="${dir_to_save}/ventricle.nii"
    cistern_after_deepreg="${dir_to_save}/cistern_after_deepreg.nii"

    # Get ventricle z-bounds
    while IFS=',' read -ra array3; do
      zoneV_min_z=${array3[3]}
      zoneV_max_z=${array3[4]}
    done < <(tail -n +2 "${ventricleboundfile}")

    #----------------------------------------
    # Call CSF Compartment Segmentation
    #----------------------------------------
    args=('call_csf_compartments_ventbound_no_hem_with_cis_1' "${greyfile}" "${csffile}" "${ventricle_after_deepreg}" "${cistern_after_deepreg}")
    python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${args[@]}"
    echo ">> Completed CSF Compartment Processing"

    #----------------------------------------
    # Upload results to XNAT
    #----------------------------------------
    URI_1=${url2%/resource*}
    filename_prefix=$(basename "${url}")
    filename_prefix=${filename_prefix%_NIFTILOCATION*}
    resource_dirname="MASKS"
    this_data_basename_noext=$(basename "${greyfile}" | cut -d'_' -f1)

    for file_name in "${dir_to_save}/${filename_prefix}"*.nii.gz; do
      if [[ ${file_name} == *"${this_data_basename_noext}"* ]] || [[ ${file_name} == *"ventricle"* ]] || [[ ${file_name} == *"sulci"* ]]; then
        args=('call_uploadsinglefile_with_URI' "${URI_1}" "${file_name}" "${resource_dirname}")
        python3 /software/download_with_session_ID.py "${args[@]}"
        echo ">> Uploaded ${file_name}"
      fi
    done

  done < <(tail -n +2 "${dir_to_save}/${filename}")
done < <(tail -n +2 "${working_dir}/${output_csvfile}")

echo ">>> DONE"
