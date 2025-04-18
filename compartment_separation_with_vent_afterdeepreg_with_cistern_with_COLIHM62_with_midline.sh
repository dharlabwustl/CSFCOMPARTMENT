#!/bin/bash

echo "Starting COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN.sh"

# Input args
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
sessionID=${1}

# Directories
working_dir=/workinginput
output_directory=/workingoutput
final_output_directory=/outputinsidedocker

# Create directories
mkdir -p "${working_dir}/ventricles" "${working_dir}/gray" "${working_dir}/csf" "${working_dir}/cisterns"

# Download metadata for NIFTI_LOCATION and extract scanID
URI="/data/experiments/${sessionID}"
resource_dir="NIFTILOCATION"
output_csvfile="${working_dir}/${sessionID}_SCANSELECTION_METADATA.csv"

python3 download_with_session_ID.py "call_get_resourcefiles_metadata_saveascsv_args" ${URI} ${resource_dir} ${working_dir} ${output_csvfile}

scanID=""
while IFS=',' read -ra line; do
  scanID=${line[2]}
  break  # only need first scanID
done < <(tail -n +2 "${output_csvfile}")
maskfile_download_dir=${working_dir}
# Resource directories, patterns, and save locations
resource_dirs=("MASKS" "MASKS" "MASKS" "PREPROCESS_SEGM_3" "PREPROCESS_SEGM_3")
file_patterns=("_ventricle" "_levelset" "_csf_unet" "warped_1_mov_VENTRICLE" "warped_1_mov_CISTERN")
save_dirs=("${maskfile_download_dir}" "${maskfile_download_dir}" "${maskfile_download_dir}" "${maskfile_download_dir}" "${maskfile_download_dir}")

# Iterate through resources
for i in "${!resource_dirs[@]}"; do
  resource="${resource_dirs[$i]}"
  pattern="${file_patterns[$i]}"
  save_subdir="${save_dirs[$i]}"
  save_dir="${working_dir}${save_subdir}"
  mkdir -p "${save_dir}"

  metadata_csv="${working_dir}/${sessionID}_${resource}_METADATA.csv"
  python3 download_with_session_ID.py "call_get_resourcefiles_metadata_saveascsv_args" ${URI} ${resource} ${working_dir} ${metadata_csv}

  while IFS=',' read -ra columns; do
    url="${columns[6]}"
    if [[ "${url}" == *"${pattern}"* ]]; then
      filename=$(basename "${url}")
      echo "Downloading: ${filename} to ${save_dir}"
      python3 download_with_session_ID.py "call_download_a_singlefile_with_URIString" "${url}" "${filename}" "${save_dir}"
    fi
  done < <(tail -n +2 "${metadata_csv}")
done

echo "All files downloaded."
