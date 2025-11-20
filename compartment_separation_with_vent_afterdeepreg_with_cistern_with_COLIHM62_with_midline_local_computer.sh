#!/bin/bash
set -euo pipefail

echo ">>> STARTING COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN_LOCAL.sh"

#############################################
# Directory setup (local, no XNAT)
#############################################

working_dir=/workinginput          # CT + masks + CSV live here (after copy)
working_dir_1=/input1              # will receive selected NIFTI / npy
output_directory=/workingoutput    # final outputs + other .npy files
dir_to_save="${working_dir}"       # same directory used in original script
zip_dir=/ZIPFILEDIR                # where .npy files are initially stored

# Ensure directories exist
for d in "$working_dir" "$working_dir_1" "$output_directory" "$zip_dir"; do
  if [ ! -d "$d" ]; then
    echo "Creating directory: $d"
    mkdir -p "$d"
  fi
done

#############################################
# COPY INPUTS FROM /input/SCANS/... (LOCAL)
#############################################

echo ">>> Copying local scan files from /input/SCANS/2/... into working dirs"

# NIFTI -> /input1
if [ -d /input/SCANS/2/NIFTI ]; then
  cp /input/SCANS/2/NIFTI/* "${working_dir_1}/"
fi

# PREPROCESS_SEGM + PREPROCESS_SEGM_3 + MASKS -> /workinginput
if [ -d /input/SCANS/2/PREPROCESS_SEGM ]; then
  cp /input/SCANS/2/PREPROCESS_SEGM/* "${working_dir}/"
fi

if [ -d /input/SCANS/2/PREPROCESS_SEGM_3 ]; then
  cp /input/SCANS/2/PREPROCESS_SEGM_3/* "${working_dir}/"
fi

if [ -d /input/SCANS/2/MASKS ]; then
  cp /input/SCANS/2/MASKS/* "${working_dir}/"
fi

echo ">>> Done copying. Now locating required NIfTI files in ${dir_to_save}"


#############################################
# Helper: find a single file by pattern
#############################################
find_single_file() {
  local pattern="$1"
  local search_dir="$2"
  # maxdepth 1 because original logic expected files directly under dir_to_save
  find "$search_dir" -maxdepth 1 -type f -name "$pattern" | head -n 1
}

#############################################
# Locate required NIfTI inputs (local only)
#############################################

echo ">>> Locating required NIfTI files in ${dir_to_save}"

# Grey image (vertical bounding box or levelset)
greyfile=$(find_single_file "*_vertical_bounding_box_512x512.nii.gz" "$dir_to_save")
if [[ -z "${greyfile}" ]]; then
  greyfile=$(find_single_file "*_levelset.nii.gz" "$dir_to_save" || true)
fi

# BET file (optional but usually present)
betfile=$(find_single_file "*_levelset_bet.nii.gz" "$dir_to_save" || true)

# CSF segmentation
csffile=$(find_single_file "*_csf_unet.nii.gz" "$dir_to_save" || true)

# Ventricle & cistern masks from DeepReg
ventricle_only_mask=$(find_single_file "*warped_1_mov_VENTRICLE_COLIHM62*.nii*" "$dir_to_save" || true)
cistern_only_mask=$(find_single_file "*warped_1_mov_CISTERN_COLIHM62*.nii*" "$dir_to_save" || true)

echo "  greyfile           : ${greyfile:-NOT FOUND}"
echo "  betfile            : ${betfile:-NOT FOUND}"
echo "  csffile            : ${csffile:-NOT FOUND}"
echo "  ventricle_only_mask: ${ventricle_only_mask:-NOT FOUND}"
echo "  cistern_only_mask  : ${cistern_only_mask:-NOT FOUND}"

# Basic sanity checks
if [[ -z "${greyfile}" || -z "${csffile}" || -z "${ventricle_only_mask}" || -z "${cistern_only_mask}" ]]; then
  echo "ERROR: One or more required NIfTI files are missing."
  echo "Check that they exist in ${dir_to_save} with expected name patterns."
  exit 1
fi

if [[ -z "${betfile}" ]]; then
  echo "WARNING: BET file not found (*_levelset_bet.nii.gz). Passing 'NONE' to Python."
  betfile="NONE"
fi

#############################################
# Move local .npy files (no download)
#############################################

echo ">>> Organizing local .npy files from ${zip_dir}"

# V2 npys -> /input1, others -> /workingoutput
while IFS= read -r -d '' each_npy; do
  if [[ "$each_npy" == *"V2"* ]]; then
    echo "  Moving V2 npy -> ${working_dir_1}: $(basename "$each_npy")"
    mv "$each_npy" "${working_dir_1}/"
  else
    echo "  Moving npy -> ${output_directory}: $(basename "$each_npy")"
    mv "$each_npy" "${output_directory}/"
  fi
done < <(find "$zip_dir" -type f -name "*.npy" -print0)

#############################################
# Ventricle bounds CSV (assumed local)
#############################################

ventricleboundfile="${dir_to_save}/ventricle_bounds.csv"

#############################################
# Run ventricle & cistern OBB / DeepReg steps
#############################################

echo ">>> Running ventricle OBB step:"
echo "python3 findventriclemaskobb_10102024.py \\"
echo "  ${ventricle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"

python3 findventriclemaskobb_10102024.py \
  "${ventricle_only_mask}" \
  "${csffile}" \
  "${dir_to_save}" \
  "${greyfile}" \
  "${betfile}"

echo ">>> Running cistern OBB step:"
echo "python3 findventriclemaskobb_03102025.py \\"
echo "  ${cistern_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"


python3 findventriclemaskobb_03102025.py \
  "${cistern_only_mask}" \
  "${csffile}" \
  "${dir_to_save}" \
  "${greyfile}" \
  "${betfile}"

#############################################
# CSF compartment computation
#############################################

ventricle_after_deepreg="${dir_to_save}/ventricle.nii"
cistern_after_deepreg="${dir_to_save}/cistern_after_deepreg.nii"
if [[ -f "${ventricle_after_deepreg}" ]]; then
  echo ">>> Found ventricle bounds CSV: ${ventricle_after_deepreg}"
else
  echo "WARNING: ${ventricle_after_deepreg} not found in ${dir_to_save}"

fi

if [[ -f "${cistern_after_deepreg}" ]]; then
  echo ">>> Found ventricle bounds CSV: ${cistern_after_deepreg}"
else
  echo "WARNING: ${cistern_after_deepreg} not found in ${dir_to_save}"

fi


if [[ -f "${ventricleboundfile}" ]]; then
  echo ">>> Found ventricle bounds CSV: ${ventricleboundfile}"
else
  echo "WARNING: ventricle_bounds.csv not found in ${dir_to_save}"
  echo "         Continuing without using zoneV_min_z / zoneV_max_z (they are unused in active code)."
fi
echo ">>> Running CSF compartment computation:"
echo "call_csf_compartments_arguments = ('call_csf_compartments_ventbound_no_hem_with_cis_1' \\"
echo "  ${greyfile} ${csffile} ${ventricle_after_deepreg} ${cistern_after_deepreg} ${output_directory})"

call_csf_compartments_arguments=(
  "call_csf_compartments_ventbound_no_hem_with_cis_1"
  "${greyfile}"
  "${csffile}"
  "${ventricle_after_deepreg}"
  "${cistern_after_deepreg}"
  "${output_directory}"
)

outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")

echo ">>> CSF compartment script output:"
echo "${outputfiles_present}"

echo ">>> DONE: COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN_LOCAL.sh"
