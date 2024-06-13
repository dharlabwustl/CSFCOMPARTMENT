#!/bin/bash
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
project_ID=${1}
counter_start=${5}
counter_end=${6}
working_dir=/workinginput
output_directory=/workingoutput
final_output_directory=/outputinsidedocker
working_dir_1=/input
ZIPFILEDIR=/ZIPFILEDIR
NIFTIFILEDIR=/NIFTIFILEDIR
DICOMFILEDIR=/DICOMFILEDIR
working=/working
input=/input
output=/output
software=/software
function directory_to_create_destroy(){

rm  -r    ${working_dir}/*
rm  -r    ${working_dir_1}/*
rm  -r    ${output_directory}/*
rm  -r    ${final_output_directory}/*
# rm  -r    ${software}/*
rm  -r    ${ZIPFILEDIR}/*
rm  -r    ${NIFTIFILEDIR}/*
rm  -r    ${DICOMFILEDIR}/*
rm  -r    ${working}/*
rm  -r    ${input}/*
rm  -r    ${output}/*


}
# function scan_selection(){
# local SESSION_ID=${1}
# git_repo='https://github.com/dharlabwustl/EDEMA_MARKERS_PROD.git'
# script_number=FILLREDCAPONLY ##DICOM2NIFTI #SCAN_SELECTION_FILL_RC #EDEMABIOMARKERS # 12 #SCAN_SELECTION_FILL_RC # EDEMABIOMARKERS #SCAN_SELECTION_FILL_RC #DICOM2NIFTI #SCAN_SELECTION_FILL_RC #REDCAP_FILL_SESSION_NAME #SCAN_SELECTION_FILL_RC #REDCAP_FILL_SESSION_NAME ##SCAN_SELECTION_FILL_RC #REDCAP_FILL_SESSION_NAME #SCAN_SELECTION_FILL_RC #12
# snipr_host='https://snipr.wustl.edu' 
# /callfromgithub/downloadcodefromgithub.sh $SESSION_ID $XNAT_USER $XNAT_PASS ${git_repo} ${script_number}  ${snipr_host}  EC6A2206FF8C1D87D4035E61C99290FF
# }
directory_to_create_destroy
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
# Get the header row and split it into columns
get_column_number(){
local  CSV_FILE=${1}
local  COLUMN_NAME=${2}
local HEADER=$(head -n 1 "$CSV_FILE")

# Convert the header to an array of column names
IFS=',' read -r -a COLUMNS <<< "$HEADER"

# Initialize column index
COLUMN_INDEX=-1

# Iterate over columns to find the index
for i in "${!COLUMNS[@]}"; do
    if [[ "${COLUMNS[$i]}" == "$COLUMN_NAME" ]]; then
        COLUMN_INDEX=$((i + 1))
        break
    fi
done
echo "$COLUMN_INDEX"
}

URI=/data/projects/${project_ID}

resource_dir="INCOMPLETE"
output_csvfile=${project_ID}_INCOMPLETE_METADATA.csv
echo ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
URI_COL_NUM=$(get_column_number ${working_dir}/$output_csvfile URI)
echo URI_COL_NUM::${URI_COL_NUM}
CSV_FILE=${working_dir}/${output_csvfile}
column_value=$(cut -d ',' -f "$URI_COL_NUM" "$CSV_FILE")
#
# ##############################
# # # Get the header row and split it into columns
# # HEADER=$(head -n 1 "$CSV_FILE")

# # # Convert the header to an array of column names
# # IFS=',' read -r -a COLUMNS <<< "$HEADER"

# # # Initialize column index
# # COLUMN_INDEX=-1

# # # Iterate over columns to find the index
# # for i in "${!COLUMNS[@]}"; do
# #     if [[ "${COLUMNS[$i]}" == "$COLUMN_NAME" ]]; then
# #         COLUMN_INDEX=$((i + 1))
# #         break
# #     fi
# # done

# ################################
# sessions_list=${software}/session.csv 
# curl -u $XNAT_USER:$XNAT_PASS -X GET $XNAT_HOST/data/projects/${project_ID}'/experiments/?xsiType=xnat:ctSessionData&format=csv' > ${sessions_list}
# ######################################
# count=0
#   while IFS=',' read -ra array; do
#   # if [ ${count} -ge ${counter_start} ]; then
#   if [[ ${counter_start} == ${array[0]} ]] ; then
#     echo SESSION_ID::${array[0]}
#     SESSION_ID=${array[0]}  #SNIPR02_E10218 ##SNIPR02_E10112 #
#     SESSION_NAME=${array[5]} 

#     # echo SESSION_NAME::${SESSION_NAME}
#     directory_to_create_destroy
#     /software/compartment_separation.sh $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST /input /output
#     # echo snipr_step::${snipr_step}
#     # scan_selection ${SESSION_ID}  

#     # echo "$SESSION_ID,$SESSION_NAME" >> ${list_accomplished}
#   fi 
#     count=$((count+1))
#     echo "THIS COUNT NUMBER IS "::${count}::${counter_end}
# #     fi
#     if [ ${count} -ge ${counter_end} ]; then
#     break
#     fi
# done < <(tail -n +2 "${sessions_list}")

