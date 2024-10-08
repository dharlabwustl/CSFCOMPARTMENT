#!/bin/bash
# echo "
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
project_ID=${1}
counter_start=${5} ## session id
counter_end=${6} ## nifti file name
zoneV_min_z=${7}
zoneV_max_z=${8} 
# "
echo ::$XNAT_USER::$XNAT_PASS::$XNAT_HOST::${zoneV_min_z}::${zoneV_max_z} 
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
sessions_list=${software}/session.csv 
curl -u $XNAT_USER:$XNAT_PASS -X GET $XNAT_HOST/data/projects/${project_ID}'/experiments/?xsiType=xnat:ctSessionData&format=csv' > ${sessions_list}
######################################
count=0
  while IFS=',' read -ra array; do
  # if [ ${count} -ge ${counter_start} ]; then
  if [[ ${counter_start} == ${array[0]} ]] ; then
    echo SESSION_ID::${array[0]}
    SESSION_ID=${array[0]}  #SNIPR02_E10218 ##SNIPR02_E10112 #
    SESSION_NAME=${array[5]} 

    # echo SESSION_NAME::${SESSION_NAME}
    directory_to_create_destroy
echo $SESSION_ID::$XNAT_USER::$XNAT_PASS::$XNAT_HOST::${zoneV_min_z}::${zoneV_max_z} 
    echo zoneV_max_z::${zoneV_max_z} 
    /software/compartment_separation_with_vent_boundgiven_specific_nifit.sh $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST ${zoneV_min_z} ${zoneV_max_z} ${counter_end}  /input /output
 echo " I AM AT compartment_separation_with_vent_boundgiven_specific_nifit.sh"

    # echo "$SESSION_ID,$SESSION_NAME" >> ${list_accomplished}
  fi 
    count=$((count+1))
    # echo "THIS COUNT NUMBER IS "::${count}::${counter_end}
#     fi
    # if [ ${count} -ge ${counter_end+1} ]; then
    # sleep 10
    # break
    # fi
done < <(tail -n +2 "${sessions_list}")

