#!/bin/bash
#export XNAT_USER=${2}
#export XNAT_PASS=${3}
#export XNAT_HOST=${4}
#sessionID=${1}
#working_dir=/workinginput
#output_directory=/workingoutput
#
#final_output_directory=/outputinsidedocker
#function call_get_resourcefiles_metadata_saveascsv_args() {
#
#  local resource_dir=${2}   #"NIFTI"
#  local output_csvfile=${4} #{array[1]}
#
#  local URI=${1} #{array[0]}
#  #  local file_ext=${5}
#  #  local output_csvfile=${output_csvfile%.*}${resource_dir}.csv
#
#  local final_output_directory=${3}
#  local call_download_files_in_a_resource_in_a_session_arguments=('call_get_resourcefiles_metadata_saveascsv_args' ${URI} ${resource_dir} ${final_output_directory} ${output_csvfile})
#  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_files_in_a_resource_in_a_session_arguments[@]}")
#  echo " I AM AT call_get_resourcefiles_metadata_saveascsv_args"
#
#}
#echo " I AM RUNNING "
################# DOWNLOAD MASKS ###############################
### METADATA in the MASK directory
#URI=/data/experiments/${sessionID}
#resource_dir="NIFTI_LOCATION"
#output_csvfile=${sessionID}_SCANSELECTION_METADATA.csv
#call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
#dir_to_save=${working_dir}
#while IFS=',' read -ra array; do
#  #xx=0
#  #
#  ##if [ ${array[1]} == "SNIPR01_E00894" ]  ; then
##  echo "${array[6]}"
#  url=${array[6]}
#  filename=$(basename ${url})
#
#  #def call_download_a_singlefile_with_URIString(args):
#  #    url=args.stuff[1]
#  #    filename=args.stuff[2]
#  #    dir_to_save=args.stuff[3]
#  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url} ${filename} ${dir_to_save})
#  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#  while IFS=',' read -ra array1; do
##      echo "${array1[0]}"
#      url1=${array1[0]}
##      URI=/data/experiments/${sessionID}
#      resource_dir="MASKS"
#      output_csvfile=${sessionID}_SCANSELECTION_METADATA.csv
#      call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile}
##      filename1=$(basename ${url1})
##  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
##  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#  while IFS=',' read -ra array2; do
#
#      url2=${array2[6]}
#      if [[ ${url2} == *"_levelset"* ]]  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
#        echo "It's there!"
#        echo "${array2[6]}"
#        filename2=$(basename ${url2})
#        call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
#        outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#      fi
##      URI=/data/experiments/${sessionID}
##      resource_dir="MASKS"
##      output_csvfile=${sessionID}_SCANSELECTION_METADATA.csv
##      call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile}
##      filename1=$(basename ${url1})
##  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename1} ${dir_to_save})
##  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
#    done < <(tail -n +2 "${working_dir}/${output_csvfile}")
#
#
#    done < <(tail -n +2 "${dir_to_save}/${filename}")
#  #  echo "${array[5]}"
#  #if [ ${array[4]} == "xnat:ctSessionData" ] ; then
#  #    echo "${array[1]}"
#  #    echo "${array[5]}"
#  #call_fill_sniprsession_list_arguments=('call_fill_sniprsession_list' ${copy_session} ${array[1]} ) ##
#  ### ${working_dir}/${project_ID}_SNIPER_ANALYTICS.csv  ${project_ID} ${output_directory} )
#  #outputfiles_present=$(python3 fillmaster_session_list.py "${call_fill_sniprsession_list_arguments[@]}")
#  #call_creat_analytics_onesessionscanasID_arguments=('call_creat_analytics_onesessionscanasID' ${array[1]} ${array[5]} ${scan_analytics}  ${scan_analytics_nofilename})
#  #outputfiles_present=$(python3 fillmaster_session_list.py "${call_creat_analytics_onesessionscanasID_arguments[@]}")
#  ##def creat_analytics_onesessionscanasID(sessionId,sessionLabel,csvfilename,csvfilename_withoutfilename)
#  ##counter=$((counter + 1))
#  #fi
#  ##if [ $counter -eq 7 ] ; then
#  ##  break
#  ##fi
#done < <(tail -n +2 "${working_dir}/${output_csvfile}")

# single filename NECT, its CSF mask and other relevant files
rm /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/error.txt
python /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/CSFCOMPARTMENT/CSF_COMPARTMENT_GITHUB_July212023.py

#pdflatex /media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/RESULTS/test.tex
