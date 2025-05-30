#!/bin/bash
#echo " I AM AT COMPARTMENT_SEPARATION_WITH_VENT_BOUNDGIVEN.sh"
export XNAT_USER=${2}
export XNAT_PASS=${3}
export XNAT_HOST=${4}
#zoneV_min_z=${5}
#zoneV_max_z=${6}
sessionID=${1}
working_dir=/workinginput
output_directory=/workingoutput

final_output_directory=/outputinsidedocker
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
URI=/data/experiments/${sessionID}
resource_dir="NIFTI_LOCATION"
output_csvfile=${sessionID}_SCANSELECTION_METADATA.csv
call_get_resourcefiles_metadata_saveascsv_args ${URI} ${resource_dir} ${working_dir} ${output_csvfile}
dir_to_save=${working_dir}
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

  while IFS=',' read -ra array1; do
    #      echo "${array1[0]}"
    url1=${array1[0]}
    #      URI=/data/experiments/${sessionID}
    resource_dir="MASKS"
    output_csvfile_1=${sessionID}_MASK_METADATA.csv
    call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir} ${output_csvfile_1}
    #      filename1=$(basename ${url1})
    #  call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url1} ${filename1} ${dir_to_save})
    #  outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")



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


    resource_dir="SAH_SEGM"
    working_dir_1="/input"
    output_csvfile_2=${sessionID}_MASK_METADATA.csv
    call_get_resourcefiles_metadata_saveascsv_args ${url1} ${resource_dir} ${working_dir_1} ${output_csvfile_2}
    while IFS=',' read -ra array3; do

      url3=${array3[6]}
            if [[ ${url3} == *"resaved_4DL_seg_ventri.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
              echo "It's there!"
              echo "${array3[6]}"
              filename3=$(basename ${url3})
              call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url3} ${filename3} ${working_dir_1})
              outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
              greyfile1=${working_dir_1}/${filename3}
              echo "${greyfile1}"
            fi
            if [[ ${url3} == *"resaved_4DL_seg_total.nii.gz"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
              echo "It's there!"
              echo "${array3[6]}"
              filename4=$(basename ${url3})
              call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url3} ${filename4} ${working_dir_1})
              outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
              sah_total=${working_dir_1}/${filename4}
              echo "${sah_total}"
            fi

    done < <(tail -n +2 "${working_dir_1}/${output_csvfile_2}")
#############################################
    ##############################################
        resource_dir="PREPROCESS_SEGM"
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
          if [[ ${url2} == *"warped_1_"* ]]; then #  || [[ ${url2} == *"_levelset_bet"* ]]  || [[ ${url2} == *"csf_unet"* ]]  ; then ##[[ $string == *"My long"* ]]; then
            echo "It's there!"
            echo "${array2[6]}"
            filename2=$(basename ${url2})
            call_download_a_singlefile_with_URIString_arguments=('call_download_a_singlefile_with_URIString' ${url2} ${filename2} ${dir_to_save})
            outputfiles_present=$(python3 download_with_session_ID.py "${call_download_a_singlefile_with_URIString_arguments[@]}")
            venticle_only_mask=${dir_to_save}/${filename2}
            echo "${venticle_only_mask}"
          fi
        done < <(tail -n +2 "${working_dir}/${output_csvfile_2}")
#        venticle_only_mask=${betfile}
#        echo "${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}"
#        python3 findventriclemaskconvexhull10112024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
        ventricleboundfile=${dir_to_save}/'ventricle_bounds.csv'
        python3 findventriclemaskobb_10102024.py  ${venticle_only_mask} ${csffile} ${dir_to_save} ${greyfile} ${betfile}
#

        ventricle_obb_mask=${dir_to_save}/ventricle_obb_mask.nii
        while IFS=',' read -ra array3; do
          echo "${array3[3]}::${array3[4]}"
          zoneV_min_z=${array3[3]}
          zoneV_max_z=${array3[4]}
        done < <(tail -n +2 "${ventricleboundfile}")
#              zoneV_min_z=0 #${array3[3]}
#              zoneV_max_z=60 #${array3[4]}
    #############################################
    ######################################
    rm /workinginput/*_resaved_levelset_sulci_total.nii.gz
    rm /workinginput/*_resaved_levelset_ventricle_total.nii.gz
    rm /workinginput/*_resaved_levelset_sulci_below_ventricle.nii.gz
    rm /workinginput/*_resaved_levelset_sulci_above_ventricle.nii.gz
    rm /workinginput/*_resaved_levelset_sulci_at_ventricle.nii.gz
#    rm  /workinginput/ventricle_obb_mask.nii
#    rm  workinginput/ventricle_contour.nii
    echo "call_csf_compartments_arguments=('call_csf_compartments_vent_obb_given' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )"
    call_csf_compartments_arguments=('call_csf_compartments_vent_obb_given' ${greyfile} ${csffile} ${betfile} ${ventricle_obb_mask} ${zoneV_min_z} ${zoneV_max_z} )
    outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
    #  echo ${outputfiles_present}
    #fi
    call_csf_compartments_arguments=('call_combine_sah_to_csf' ${sah_total} ${csffile}  )
    outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}")
    echo ${outputfiles_present}
    URI_1=${url2%/resource*}
    filename_prefix=$(basename ${url}) #${url2%/resource*} #filename=
    filename_prefix=${filename_prefix%_NIFTILOCATION*}
    resource_dirname="MASKS"
    this_data_basename=$(basename ${greyfile})
    this_data_basename_noext=${this_data_basename%_resaved*}
    for file_name in ${dir_to_save}/${filename_prefix}*.nii.gz; do
      echo ${file_name}
      if [[ ${file_name} == *"${this_data_basename_noext}"* ]] || [[ ${file_name} == *"ventricle"* ]] || [[ ${file_name} == *"sulci"* ]]; then
        call_uploadsinglefile_with_URI_arguments=('call_uploadsinglefile_with_URI' ${URI_1} ${file_name} ${resource_dirname})
        outputfiles_present=$(python3 /software/download_with_session_ID.py "${call_uploadsinglefile_with_URI_arguments[@]}")
        echo ${outputfiles_present}
      fi
    done
    file_name=$(ls ${working_dir}/${this_data_basename_noext}*_resaved_csf_unet_with_sah.nii.gz)
    echo " ${URI_1} ${file_name} ${resource_dirname})"
    call_uploadsinglefile_with_URI_arguments=('call_uploadsinglefile_with_URI' ${URI_1} ${file_name} ${resource_dirname})
    outputfiles_present=$(python3 /software/download_with_session_ID.py "${call_uploadsinglefile_with_URI_arguments[@]}")
    echo ${outputfiles_present}
  done < <(tail -n +2 "${dir_to_save}/${filename}")

done \
  < \
  <(tail -n +2 "${working_dir}/${output_csvfile}")

## single filename NECT, its CSF mask and other relevant files
#rm /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/error.txt
#greyfile='/workinginput/SAH_1_01052014_2003_2_resaved_levelset.nii.gz'
### '/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset.nii.gz'
#betfile='/workinginput/SAH_1_01052014_2003_2_resaved_levelset_bet.nii.gz'
##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_levelset_bet.nii.gz'
#csffile='/workinginput/SAH_1_01052014_2003_2_resaved_csf_unet.nii.gz'
##'/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/Krak_003_09042014_0949_MOZG_6.0_H31s_final_seg.nii.gz'
#
#call_csf_compartments_arguments=('call_csf_compartments' ${greyfile} ${csffile} ${betfile} )
#outputfiles_present=$(python3 /software/CSF_COMPARTMENT_GITHUB_July212023.py "${call_csf_compartments_arguments[@]}" )
#echo ${outputfiles_present}
###python /media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/CSFCOMPARTMENT/CSF_COMPARTMENT_v1_part2_July18_2023.py
##
###pdflatex /media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/SAH_N_CSF_Compartment/RESULTS/test.tex
