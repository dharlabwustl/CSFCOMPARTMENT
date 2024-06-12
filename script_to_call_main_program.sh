#!/bin/bash 

SESSION_ID=${1}
XNAT_USER=${2}
XNAT_PASS=${3}
TYPE_OF_PROGRAM=${4}
export XNAT_HOST=${5}
input=$XNAT_HOST ##"one::two::three::four"
# Check if '::' is present
if echo "$input" | grep -q "+"; then
  # Set the delimiter
  IFS='+'

  # Read the split words into an array
  read -ra ADDR <<< "$input"
  export XNAT_HOST=${ADDR[0]} 
  SUBTYPE_OF_PROGRAM=${ADDR[1]} 
else
export XNAT_HOST=${5} 
    echo "'+' is not present in the string"
fi


echo ${TYPE_OF_PROGRAM}::TYPE_OF_PROGRAM

export XNAT_USER=$XNAT_USER
export XNAT_PASS=$XNAT_PASS
echo XNAT_USER=$XNAT_USER
echo XNAT_PASS=$XNAT_PASS
echo "I am "
if [[ ${TYPE_OF_PROGRAM} == 1 ]] ;
then
echo " I AM IN TYPE_OF_PROGRAM == 1"
/software/compartment_separation.sh  $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST
fi
if [[ ${TYPE_OF_PROGRAM} == 2 ]] ;
then
echo " I AM IN TYPE_OF_PROGRAM == ${TYPE_OF_PROGRAM}"
/software/compartment_separation_v1point1_Oct13_2023.sh  $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST
fi
if [[ ${TYPE_OF_PROGRAM} == 'PROJECT_LEVEL' ]] ;
then
if [[ ${SUBTYPE_OF_PROGRAM} == 'PROJECT_LEVEL_CSF_COMPARTMENT' ]] ;
then
echo " I AM IN SUBTYPE_OF_PROGRAM == ${SUBTYPE_OF_PROGRAM}"
echo $SESSION_ID $XNAT_USER $XNAT_PASS "${ADDR[0]}" "${ADDR[2]}" "${ADDR[3]}"   "${ADDR[4]}" "${ADDR[5]}"  

# /software/project_level_csf_compartment_with_threshold.sh  $SESSION_ID $XNAT_USER $XNAT_PASS "${ADDR[0]}" "${ADDR[2]}" "${ADDR[3]}"   "${ADDR[4]}" "${ADDR[5]}"  
fi
fi
#
#if [[ ${TYPE_OF_PROGRAM} == 2 ]] ;
#then
#
#  PROJECT_ID=${1}
#    /software/combine_csvs_and_copy_pdfs_projectlevel_Jan9_2023.sh  ${PROJECT_ID} $XNAT_USER $XNAT_PASS $XNAT_HOST
#fi
#if [[ ${TYPE_OF_PROGRAM} == 3 ]] ;
#then
#
#  PROJECT_ID=${1}
#echo ${PROJECT_ID}::$XNAT_USER::$XNAT_PASS::$XNAT_HOST
#   /software/analyzed_session_list.sh  ${PROJECT_ID} $XNAT_USER $XNAT_PASS $XNAT_HOST
#fi
##if [[ ${TYPE_OF_PROGRAM} == 2 ]] ;
##then
##    /software/nwucalculation_session_level_allsteps_November14_2022.sh $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST /input /output
##fi
##if [[ ${TYPE_OF_PROGRAM} == 1 ]] ;
##then
##    /software/dicom2nifti_call_sessionlevel_selected.sh  ${SESSION_ID} $XNAT_USER $XNAT_PASS $XNAT_HOST
##fi
##
##if [[ ${TYPE_OF_PROGRAM} == 3 ]] ;
##then
##  PROJECT_ID=${1}
##    /software/combine_csvs_and_copy_pdfs_projectlevel_Jan9_2023.sh  ${PROJECT_ID} $XNAT_USER $XNAT_PASS $XNAT_HOST
##fi
##
##if [[ ${TYPE_OF_PROGRAM} == 4 ]] ;
##then
##
##   /software/nwucalculation_scan_level_allsteps.sh
##fi
##
##
##if [[ ${TYPE_OF_PROGRAM} == 5 ]] ;
##then
##
##   /software/nwucalculation_onlocalcomp_aftersegJan172022.sh
##fi
##if [[ ${TYPE_OF_PROGRAM} == 6 ]] ;
##then
##
##   /software/combine_csvs_and_copy_pdfs_projectlevel_Jan17_2023_LocalComputer.sh
##fi
#
