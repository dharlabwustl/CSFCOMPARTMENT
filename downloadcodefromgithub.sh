#!/bin/bash
#export TORCH_HOME=/software
export SENTENCE_TRANSFORMERS_HOME=/software
cd /software/
echo ${4}
git_link=${4}
git clone ${git_link} #https://github.com/dharlabwustl/EDEMA_MARKERS_PROD.git
y=${git_link%.git}
git_dir=$(basename $y)
mv ${git_dir}/* /software/
chmod +x /software/*.sh 

SESSION_ID=${1}
XNAT_USER=${2}
XNAT_PASS=${3}
TYPE_OF_PROGRAM=${5}
export XNAT_HOST=${6}

/software/script_to_call_main_program.sh $SESSION_ID $XNAT_USER $XNAT_PASS ${TYPE_OF_PROGRAM} ${6}
