#!/bin/bash
../CSFCOMPARTMENT/compartment_separation.sh
## docker build -t sharmaatul11/yashengcsfinfarctseg1 .
## docker push  sharmaatul11/yashengcsfinfarctseg1
## docker run -v $PWD/workinginput:/workinginput -v $PWD/workingoutput:/workingoutput -v $PWD/ZIPFILEDIR:/ZIPFILEDIR -v$PWD/output:/output  -it sharmaatul11/yashengcsfinfarctseg1  /Stroke_CT_Processing/call_preproc_segm_session_level_1.sh SNIPR_E03523
## docker build -t sharmaatul11/dicom2nifti_nwu_1 .
#prefix_dir1=/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/LUNGS/PYCHARM/atelectasis
#cd ${prefix_dir1}
#imagename=atelectasissegmentation2
##./builddockerimage.sh ${imagename} ${prefix_dir1}
#prefix_dir='/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/LUNGS/PYCHARM/TEST_ATELECTASIS'
#
#workinginput=$prefix_dir/workinginput
#workingoutput=$prefix_dir/workingoutput
#ZIPFILEDIR=$prefix_dir/ZIPFILEDIR
#outputinsidedocker=$prefix_dir/outputinsidedocker
#software=$prefix_dir/software
#calculation=$prefix_dir/calculation
#latex=$prefix_dir/latex
#images=$prefix_dir/images
#running_dir='/media/atul/WDJan2022/WASHU_WORKS/PROJECTS/DOCKERIZE/LUNGS/PYCHARM/TEST_ATELECTASIS'
#cd ${running_dir}
#mkdir $workingoutput
#mkdir $workinginput
#mkdir $ZIPFILEDIR
#mkdir $outputinsidedocker
#mkdir $software
#mkdir $calculation
#mkdir $latex
#mkdir $images
#SESSION_ID=SNIPR02_E06024 #SNIPR02_E04970 ##'LUNGATELECTASIS' #TEST ##SNIPR02_E04970 ##SNIPR_E03614 #SNIPR_E03516
#XNAT_PASS='Mrityor1!'
#XNAT_USER=atulkumar
#XNAT_HOST='https://snipr.wustl.edu'
#script_number=1 ##3 #3 #2 #3 #2 ##1 #3 #5
#rm -r $workingoutput/*
#rm -r $workinginput/*
#rm -r $ZIPFILEDIR/*
#rm -r $outputinsidedocker/*
#rm -r $software/*
#rm -r $calculation/*
#rm -r $images/*
#rm -r $latex/*
#
#rm -r $software/*
#
#docker_image=registry.nrg.wustl.edu/docker/nrg-repo/sharmaatul11/${imagename}
#docker run --gpus all -v $images:/images  -v $latex:/latex  -v $calculation:/calculation  -v $software:/software -v $workinginput:/workinginput -v $workingoutput:/workingoutput -v $ZIPFILEDIR:/ZIPFILEDIR -v $outputinsidedocker:/outputinsidedocker  -it  ${docker_image}   /callfromgithub/downloadcodefromgithub.sh $SESSION_ID $XNAT_USER $XNAT_PASS https://github.com/dharlabwustl/ATELECTASIS.git ${script_number}  https://snipr.wustl.edu
#
##docker run --gpus all  -v $PWD/workingoutput:/workingoutput  -v $PWD/outputinsidedocker:/outputinsidedocker -v $PWD/workinginput:/workinginput -v $PWD/software:/software -v $PWD/NIFTIFILEDIR:/NIFTIFILEDIR  -v $PWD/DICOMFILEDIR:/DICOMFILEDIR  -v $PWD/working:/working -v $PWD/input:/input -v $PWD/ZIPFILEDIR:/ZIPFILEDIR -v $PWD/output:/output  -it  registry.nrg.wustl.edu/docker/nrg-repo/sharmaatul11/${imagename}  /callfromgithub/downloadcodefromgithub.sh $SESSION_ID $XNAT_USER $XNAT_PASS https://github.com/dharlabwustl/EDEMA_MARKERS_PROD.git $script_number  https://snipr.wustl.edu    ###### /software/dicom2nifti_call_sessionlevel_selected.sh ${sessionID} $XNAT_USER $XNAT_PASS $XNAT_HOST #https://snipr-dev-test1.nrg.wustl.edu

