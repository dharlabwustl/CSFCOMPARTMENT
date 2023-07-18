#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Timporthu Mar 12 15:40:53 2020

@author: atul
"""
import subprocess,os,sys,glob,datetime
import os,csv
import glob,re
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
from skimage import exposure
import pandas as pd
import smtplib
def latex_start(filename):
    file1 = open(filename,"w")
    file1.writelines("\\documentclass{article}\n")
    file1.writelines("\\usepackage[margin=0.5in]{geometry}\n")
    file1.writelines("\\usepackage{graphicx}\n")   
    file1.writelines("\\usepackage[T1]{fontenc} \n") 
    file1.writelines("\\usepackage{datetime} \n") 

#    file1.writelines("\\begin{document}\n")
    return file1
def latex_end(filename):
    file1 = open(filename,"a")
    file1.writelines("\\end{document}\n")
    file1.close()
    return "X"
def latex_begin_document(filename):
    file1 = open(filename,"a")
    file1.writelines("\\begin{document}\n")
    return file1
def latex_insert_line(filename,text="ATUL KUMAR"):
    command= "sed -i 's#\\end{document}##'   " +  filename
    subprocess.call(command,shell=True)
    file1 = open(filename,"a")
    currentDT = str(datetime.datetime.now())
    file1.writelines("\\section{" + currentDT + "}" ) #\\today : \\currenttime}")
#    file1.writelines("\\date{\\today}")
    file1.writelines("\\detokenize{")
    file1.writelines(text)
    file1.writelines("\n")
    file1.writelines("}")
    return file1
def latex_insert_line_nodek(filename,text="ATUL KUMAR"):
    command= "sed -i 's#\\end{document}##'   " +  filename
    subprocess.call(command,shell=True)
    file1 = open(filename,"a")
    file1.writelines(text)
    file1.writelines("\n")

    return file1
def latex_insert_line_nodate(filename,text="ATUL KUMAR"):
    command= "sed -i 's#\\end{document}##'   " +  filename
    subprocess.call(command,shell=True)
    file1 = open(filename,"a")
#    currentDT = str(datetime.datetime.now())
#    file1.writelines("\\section{" + currentDT + "}" ) #\\today : \\currenttime}")
#    file1.writelines("\\date{\\today}")
    file1.writelines("\\detokenize{")
    file1.writelines(text)
    file1.writelines("\n")
    file1.writelines("}")
    return file1
def writetolabnotebook(labnotebook,text):
#    latex_start(labnotebook)
#    latex_begin_document(labnotebook)
    latex_insert_line(labnotebook,text)
    latex_end(labnotebook)
def writetoanewlabnotebook(labnotebook):
    latex_start(labnotebook)
    latex_begin_document(labnotebook)
    latex_insert_line(labnotebook,"THIS IS A NEW LABNOTEBOOK")
    latex_end(labnotebook)
#def write_latex_body(labnotebook,text="HELLO WORLD"):
#    command= "sed -i 's#\\end{document}##'   " +  labnotebook
#    subprocess.call(command,shell=True)
#    command= "echo  " + "\\\section >>   " +  labnotebook
#    subprocess.call(command,shell=True)
#    command= 'echo $(date) >>   ' +  labnotebook
#    subprocess.call(command,shell=True)
#    command= 'echo  ' + text  +  '>>   ' +  labnotebook
#    subprocess.call(command,shell=True)
#    return "x"
#
#def write_latex_end(labnotebook):
#    command= "echo  " + "\\\end{document}  >>  " +  labnotebook
#    subprocess.call(command,shell=True)
#    return "x"

def combinecsvs(inputdirectory,outputdirectory,outputfilename):
    outputfilepath=os.path.join(outputdirectory,outputfilename)
    extension = 'csv'
    all_filenames = [i for i in glob.glob(os.path.join(inputdirectory,'*.{}'.format(extension)))]
#    os.chdir(inputdirectory)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv(outputfilepath, index=False, encoding='utf-8-sig')

def combinecsvs_sh():
    inputdirectory=sys.argv[1]
    outputdirectory=sys.argv[2]
    outputfilename=sys.argv[3]
    outputfilepath=os.path.join(outputdirectory,outputfilename)
    extension = 'csv'
    all_filenames = [i for i in glob.glob(os.path.join(inputdirectory,'*.{}'.format(extension)))]
#    os.chdir(inputdirectory)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv(outputfilepath, index=False, encoding='utf-8-sig')

  
def write_csv(csv_file_name,csv_columns,data_csv):
    try:
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in data_csv:
                print("data")
                print(data)
                writer.writerow(data)
    except IOError:
        print("I/O error")
        
def diff_two_csv(file1,file2,outputfile="diff.csv"):
    
    
    return "XX"

def write_tex_im_in_afolder(foldername,max_num_img,img_ext="*.png"):
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername,img_ext))
    counter=0
    for each_png_file in png_files:
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=each_png_file,caption= thisfilebasename.split('.png'),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.png')[0] )
            counter=counter+1
    latex_end(latexfilename)
#    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
#    subprocess.call(command,shell=True)    

def filename_replace_dots(foldername,img_ext):
    files=glob.glob(os.path.join(foldername,"*"+img_ext))
    for each_file in files:
        each_f_basename=os.path.basename(each_file)
        each_f_basename_wo_ext=each_f_basename.split(img_ext)
        each_f_basename_wo_extNew= each_f_basename_wo_ext[0].replace(".","_") #re.sub('[^a-zA-Z0-9 \n\.]', '_', each_f_basename_wo_ext[0])
        each_f_basename_new=each_f_basename_wo_extNew + img_ext
        each_f_basename_new_w_path=os.path.join(foldername,each_f_basename_new)
        command = "mv   "  + each_file  + "   " + each_f_basename_new_w_path
        print(each_file)
#        print()
        subprocess.call(command,shell=True)
def filename_replace_dots1(foldername,img_ext):
    files=glob.glob(os.path.join(foldername,"*_"+img_ext))
    for each_file in files:
        each_f_basename=os.path.basename(each_file)
        each_f_basename_wo_ext=each_f_basename.split("_"+img_ext)
        each_f_basename_wo_extNew= each_f_basename_wo_ext[0] +"." + img_ext #.replace(".","_") #re.sub('[^a-zA-Z0-9 \n\.]', '_', each_f_basename_wo_ext[0])
        each_f_basename_new=each_f_basename_wo_extNew # + img_ext
        each_f_basename_new_w_path=os.path.join(foldername,each_f_basename_new)
        command = "mv   "  + each_file  + "   " + each_f_basename_new_w_path
        print(each_f_basename_new)
#        print()
        subprocess.call(command,shell=True)
    
   
    
    
    
def write_tex_im_in_3folders(foldername1,foldername2,foldername3,max_num_img,extension=".png"):
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    foldername1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/GaborOnly"
    foldername2="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly"
    foldername3="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegnGabor"
    foldername=os.path.join(os.path.dirname(foldername1),os.path.basename(foldername1)+os.path.basename(foldername2)+os.path.basename(foldername3))
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername1,"*" + extension )) #.png"))
    counter=0
#    max_num_img=5

    for each_png_file in png_files:
        if counter < max_num_img:
            images=[]
            thisfilebasename=os.path.basename(each_png_file)
            path2=os.path.join(foldername2,thisfilebasename)
            path3=os.path.join(foldername3,thisfilebasename)
            if os.path.exists(path2) and os.path.exists(path3):
                images.append(each_png_file)
                images.append(path2)
                images.append(path2)
                latex_start_tableNc(latexfilename,3)
                images.append(each_png_file)
                latex_insertimage_tableNc(latexfilename,images,3,caption= thisfilebasename.split(extension),imagescale=0.3)
                latex_end_table2c(latexfilename)
                latex_insert_line_nodate(latexfilename, thisfilebasename.split(extension)[0] )
                counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)  

def write_tex_im_in_3folders_(foldername1,foldername2,foldername3,max_num_img,extension=".png"):
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    foldername1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/GaborOnly"
    foldername2="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly"
    foldername3="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegnGabor"
    foldername=os.path.join(os.path.dirname(foldername1),os.path.basename(foldername1)+os.path.basename(foldername2)+os.path.basename(foldername3))
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=sorted(glob.glob(os.path.join(foldername1,"*GTvsGABOR*" + extension ))) #.png"))
    counter=0
#    max_num_img=5

    for each_png_file in png_files:
        if counter < max_num_img:
            images=[]
            thisfilebasename=os.path.basename(each_png_file)
            numberofslice=thisfilebasename.split("GTvsGABOR")[1].split(".jpg")[0]
            secondfile=thisfilebasename.split("GTvs")[0]+"GTvsRegist" + str(numberofslice) + ".jpg"
            thirddfile=thisfilebasename.split("GTvs")[0]+"GTvsGaborNRegist" + str(numberofslice) + ".jpg"
            path2=os.path.join(foldername2,secondfile)
            path3=os.path.join(foldername3,thirddfile)
            if os.path.exists(path2) and os.path.exists(path3):
                images.append(each_png_file)
                images.append(path2)
                images.append(path3)
                latex_start_tableNc(latexfilename,3)
                images.append(each_png_file)
                latex_insertimage_tableNc(latexfilename,images,3,caption= thisfilebasename.split(extension),imagescale=0.3)
                latex_end_table2c(latexfilename)
                latex_insert_line_nodate(latexfilename, thisfilebasename.split(extension)[0] )
                counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)  

def write_tex_im_in_3folders_sh():
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    foldername1=sys.argv[1] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/GaborOnly"
    foldername2=sys.argv[2] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegistrationOnly"
    foldername3=sys.argv[3] #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/RESULTS/RegnGabor"
    foldername=os.path.join(os.path.dirname(foldername1),os.path.basename(foldername1)+os.path.basename(foldername2)+os.path.basename(foldername3))
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername1,"*.png"))
    counter=0
    max_num_img=int(sys.argv[4])

    for each_png_file in png_files:
        if counter < max_num_img:
            images=[]
            thisfilebasename=os.path.basename(each_png_file)
            path2=os.path.join(foldername2,thisfilebasename)
            path3=os.path.join(foldername3,thisfilebasename)
            if os.path.exists(path2) and os.path.exists(path3):
                images.append(each_png_file)
                images.append(path2)
                images.append(path3)
                latex_start_tableNc(latexfilename,3)
                images.append(each_png_file)
                latex_insertimage_tableNc(latexfilename,images,3,caption= thisfilebasename.split('.png'),imagescale=0.3)
                latex_end_table2c(latexfilename)
                latex_insert_line_nodate(latexfilename, thisfilebasename.split('.png')[0] )
                counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)     
def write_tex_im_in_afolder_sh():
    foldername=sys.argv[1]
    max_num_img=int(sys.argv[2])
    # get the folder name
#    foldername="" # complete path
    # start writing tex file
    latexfilename=foldername+".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername,"*.png"))
    png_files=png_files.sort(key=os.path.getmtime)
    counter=0
    for each_png_file in png_files:
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=each_png_file,caption= thisfilebasename.split('.png'),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.png')[0] )
            counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)   
    
def write_tex_im_in_afolder_py(foldername,max_num_img=200,fileext="png"):

    # get the folder name
#    foldername="" # complete path
    # start writing tex file

    # for each image file in the folder insert text to include the image a figure
    png_files=glob.glob(os.path.join(foldername,"*."+ fileext))
    png_files.sort(key=os.path.getmtime)
    counter=0
    filecount=0
    latexfilename=foldername+ str(filecount) + ".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    for each_png_file in png_files:
        if counter%1000 == 0:
            latex_end(latexfilename)
            filecount=filecount+1
            latexfilename=foldername+ str(filecount) + ".tex"
            latex_start(latexfilename)
            latex_begin_document(latexfilename)
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            thisfilebasename_S=thisfilebasename.split("."+ fileext)
            thisfilebasename_S=thisfilebasename_S[0].replace(".","_")
            thisfilebasenameNPath=os.path.join(foldername,thisfilebasename_S+"."+ fileext)
            command= "mv   " + each_png_file +  "   " + thisfilebasenameNPath
            subprocess.call(command,shell=True)
            thisfilebasename=os.path.basename(thisfilebasenameNPath)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=thisfilebasenameNPath,caption= thisfilebasename.split('.' + fileext),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.' + fileext)[0] )
            counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)

def write_tex_im_in_afolder_v1(foldername,max_num_img=200,filenamepattern=".png"):

    # get the folder name
#    foldername="" # complete path
    # start writing tex file

    # for each image file in the folder insert text to include the image a figure
    fileext=filenamepattern.split(".")[1]
    png_files=glob.glob(os.path.join(foldername,"*"+ filenamepattern))
    png_files.sort(key=os.path.getmtime)
    counter=0
    filecount=0
    latexfilename=foldername+ str(filecount) + ".tex"
    latex_start(latexfilename)
    latex_begin_document(latexfilename)
    for each_png_file in png_files:
        if counter%1000 == 0:
            latex_end(latexfilename)
            filecount=filecount+1
            latexfilename=foldername+ str(filecount) + ".tex"
            latex_start(latexfilename)
            latex_begin_document(latexfilename)
        if counter < max_num_img:
            thisfilebasename=os.path.basename(each_png_file)
            thisfilebasename_S=thisfilebasename.split("."+ fileext)
            thisfilebasename_S=thisfilebasename_S[0].replace(".","_")
            thisfilebasenameNPath=os.path.join(foldername,thisfilebasename_S+"."+ fileext)
            command= "mv   " + each_png_file +  "   " + thisfilebasenameNPath
            subprocess.call(command,shell=True)
            thisfilebasename=os.path.basename(thisfilebasenameNPath)
            latex_start_table1c(latexfilename)
            latex_insertimage_table1c(latexfilename,image1=thisfilebasenameNPath,caption= thisfilebasename.split('.' + fileext),imagescale=0.3)
            latex_end_table2c(latexfilename)
            latex_insert_line_nodate(latexfilename, thisfilebasename.split('.' + fileext)[0] )
            counter=counter+1
    latex_end(latexfilename)
    command= "mv  " + latexfilename.split('.')[0] + "*     " + os.path.dirname(foldername)
    subprocess.call(command,shell=True)
   
def latex_start_table2c(filename):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    file1.writelines("\\begin{tabular}{ c c  }\n")
    return file1
def latex_start_tableNc(filename,N):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    texttowrite=""
    for x in range(N):
        texttowrite = texttowrite + "  " + "c" + " " 
    file1.writelines("\\begin{tabular}{ " + texttowrite + "  }\n")
    return file1
def latex_start_table1c(filename):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    file1.writelines("\\begin{tabular}{ c  }\n")
    return file1

def latex_end_table2c(filename):
    file1 = open(filename,"a")
    file1.writelines("\n")
    file1.writelines("\\end{tabular}\n")
    file1.writelines("\\end{center}\n")
    return file1

def latex_insertimage_table2c(filename,image1="lion.jpg", image2="lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{" + image1 + "}\n")
    file1.writelines("&")
    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")

    return file1
def latex_insertimage_tableNc(filename,images,N, caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    for x in range(N):
        if x < N-1:
            file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{" + images[x] + "}\n")
            file1.writelines("&")
        else:
            file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{" + images[x] + "}\n")
            
#    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{"+  image2 + "}\n")

    return file1

def latex_insertimage_table1c(filename,image1="lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\includegraphics[width=" + str(imagescale) + "\\textwidth]{" + image1 + "}\n")
    return file1
def latex_inserttext_table2c(filename,text1="lion.jpg", text2="lion.jpg"):
    file1 = open(filename,"a")
    file1.writelines(text1)
    file1.writelines("&")
    file1.writelines(text2)
    
def latex_inserttext_table1c(filename,text1="lion.jpg"):
    file1 = open(filename,"a")
    file1.writelines(text1)

    return file1    
def saveslicesofnifti(filename,savetodir=""):
    filename_nib=nib.load(filename)
    filename_gray_data_np=filename_nib.get_fdata()
    min_img_gray=np.min(filename_gray_data_np)
    img_gray_data=0
    if not os.path.exists(savetodir):
        savetodir=os.path.dirname(filename)
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    for x in range(img_gray_data.shape[2]):
        cv2.imwrite(os.path.join(savetodir,os.path.basename(filename).split(".nii")[0]+str(x)+".jpg" ),img_gray_data[:,:,x]*255 )
    
def savesingleslicesofnifti(filename,slicenumber=0,savetodir=""):
    filename_nib=nib.load(filename)
    filename_gray_data_np=filename_nib.get_fdata()
    min_img_gray=np.min(filename_gray_data_np)
    img_gray_data=0
    if not os.path.exists(savetodir):
        savetodir=os.path.dirname(filename)
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
#    for x in range(img_gray_data.shape[2]):
    x=slicenumber
    filenamejpg=os.path.join(savetodir,os.path.basename(filename).split(".nii")[0]+str(x)+".jpg" )
    cv2.imwrite(filenamejpg,img_gray_data[:,:,x]*255 )
    return filenamejpg

def sas7bdatTOcsv(inputfilename,outputfilename=""):
    if len(outputfilename)==0:
        outputfilename=inputfilename.split(".sas7bdat")[0] + ".csv"
#    inputfilename="/home/atul/Downloads/dra.sas7bdat"
#    outputfilename="/home/atul/Downloads/dra.csv"
    inputdataframe=pd.read_sas(inputfilename, format = 'sas7bdat', encoding="latin-1")
    inputdataframe.to_csv(outputfilename, index=False)
    
def print_number_slices(inputdirectory):
    return "X"
    
def contrast_stretch(img,threshold_id):
    if threshold_id==1:
        ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(0, 200))
    if threshold_id==2:
        ct_image=exposure.rescale_intensity(img.get_fdata() , in_range=(1000, 1200))        
    return ct_image
def contrast_stretch_np(img,threshold_id):
    if threshold_id==1:
        ct_image=exposure.rescale_intensity(img , in_range=(0, 200))
    if threshold_id==2:
        ct_image=exposure.rescale_intensity(img, in_range=(1000, 1200))        
    return ct_image
def saveslicesofnumpy3D(img_gray_data,savefilename="",savetodir=""):
##    filename_nib=nib.load(filename)
##    filename_gray_data_np=filename_nib.get_fdata()
#    min_img_gray=np.min(filename_gray_data_np)
#    img_gray_data=0
#    if not os.path.exists(savetodir):
#        savetodir=os.path.dirname(filename)
#    if min_img_gray>=0:
#        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200)) 
#    else:
#        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    for x in range(img_gray_data.shape[2]):
        slice_num="{0:0=3d}".format(x)
        cv2.imwrite(os.path.join(savetodir,os.path.basename(savefilename).split(".nii")[0]+str(slice_num)+".png" ),img_gray_data[:,:,x] )
  
def saveslicesofnumpy3D_non_zero(img_gray_data,savefilename="",savetodir=""):
##    filename_nib=nib.load(filename)
##    filename_gray_data_np=filename_nib.get_fdata()
#    min_img_gray=np.min(filename_gray_data_np)
#    img_gray_data=0
#    if not os.path.exists(savetodir):
#        savetodir=os.path.dirname(filename)
#    if min_img_gray>=0:
#        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(1000, 1200)) 
#    else:
#        img_gray_data=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    for x in range(img_gray_data.shape[2]):
        if np.sum(img_gray_data[:,:,x]>0):
            slice_num="{0:0=3d}".format(x)
            cv2.imwrite(os.path.join(savetodir,os.path.basename(savefilename).split(".nii")[0]+str(slice_num)+".png" ),img_gray_data[:,:,x] )
  


#def tex_for_each_subject():
    # find unique CT names:
    
    #create a tex file:
    # for each file of a CT:
    # find the files with that name arranged in ascending order of time:
    
    # write those images into the tex file
    
    # come out of for loop and close the latex file
    
    
    
def send_email():
    
    gmail_user ='atulkumar@wustl.edu' # 'booktonotesbtn@gmail.com'
    gmail_password ='Pushti1!' # 'AtulAtul1!@#$'
    
    sent_from = gmail_user
    to = ['sharmaatul11@gmail.com','atulkumar@wustl.edu']
    subject = 'Program execution'
    body = "Your program has completed its task"
    
    email_text = 'Subject: {}\n\n{}'.format(subject, body) #body
#    """\
#    From: %s
#    To: %s
#    Subject: %s
#    
#    %s
#    """ % (sent_from, ", ".join(to), subject, body)
    
    try:
        server = smtplib.SMTP_SSL('mail.smtp2go.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
    
        print('Email sent!')
    except:
        print('Something went wrong...')

def send_email_sh():
    gmail_user ='atulkumar@wustl.edu' # 'booktonotesbtn@gmail.com'
    gmail_password ='Pushti1!' # 'AtulAtul1!@#$'
    sent_from = gmail_user
    to = [sys.argv[1],'atulkumar@wustl.edu']
#    subject = sys.argv[2]
    body = sys.argv[2]
    email_text = body
    
    try:
        server = smtplib.SMTP_SSL('mail.smtp2go.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
    
        print('Email sent!')
    except:
        print('Something went wrong...')   
    
    
def heatmap_intensities():
    image_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/FU_CTs_Masks/CTs/WUSTL_660_05062018_1027_gray.nii.gz"
    csf_mask_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/FU_CTs_Masks/CTs/Masks/WUSTL_660_05062018_1027_final_seg.nii.gz" 
    
    infarct_mask_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/FU_CTs_Masks/CTs/Masks/WUSTL_660_05062018_1027_INFARCT.hdr"
    
    image_nib=nib.load(image_file)
    image_nib_original=image_nib.get_fdata()
#    image_nib_original_copy=np.copy(image_nib_original)
    image_nib_original_copy=contrast_stretch(image_nib,1)*255
    image_matx=contrast_stretch(image_nib,1)
    csf_mask_matx=nib.load(csf_mask_file).get_fdata()
    infarct_mask_matx=nib.AnalyzeImage.from_filename(infarct_mask_file).get_fdata()
    threshold_range=[0,15,37]
    for i in range(image_matx.shape[2]):
        slice_3_layer= np.zeros([image_matx.shape[0],image_matx.shape[1],3])
        slice_3_layer_mask= np.zeros([csf_mask_matx.shape[0],csf_mask_matx.shape[1],3])
        slice_3_layer_infarct_mask= np.zeros([infarct_mask_matx.shape[0],infarct_mask_matx.shape[1],3])
        slice_3_layer_original= np.zeros([image_nib_original.shape[0],image_nib_original.shape[1],3])
        
        slice_3_layer_coloredmask= np.zeros([image_nib_original.shape[0],image_nib_original.shape[1],3])
#        slice_3_layer_coloredmask[:,:,0]=image_nib_original[:,:,i]*0
#        slice_3_layer_coloredmask[:,:,1]=image_nib_original[:,:,i]*0
#        slice_3_layer_coloredmask[:,:,2]=image_nib_original[:,:,i]*0
        slice_3_layer_coloredmask[:,:,0][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]>=15)  & (image_nib_original[:,:,i]<=37) ]       = 0
        slice_3_layer_coloredmask[:,:,1][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]>=15)  & (image_nib_original[:,:,i]<=37) ]       = 0
        slice_3_layer_coloredmask[:,:,2][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]>=15)  & (image_nib_original[:,:,i]<=37) ]       = 200
        
        slice_3_layer_coloredmask[:,:,0][(infarct_mask_matx[:,:,i]>0) &  (image_nib_original[:,:,i]<15) ]       = 0
        slice_3_layer_coloredmask[:,:,1][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]<15) ]       = 200
        slice_3_layer_coloredmask[:,:,2][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]<15) ]       = 0
        
        slice_3_layer_coloredmask[:,:,0][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]>37)  ]       = 200
        slice_3_layer_coloredmask[:,:,1][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]>37)   ]       = 200
        slice_3_layer_coloredmask[:,:,2][(infarct_mask_matx[:,:,i]>0) & (image_nib_original[:,:,i]>37)   ]       = 200
        SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/RESULTS/TEST"
        imagefilename=os.path.basename(image_file).split(".nii")[0]
        imagename1=  os.path.join(SLICE_OUTPUT_DIRECTORY, imagefilename+"TEST"+ str(i)+".png" )
#        slice_3_layer[:,:,0]=image_nib_original[:,:,i]#*255
#        slice_3_layer[:,:,1]=image_nib_original[:,:,i]#*255
#        slice_3_layer[:,:,2]=image_nib_original[:,:,i]#*255
        
        imagename2=  os.path.join(SLICE_OUTPUT_DIRECTORY, imagefilename+"TEST1"+ str(i)+".png" )
        
        cv2.imwrite(imagename1,slice_3_layer_coloredmask)
        
        slice_3_layer[:,:,0]=image_matx[:,:,i]*255
        slice_3_layer[:,:,1]=image_matx[:,:,i]*255
        slice_3_layer[:,:,2]=image_matx[:,:,i]*255
        
        slice_3_layer_infarct_mask[:,:,0]=infarct_mask_matx[:,:,i]
        slice_3_layer_infarct_mask[:,:,1]=infarct_mask_matx[:,:,i]
        slice_3_layer_infarct_mask[:,:,2]=infarct_mask_matx[:,:,i]
        
        slice_3_layer[slice_3_layer_infarct_mask>1]=np.min(slice_3_layer)
        cv2.imwrite(imagename2,slice_3_layer)
        image1=cv2.imread(imagename1)
        image2=cv2.imread(imagename2)
        
        slice_3_layer_original_colored = image1+image2 # cv2.add(image1,image2)
        imagename3=  os.path.join(SLICE_OUTPUT_DIRECTORY, imagefilename+ str(i)+".png" )
        
        cv2.imwrite(imagename3,slice_3_layer_original_colored)
        
#        
#        slice_3_layer_original[:,:,0]=image_nib_original_copy[:,:,i]#*255
#        slice_3_layer_original[:,:,1]=image_nib_original_copy[:,:,i]#*255
#        slice_3_layer_original[:,:,2]=image_nib_original_copy[:,:,i]#*255
#        
##        slice_3_layer[:,:,0]=image_matx[:,:,i]#*255
##        slice_3_layer[:,:,1]=image_matx[:,:,i]#*255
##        slice_3_layer[:,:,2]=image_matx[:,:,i]#*255
#
#        slice_3_layer_mask[:,:,0]=csf_mask_matx[:,:,i]
#        slice_3_layer_mask[:,:,1]=csf_mask_matx[:,:,i]
#        slice_3_layer_mask[:,:,2]=csf_mask_matx[:,:,i]
#                
#        slice_3_layer_infarct_mask[:,:,0]=infarct_mask_matx[:,:,i]
#        slice_3_layer_infarct_mask[:,:,1]=infarct_mask_matx[:,:,i]
#        slice_3_layer_infarct_mask[:,:,2]=infarct_mask_matx[:,:,i]
#        
#        slice_3_layer[:,:,0][slice_3_layer_mask[:,:,0]>0]=200
#        slice_3_layer[:,:,1][slice_3_layer_mask[:,:,1]>0]=0
#        slice_3_layer[:,:,2][slice_3_layer_mask[:,:,2]>0]=0
#        slice_3_layer[slice_3_layer_infarct_mask<1]=0 # np.min(slice_3_layer)
#        slice_3_layer_original[slice_3_layer_infarct_mask>1]=0 #np.min(slice_3_layer_original)
#        np.max(slice_3_layer)
##        slice_3_layer[:,:,0][(slice_3_layer[:,:,0]>0) & (slice_3_layer[:,:,0]<15)]=200
##        slice_3_layer[:,:,1][(slice_3_layer[:,:,1]>0) & (slice_3_layer[:,:,1]<15)]=200
#        
#        slice_3_layer[:,:,0][(slice_3_layer[:,:,0]>np.min(slice_3_layer)) &(image_nib_original[:,:,0]>=np.min(image_nib_original)) & (image_nib_original[:,:,0]<15)]=200
#        slice_3_layer[:,:,1][(slice_3_layer[:,:,1]>np.min(slice_3_layer)) & (image_nib_original[:,:,1]>=np.min(image_nib_original)) & (image_nib_original[:,:,1]<15)]=200
#        slice_3_layer[:,:,2][(slice_3_layer[:,:,2]>np.min(slice_3_layer)) & (image_nib_original[:,:,2]>=np.min(image_nib_original)) & (image_nib_original[:,:,2]<15)]=0 
#        
##        slice_3_layer[:,:,0][(slice_3_layer[:,:,0]>np.min(slice_3_layer)) &(image_nib_original[:,:,0]>=15) & (image_nib_original[:,:,0]<=37)]=0
##        slice_3_layer[:,:,1][(slice_3_layer[:,:,1]>np.min(slice_3_layer)) & (image_nib_original[:,:,1]>=15) & (image_nib_original[:,:,1]<=37)]=200
##        slice_3_layer[:,:,2][(slice_3_layer[:,:,2]>np.min(slice_3_layer)) & (image_nib_original[:,:,2]>=15) & (image_nib_original[:,:,2]<=37)]=0 
##        
##        slice_3_layer[:,:,0][(slice_3_layer[:,:,0]>np.min(slice_3_layer)) &(image_nib_original[:,:,0]>37)]=0
##        slice_3_layer[:,:,1][(slice_3_layer[:,:,1]>np.min(slice_3_layer)) & (image_nib_original[:,:,1]>37) ]=0
##        slice_3_layer[:,:,2][(slice_3_layer[:,:,2]>np.min(slice_3_layer)) & (image_nib_original[:,:,2]>37)]=200  
#        
#        
#        
# #  + 
#        
#
#        
##        slice_3_layer[:,:,2][(slice_3_layer_infarct_mask[:,:,0]>0) & (slice_3_layer_infarct_mask[:,:,0]<15)]=200
##        slice_3_layer[:,:,1][(slice_3_layer_infarct_mask[:,:,0]>=15) & (slice_3_layer_infarct_mask[:,:,0]<=37)]=200
#
##        slice_3_layer[:,:,0]=image_matx[:,:,i]*255
##        show_slice(slice_3_layer_mask)
#        show_slice(slice_3_layer_original)

#        cv2.imwrite(imagename2,slice_3_layer)
#        image1=cv2.imread(imagename1)
#        image2=cv2.imread(imagename2)
#        slice_3_layer_original_colored = cv2.add(image1,image2)
#        imagename3=  os.path.join(SLICE_OUTPUT_DIRECTORY, imagefilename+ str(i)+".png" )
#        
#        cv2.imwrite(imagename3,slice_3_layer_original_colored)
    
    
    
    
    
    
    
    
    
    