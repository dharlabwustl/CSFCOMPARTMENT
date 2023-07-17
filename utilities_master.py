from nilearn.image import resample_img
import pylab as plt
import nibabel as nb
import numpy as np
import itk,math,re
import numpy as np
import nibabel as nib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageEnhance
import os
import subprocess as subp 
import csv 
import glob
from datetime import datetime
import os 
import cv2 as cv 
import cv2
from matplotlib import pyplot as plt 
import subprocess,signal
from skimage import exposure
from tkinter.filedialog import askopenfilename
import vtk
from vtk import * 
from vtk.util import numpy_support
import numpy as np
import cv2 as cv
import numpy as np
import argparse
import random as rng
from scipy import spatial
rng.seed(12345)
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import smtplib,pickle
import matplotlib.pyplot as plt
#import image_functions as imf
import glob
import pandas as pd
import skimage.io as io
import subprocess
import threading
import csv,sys
import math
from scipy.spatial import distance
import SimpleITK as sitk
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank
import SimpleITK as sitk
import vtk
import numpy as np
import sys
from vtk.util.vtkConstants import *
import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import h5py
import subprocess
#import image_functions as imf
import cv2
import glob
from skimage import exposure
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.linear_model import RANSACRegressor
# import operating system and glob libraries
from scipy import stats
import os, glob
import h5py
from nilearn.datasets import fetch_neurovault_motor_task
from nilearn.datasets import load_mni152_template   
from nilearn.image import resample_to_img
from nilearn.image import load_img
from nilearn import plotting
# import some useful date functions

from datetime import datetime
import nibabel as nib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageEnhance
import os
import subprocess as subp
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.datasets import make_regression
import ants,os,sys
IMAGE_SIZE_TO=512
import numpy as np
from shapely.geometry import Point
from shapely.geometry import LineString
from image_features import image_features 
from utilities_simple import * 
def project_point_online(point1,line1):
    point = Point(point1[0], point1[1])
    line = LineString([line1[0], line1[1]])
    
    x = np.array(point.coords[0])
    
    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])
    
    n = v - u
    n /= np.linalg.norm(n, 2)
    P = u + n*np.dot(x - u, n)
    print(P) #0.2 1.
    
    return P

def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2
#    cv2.imshow("Biggest component", img2)
#    cv2.waitKey()
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")
 
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi




def  find_falxline_upper(gray_image_filepath,gray_bet_image_filepath,RESULT_DIR="",filter_type="",slicenumber=1):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    min_img_gray=np.min(img_gray.get_fdata())
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    for i in reversed(range( slicenumber,slicenumber+1 )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i>0:# 45: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            filtered_img=undesired_objects(filtered_img)
            score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefine(slice_3_layer,filtered_img)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            print(" I M WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return slice_3_layer, filtered_img, binary_image_copy,score_diff_from_1,slope_of_lines, pointA, pointB

def  find_falxline(gray_image_filepath,gray_bet_image_filepath,RESULT_DIR="",filter_type="",slicenumber=1):
    print(" I AM IN find_falxline")

    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    i=slicenumber
#    for i in reversed(range( slicenumber,slicenumber+1 )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
#        if i>0:# 45: 
#            show_slice(img[:,:,img.shape[2]-6])
    slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
    slice_3_layer[:,:,0]=img[:,:,i]*255
    slice_3_layer[:,:,1]=img[:,:,i]*255
    slice_3_layer[:,:,2]=img[:,:,i]*255
#                cv2.imwrite('slice.jpg',slice_3_layer)
#                image=cv2.imread('slice.jpg')

    g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
    gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
#                gray1, gray = detect_ridges(gray, sigma=3.0)
#                gray = cv2.bilateralFilter(gray,9,150,150)
#                gray = cv2.blur(gray,(3,3))
    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
    kernel = np.ones((3,3),np.uint8)
    img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
#            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
#            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
    filtered_img[img_gray_bet_edges_dil>0]=0
#                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
    filtered_img[filtered_img>0]=255
    alpha = 0.3
    beta = (1.0 - alpha)
#            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
    filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
    filtered_img_3_layer[:,:,0]=filtered_img
    filtered_img_3_layer[:,:,1]=filtered_img
    filtered_img_3_layer[:,:,2]=filtered_img
    filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
    score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefine(slice_3_layer,filtered_img)
    print("score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB")
    print([score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB])
    slope_of_lines.append((slope_of_line,pointA,pointB,i))
    print(" IM WORKING WELL")
    command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
    subprocess.call(command,shell=True)
    print("os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]")
    print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
    cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
    cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
    cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return slice_3_layer, filtered_img, binary_image_copy,score_diff_from_1,slope_of_lines, pointA, pointB

def  find_falxline_withREGNGabor(gray_image_filepath,gray_bet_image_filepath,upperslicepointsfromREG,RESULT_DIR="",filter_type="",slicenumber=1):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
#    print('gray_image_filepath')
#    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    img=exposure.rescale_intensity(img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    for i in reversed(range( slicenumber,slicenumber+1 )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i>0:# 45: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefinewithREGsupport(slice_3_layer,filtered_img,upperslicepointsfromREG)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            print("I M WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
#            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return slice_3_layer, filtered_img, binary_image_copy,score_diff_from_1,slope_of_lines, pointA, pointB

def  find_falxline_withREGNGabor_v1(gray_image_filepath,gray_bet_image_filepath,upperslicepointsfromREG,RESULT_DIR="",filter_type="",slicenumber=1):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
#    print('gray_image_filepath')
#    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
#    img=exposure.rescale_intensity(img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    for i in reversed(range( slicenumber,slicenumber+1 )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i>0:# 45: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefinewithREGsupport_v1(slice_3_layer,filtered_img,upperslicepointsfromREG)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            print("I M WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
#            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return slice_3_layer, filtered_img, binary_image_copy,score_diff_from_1,slope_of_lines, pointA, pointB


def  find_falxline_withREGOnly(gray_image_filepath,gray_bet_image_filepath,upperslicepointsfromREG,RESULT_DIR="",filter_type="",slicenumber=1):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    img=exposure.rescale_intensity(img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    for i in reversed(range( slicenumber,slicenumber+1 )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i>0:# 45: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefinewithREGsupport(slice_3_layer,filtered_img,upperslicepointsfromREG)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            print("I M WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return slice_3_layer, filtered_img, binary_image_copy,score_diff_from_1,slope_of_lines, pointA, pointB

def  find_falxline_v1(gray_image_filepath,gray_bet_image_filepath,RESULT_DIR="",filter_type="",slicenumber=1):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    img=exposure.rescale_intensity(img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    for i in reversed(range( slicenumber,slicenumber+1 )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i>0:# 45: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefine(slice_3_layer,filtered_img)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            print("I M WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return slice_3_layer, filtered_img, binary_image_copy,score_diff_from_1,slope_of_lines, pointA, pointB


def find_four_points_for_plane_nosave(gray_image_filepath,RESULT_DIR,filter_type,slicefirst=1,sliceend=15):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    img_nii=nib.load(gray_image_filepath) #filesindir[file])
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    print('file_base_name')
    print(gray_image_filepath)
    img_gray=nib.load(os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    #        img = cv2.imread(filesindir[file]) # img_nii.get_fdata() #cv2.imread('0.jpg')
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*img_nii.get_fdata()
    slope_of_lines=[]
    pointAs=[]
    pointBs=[]
    #        
    ##        np3darray_show_slice(img)
    for i in range( slicefirst,sliceend+1 ): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i >= 0: #img.shape[2] * 1/3: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac(slice_3_layer,filtered_img)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            pointAs.append(pointA)
            pointBs.append(pointB)

    xx=np.array(slope_of_lines)
    print(xx)
    yy=xx[:,0]
    zz=np.copy(yy)
#    zz[np.nonzero(yy)[0][1]]=9878979
    zz[0:int(len(zz)/2)]=9878979
    idx = (np.abs(zz - yy[np.nonzero(yy)[0][1]])).argmin()
    print(xx[np.nonzero(yy)[0][1]])
    print(xx[idx])
    return slice_3_layer, filtered_img, binary_image_copy, xx[np.nonzero(yy)[0][1]], xx[idx]

def find_four_points_for_plane_v1(gray_image_filepath,gray_bet_image_filepath,RESULT_DIR="",filter_type="",slicefirst=1,sliceend=15):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    img_nii=nib.load(gray_image_filepath) #filesindir[file])
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    #        img = cv2.imread(filesindir[file]) # img_nii.get_fdata() #cv2.imread('0.jpg')
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    pointAs=[]
    pointBs=[]
    sliceend=img_gray.get_fdata().shape[2]-1
    #        
    ##        np3darray_show_slice(img)
    for i in reversed(range( slicefirst,sliceend )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i >= 0: #img.shape[2] * 1/3: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac(slice_3_layer,filtered_img)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            pointAs.append(pointA)
            pointBs.append(pointB)
            print(" IM WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)

    xx=np.array(slope_of_lines)
    yy=xx[:,0]
    zz=np.copy(yy)
#    zz[np.nonzero(yy)[0][1]]=9878979
    zz[0:int(len(zz)/2)]=9878979
    idx = (np.abs(zz - yy[np.nonzero(yy)[0][2]])).argmin()
    print(xx[np.nonzero(yy)[0][1]])
    print(xx[idx])
    return slice_3_layer, filtered_img, binary_image_copy, xx[np.nonzero(yy)[0][3]], xx[idx],slope_of_lines
def find_four_points_for_plane_v1_1(gray_image_filepath,gray_bet_image_filepath,RESULT_DIR="",filter_type="",slicefirst=1,sliceend=15):
#####    file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicefirst=0,sliceend=15
#    gray_image_filepath=file_gray
#    gray_bet_image_filepath=file_gray_bet
#    slicefirst=1
#    sliceend=15
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
#    img_nii=nib.load(gray_image_filepath) #filesindir[file])
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(gray_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(gray_bet_image_filepath) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    #        img = cv2.imread(filesindir[file]) # img_nii.get_fdata() #cv2.imread('0.jpg')
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    slope_of_lines=[]
    pointAs=[]
    pointBs=[]
    sliceend=img_gray.get_fdata().shape[2]-1
    score_diff_from_1_array=[]
    #        
    ##        np3darray_show_slice(img)
    for i in reversed(range( slicefirst,sliceend )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i>0:# 45: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 

    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
#            filtered_img[gray<50]
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            score_diff_from_1, binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac_withrefine(slice_3_layer,filtered_img)
            slope_of_lines.append((slope_of_line,pointA,pointB,i,score_diff_from_1))
            pointAs.append(pointA)
            pointBs.append(pointB)
            print('pointA:\n')
            print(np.array(pointA).shape[0])
#            if (np.array(pointA).shape[0]>0):
#                score_diff_from_1 ,binary_image_copy, slope_of_line,pointA,pointB= upper_slice_midline_refinement(np.uint8(img_gray_bet.get_fdata()[:,:,i]*255),gray,filtered_img,(pointA[0],pointA[1]),(pointB[0],pointB[1]) )
#                score_diff_from_1_array.append([i,score_diff_from_1 ,binary_image_copy, slope_of_line,pointA,pointB])
            print(" IM WORKING WELL")
            command="mkdir -p "  + os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type)
            subprocess.call(command,shell=True)
            print(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'))
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)

    score_diff_from_1_array=np.array(score_diff_from_1_array)
    file = open('slopeoflinesWUSTL660.pickle', 'rb')
    
#    # dump information to that file
#    data = pickle.load(file)
#    
#    # close the file
#    file.close()
#    slope_of_lines=data['slope_of_lines']
    slope_of_lines=np.array(slope_of_lines)
    output_vars = {"slope_of_lines":slope_of_lines} #,"img":img, "min_score_slice_n_points_top": min_score_slice_n_points_top, "min_slope_diff_topN_bottom": min_slope_diff_topN_bottom,"score_diff_from_1_array":score_diff_from_1_array }
    f_myfile = open('slopeoflinesWUSTL660.pickle', 'wb')
    pickle.dump(output_vars, f_myfile)
    f_myfile.close()
    
    slope_of_lines_nonempty=[]
    for x in slope_of_lines:
        if len(x[1])>0:
            slope_of_lines_nonempty.append(x)
    slope_of_lines_nonempty=np.array(slope_of_lines_nonempty)    
    slope_of_lines_nonempty_slice_min=np.min(slope_of_lines_nonempty[:,3])
    slope_of_lines_nonempty_slice_max=np.max(slope_of_lines_nonempty[:,3])
    slope_of_lines_nonempty_slice_range=slope_of_lines_nonempty_slice_max-slope_of_lines_nonempty_slice_min
    print('slope_of_lines')
    print(slope_of_lines)
    score_diff_from_1_min=9999999999999
    min_score_slice_n_points_top=[]
    count_this=0
    for i in range(slope_of_lines_nonempty.shape[0]):
        if slope_of_lines_nonempty[i][3] > (slope_of_lines_nonempty_slice_max - slope_of_lines_nonempty_slice_range*2/10 ):
            if slope_of_lines_nonempty[i][4] < score_diff_from_1_min:
                score_diff_from_1_min=slope_of_lines_nonempty[i][4]
                print(score_diff_from_1_min)
    #            if score_diff_from_1_min > score_diff_from_1_array[i][1]:
    #                score_diff_from_1_min = score_diff_from_1_array[i][1]
                min_score_slice_n_points_top=slope_of_lines_nonempty[i]
#    for i in range(slope_of_lines.shape[0]):
#        if slope_of_lines[i][3] > (img.shape[2]*2/3):
#            if slope_of_lines[i][4] < score_diff_from_1_min:
#                score_diff_from_1_min=slope_of_lines[i][4]
#                print(score_diff_from_1_min)
#    #            if score_diff_from_1_min > score_diff_from_1_array[i][1]:
#    #                score_diff_from_1_min = score_diff_from_1_array[i][1]
#                min_score_slice_n_points_top=slope_of_lines[i]
    
    slope_diff=999999
    min_slope_diff_topN_bottom=[]
    
    for i in range(slope_of_lines_nonempty.shape[0]):
        if  slope_of_lines_nonempty[i][3] < (slope_of_lines_nonempty_slice_max - slope_of_lines_nonempty_slice_range*5/10 ) and slope_of_lines[i][3] >  (slope_of_lines_nonempty_slice_max - slope_of_lines_nonempty_slice_range*6/10 ): #score_diff_from_1_array[i][0] <  (img.shape[2]*9/10) and
            slope_dif_current= slope_of_lines_nonempty[i][4]- min_score_slice_n_points_top[3]
            if slope_dif_current< slope_diff:
                slope_diff=slope_dif_current
                print(score_diff_from_1_min)
    #            if score_diff_from_1_min > score_diff_from_1_array[i][1]:
    #                score_diff_from_1_min = score_diff_from_1_array[i][1]
                min_slope_diff_topN_bottom=slope_of_lines_nonempty[i]
                
#    for i in range(slope_of_lines.shape[0]):
#        if  slope_of_lines[i][3] <  (img.shape[2]*1/2) and slope_of_lines[i][3] >  (img.shape[2]*1/3): #score_diff_from_1_array[i][0] <  (img.shape[2]*9/10) and
#            slope_dif_current= slope_of_lines[i][4]- min_score_slice_n_points_top[3]
#            if slope_dif_current< slope_diff:
#                slope_diff=slope_dif_current
#                print(score_diff_from_1_min)
#    #            if score_diff_from_1_min > score_diff_from_1_array[i][1]:
#    #                score_diff_from_1_min = score_diff_from_1_array[i][1]
#                min_slope_diff_topN_bottom=slope_of_lines[i]
    
#    print(score_diff_from_1_array)
#    score_diff_from_1_min=99999999
#    min_score_slice_n_points_top=[]
#    for i in range(score_diff_from_1_array.shape[0]):
#        if score_diff_from_1_array[i][0] > (img.shape[2]*9/10):
#            if score_diff_from_1_array[i][1] < score_diff_from_1_min:
#                score_diff_from_1_min=score_diff_from_1_array[i][1]
#                print(score_diff_from_1_min)
#    #            if score_diff_from_1_min > score_diff_from_1_array[i][1]:
#    #                score_diff_from_1_min = score_diff_from_1_array[i][1]
#                min_score_slice_n_points_top=score_diff_from_1_array[i]
#    slope_diff=999999
#    min_slope_diff_topN_bottom=[]
#    for i in range(score_diff_from_1_array.shape[0]):
#        if  score_diff_from_1_array[i][0] >  (img.shape[2]*1/2): #score_diff_from_1_array[i][0] <  (img.shape[2]*9/10) and
#            slope_dif_current= score_diff_from_1_array[i][1]- min_score_slice_n_points_top[3]
#            if slope_dif_current< slope_diff:
#                slope_diff=slope_dif_current
#                print(score_diff_from_1_min)
#    #            if score_diff_from_1_min > score_diff_from_1_array[i][1]:
#    #                score_diff_from_1_min = score_diff_from_1_array[i][1]
#                min_slope_diff_topN_bottom=score_diff_from_1_array[i]
      
    print(min_score_slice_n_points_top)        
#    xx=np.array(min_score_slice_n_points_top[3])
    print(min_slope_diff_topN_bottom)  
    output_vars = {"img_gray_bet":img_gray_bet,"img":img, "min_score_slice_n_points_top": min_score_slice_n_points_top, "min_slope_diff_topN_bottom": min_slope_diff_topN_bottom,"score_diff_from_1_array":score_diff_from_1_array }
    f_myfile = open('upper_lower_row.pickle', 'wb')
    pickle.dump(output_vars, f_myfile)
    f_myfile.close()
    
    np.save('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/SOFTWARES/score_diff_from_1_array.npy', score_diff_from_1_array)
    pointA=(int(min_slope_diff_topN_bottom[1][0]),int(min_slope_diff_topN_bottom[1][1])) 
    pointB=(int(min_slope_diff_topN_bottom[2][0]),int(min_slope_diff_topN_bottom[2][1])) 
    
#    pointA=(int(min_slope_diff_topN_bottom[4][0]),int(min_slope_diff_topN_bottom[4][1])) 
#    pointB=(int(min_slope_diff_topN_bottom[5][0]),int(min_slope_diff_topN_bottom[5][1])) 
    PointA,PointB=lower_slice_midline_refinement_v1_1(np.uint8(img_gray_bet.get_fdata()[:,:,min_slope_diff_topN_bottom[3]]*255),np.uint8(img[:,:,min_slope_diff_topN_bottom[3]]*255),pointA,pointB,anglethreshold=10)
    min_slope_diff_topN_bottom[1][0],min_slope_diff_topN_bottom[1][1]=PointA
    min_slope_diff_topN_bottom[2][0],min_slope_diff_topN_bottom[2][1]=PointB
    return slice_3_layer, filtered_img, binary_image_copy, min_score_slice_n_points_top, min_slope_diff_topN_bottom,score_diff_from_1_array


def find_four_points_for_plane(gray_image_filepath,RESULT_DIR="",filter_type="",slicefirst=1,sliceend=15):
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    img_nii=nib.load(gray_image_filepath) #filesindir[file])
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    img_gray=nib.load(os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    
    #        img = cv2.imread(filesindir[file]) # img_nii.get_fdata() #cv2.imread('0.jpg')
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*img_gray_bet.get_fdata()
    slope_of_lines=[]
    pointAs=[]
    pointBs=[]
    #        
    ##        np3darray_show_slice(img)
    for i in reversed(range( slicefirst,sliceend )): #img.shape[2]) : #,img.shape[2]): # 1,15) : #
        if i >= 0: #img.shape[2] * 1/3: 
    #            show_slice(img[:,:,img.shape[2]-6])
            slice_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            slice_3_layer[:,:,0]=img[:,:,i]*255
            slice_3_layer[:,:,1]=img[:,:,i]*255
            slice_3_layer[:,:,2]=img[:,:,i]*255
    #                cv2.imwrite('slice.jpg',slice_3_layer)
    #                image=cv2.imread('slice.jpg')
    
            g_kernel = cv2.getGaborKernel((10, 10), 10.0, np.pi/2, 10, 1, 0, ktype=cv2.CV_32F)
            gray =np.uint8(img[:,:,i]*255) # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #np.uint8(img)# img[:,:,img.shape[2]-6] # 
    #                gray1, gray = detect_ridges(gray, sigma=3.0)
    #                gray = cv2.bilateralFilter(gray,9,150,150)
    #                gray = cv2.blur(gray,(3,3))
            filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            img_gray_bet_edges = cv2.Canny(np.uint8(img_gray_bet.get_fdata()[:,:,i]),0,1,apertureSize = 3)
            kernel = np.ones((3,3),np.uint8)
            img_gray_bet_edges_dil=cv2.dilate(img_gray_bet_edges, kernel,  iterations=5)
    #            filtered_img=cv2.erode(filtered_img, kernel,iterations=1)
    #            filtered_img=cv2.dilate(filtered_img, kernel,iterations=2)
            filtered_img[img_gray_bet_edges_dil>0]=0
    #                filtered_img[filtered_img<np.max(filtered_img)*.80]=0
            filtered_img[filtered_img>0]=255
            alpha = 0.3
            beta = (1.0 - alpha)
    #            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
            filtered_img_3_layer= np.zeros([img.shape[0],img.shape[1],3])
            filtered_img_3_layer[:,:,0]=filtered_img
            filtered_img_3_layer[:,:,1]=filtered_img
            filtered_img_3_layer[:,:,2]=filtered_img
            filtered_img_1 = cv2.addWeighted(slice_3_layer[:,:,1], alpha, filtered_img_3_layer[:,:,1], beta, 0.0)
            binary_image_copy,slope_of_line,pointA,pointB=fit_a_line_ransac(slice_3_layer,filtered_img)
            slope_of_lines.append((slope_of_line,pointA,pointB,i))
            pointAs.append(pointA)
            pointBs.append(pointB)
            print(" IM WORKING WELL")
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
            
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
            
            cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)

    xx=np.array(slope_of_lines)
    yy=xx[:,0]
    zz=np.copy(yy)
#    zz[np.nonzero(yy)[0][1]]=9878979
    zz[0:int(len(zz)/2)]=9878979
    idx = (np.abs(zz - yy[np.nonzero(yy)[0][1]])).argmin()
    print(xx[np.nonzero(yy)[0][1]])
    print(xx[idx])
    return slice_3_layer, filtered_img, binary_image_copy, xx[np.nonzero(yy)[0][1]], xx[idx]


def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def fit_a_line_ransac(slice_3_layer,binary_image):
    binary_image_copy=np.copy(slice_3_layer)
    pointA=[]
    pointB=[]
    slope_of_line=0
    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    #    print(len(binary_image_non_zero_cord))
    if len(binary_image_non_zero_cord)>300:
        X=binary_image_non_zero_cord[:,1].reshape(-1,1)
        y=binary_image_non_zero_cord[:,0].reshape(-1,1)
        boston_df = pd.DataFrame(y)
        boston_df.columns = ['y']
        z = np.abs(stats.zscore(boston_df))
        threshold = 3
        aa=np.where(z > threshold)
    
        reg = RANSACRegressor(random_state=0,min_samples=300, max_trials=10000).fit(X,y)
        X_test=np.arange(0,512).reshape(-1, 1)
        Y_pred = reg.predict(X_test) 
        slope_of_line=(Y_pred[511] - Y_pred[0]) / (X_test[511]-X_test[0])  #np.array([X_test[511],Y_pred[511]])-np.array([X_test[0],Y_pred[0]])
        pointA= np.array([X_test[0][0],Y_pred[0][0]])
        pointB= np.array([X_test[511][0],Y_pred[511][0]])
        for  each_point in binary_image_non_zero_cord : #range(0,512): #
            cv2.circle(binary_image_copy,(each_point[1],each_point[0]),2,(0,200,0),1)
    
        for k in range(len(X_test)):
                cv2.circle(binary_image_copy,(X_test[k],Y_pred[k]),2,(188,0,0),1)
    
    return binary_image_copy, slope_of_line,pointA,pointB

#    for each_point in upperhalf:
#    #    print(each_point[0])
#        v1= np.array(pointAnp) - mean_point_contournp_mean[0]
#        v2= each_point - mean_point_contournp_mean[0]
#        angle1=np.abs(angle_bet_two_vector(v1,v2))
#        if angle1>270:
#            print(angle1)
#            angle1=360-angle1
#        if angle1 <anglethreshold :
#            upperhalf_20deg.append(each_point)
#            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
#    upperhalf_20deg=np.array(upperhalf_20deg)

def fit_a_line_ransac_withrefinewithREGsupport(slice_3_layer,binary_image,pointsfromREG):
#    print(pointsfromREG)
    print("I AM HERE")
    binary_image_copy=np.copy(slice_3_layer)
    pointA=[]
    pointB=[]
    slope_of_line=0
    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
    print("len(binary_image_non_zero_cord_t)")
    print(len(binary_image_non_zero_cord_t[0]))
    if len(binary_image_non_zero_cord_t[0])>300:
        print("I AM >300")
        binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
        #    print(len(binary_image_non_zero_cord))
        score_diff_from_1=9999999999
        pointAfromREG=np.array([pointsfromREG[4],pointsfromREG[3]])
        pointBfromREG=np.array([pointsfromREG[2],pointsfromREG[1]])
    #    print(pointAfromREG)
    #    print(pointBfromREG)
        pointABfromREG_mean=np.mean([pointAfromREG,pointBfromREG],axis=0)
    #    print(pointABfromREG_mean)
        upperhalf_20deg=[]
        distance_series=[]
        anglethreshold=20
#        print("len(binary_image_non_zero_cord)")
#        print(len(binary_image_non_zero_cord))
        for each_point in binary_image_non_zero_cord:
    #        upperhalf_20deg.append(each_point)
    #        print(each_point[0])
            d = np.linalg.norm(np.cross(pointAfromREG-pointBfromREG, pointBfromREG-each_point))/np.linalg.norm(pointAfromREG-pointBfromREG)
    #        if d < 10:
            distance_series.append([d,each_point[0],each_point[1]])
#        print(distance_series)
        distance_series=np.array(distance_series)
        distance_series=distance_series[distance_series[:, 0].argsort()]
#        print(distance_series)
        counterr=0
        for eachpoint in distance_series:
            if counterr <=400:
                upperhalf_20deg.append([int(eachpoint[1]),int(eachpoint[2])])
                counterr=counterr+1
    #            cv2.circle(binary_image_copy,(each_point[0],each_point[1]),2,(100,200,0),1)
    ##    #    print(each_point[0])
    ##        v1= np.array(pointAfromREG) - pointABfromREG_mean[0]
    ##        v2= each_point - pointABfromREG_mean[0]
    ##        angle1=np.abs(angle_bet_two_vector(v1,v2))
    ##        if angle1>270:
    ##            print(angle1)
    ##            angle1=360-angle1
    ##        if angle1 <anglethreshold :
    ##            upperhalf_20deg.append(each_point)
    ##            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
        binary_image_non_zero_cord=np.array(upperhalf_20deg)
#        print("len(upperhalf_20deg)")
#        print(len(upperhalf_20deg))
        if len(binary_image_non_zero_cord)>50:
            X=binary_image_non_zero_cord[:,1].reshape(-1,1)
            y=binary_image_non_zero_cord[:,0].reshape(-1,1)
            boston_df = pd.DataFrame(y)
            boston_df.columns = ['y']
            z = np.abs(stats.zscore(boston_df))
            threshold = 3
            aa=np.where(z > threshold)
            reg = RANSACRegressor(random_state=0,min_samples=50, max_trials=100000,residual_threshold =1).fit(X,y)
            score_diff_from_1= 1-reg.score(X, y)
            X_test=np.arange(0,512).reshape(-1, 1)
            Y_pred = reg.predict(X_test) 
            slope_of_line=(Y_pred[511] - Y_pred[0]) / (X_test[511]-X_test[0])  #np.array([X_test[511],Y_pred[511]])-np.array([X_test[0],Y_pred[0]])
            pointA= np.array([X_test[0][0],Y_pred[0][0]])
            pointB= np.array([X_test[511][0],Y_pred[511][0]])
            for  each_point in binary_image_non_zero_cord : #range(0,512): #
                cv2.circle(binary_image_copy,(each_point[1],each_point[0]),2,(0,200,0),1)   
            for k in range(len(X_test)):
                    cv2.circle(binary_image_copy,(X_test[k],Y_pred[k]),2,(188,0,0),1)
        cv2.line(binary_image_copy, (int(pointAfromREG[1]),int(pointAfromREG[0])), (int(pointBfromREG[1]),int(pointBfromREG[0])),  (255,255,0), 3)
    else:
        pointA=np.array([pointsfromREG[3],pointsfromREG[4]])
        pointB=np.array([pointsfromREG[1],pointsfromREG[2]])
        cv2.line(binary_image_copy, (int(pointA[0]),int(pointA[1])), (int(pointB[0]),int(pointB[1])),  (255,255,0), 3)
        score_diff_from_1=0


    return score_diff_from_1, binary_image_copy, slope_of_line,pointA,pointB

def fit_a_line_ransac_withrefinewithREGsupport_v1(slice_3_layer,binary_image,pointsfromREG):
#    print(pointsfromREG)
    print("I AM HERE")
    binary_image_copy=np.copy(slice_3_layer)
    pointA=[]
    pointB=[]
    slope_of_line=0
    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
    print("len(binary_image_non_zero_cord_t)")
    print(len(binary_image_non_zero_cord_t[0]))
    if len(binary_image_non_zero_cord_t[0])>300:
        print("I AM >300")
        binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
        #    print(len(binary_image_non_zero_cord))
        score_diff_from_1=9999999999
        pointAfromREG=np.array([pointsfromREG[4],pointsfromREG[3]])
        pointBfromREG=np.array([pointsfromREG[2],pointsfromREG[1]])
    #    print(pointAfromREG)
    #    print(pointBfromREG)
        pointABfromREG_mean=np.mean([pointAfromREG,pointBfromREG],axis=0)
    #    print(pointABfromREG_mean)
        upperhalf_20deg=[]
        distance_series=[]
        anglethreshold=20
#        print("len(binary_image_non_zero_cord)")
#        print(len(binary_image_non_zero_cord))
        for each_point in binary_image_non_zero_cord:
    #        upperhalf_20deg.append(each_point)
    #        print(each_point[0])
            d = np.linalg.norm(np.cross(pointAfromREG-pointBfromREG, pointBfromREG-each_point))/np.linalg.norm(pointAfromREG-pointBfromREG)
    #        if d < 10:
            distance_series.append([d,each_point[0],each_point[1]])
#        print(distance_series)
        distance_series=np.array(distance_series)
        distance_series=distance_series[distance_series[:, 0].argsort()]
        print(np.min(distance_series[:,0]))
        counterr=0
        for eachpoint in distance_series:
            if eachpoint[0]<   np.median(distance_series[:,0])+np.std(distance_series[:,0]):#counterr <=400:
                upperhalf_20deg.append([int(eachpoint[1]),int(eachpoint[2])])
                counterr=counterr+1
    #            cv2.circle(binary_image_copy,(each_point[0],each_point[1]),2,(100,200,0),1)
    ##    #    print(each_point[0])
    ##        v1= np.array(pointAfromREG) - pointABfromREG_mean[0]
    ##        v2= each_point - pointABfromREG_mean[0]
    ##        angle1=np.abs(angle_bet_two_vector(v1,v2))
    ##        if angle1>270:
    ##            print(angle1)
    ##            angle1=360-angle1
    ##        if angle1 <anglethreshold :
    ##            upperhalf_20deg.append(each_point)
    ##            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
        binary_image_non_zero_cord=np.array(upperhalf_20deg)
#        print("len(upperhalf_20deg)")
#        print(len(upperhalf_20deg))
        print("len(binary_image_non_zero_cord)")
        print(len(binary_image_non_zero_cord))
        if len(binary_image_non_zero_cord)>50:
            print("I am at line number 1076")
            X=binary_image_non_zero_cord[:,1].reshape(-1,1)
            y=binary_image_non_zero_cord[:,0].reshape(-1,1)
            boston_df = pd.DataFrame(y)
            boston_df.columns = ['y']
            z = np.abs(stats.zscore(boston_df))
            threshold = 3
            aa=np.where(z > threshold)
            reg = RANSACRegressor(random_state=0,min_samples=50, max_trials=100000,residual_threshold =1).fit(X,y)
            score_diff_from_1= 1-reg.score(X, y)
            X_test=np.arange(0,512).reshape(-1, 1)
            Y_pred = reg.predict(X_test) 
            slope_of_line=(Y_pred[511] - Y_pred[0]) / (X_test[511]-X_test[0])  #np.array([X_test[511],Y_pred[511]])-np.array([X_test[0],Y_pred[0]])
            pointA= np.array([X_test[0][0],Y_pred[0][0]])
            pointB= np.array([X_test[511][0],Y_pred[511][0]])
            for  each_point in binary_image_non_zero_cord : #range(0,512): #
                cv2.circle(binary_image_copy,(each_point[1],each_point[0]),2,(0,200,0),1)   
            for k in range(len(X_test)):
                    cv2.circle(binary_image_copy,(X_test[k],Y_pred[k]),2,(188,0,0),1)
        cv2.line(binary_image_copy, (int(pointAfromREG[1]),int(pointAfromREG[0])), (int(pointBfromREG[1]),int(pointBfromREG[0])),  (255,255,0), 3)
    else:
        pointA=np.array([pointsfromREG[3],pointsfromREG[4]])
        pointB=np.array([pointsfromREG[1],pointsfromREG[2]])
        cv2.line(binary_image_copy, (int(pointA[0]),int(pointA[1])), (int(pointB[0]),int(pointB[1])),  (255,255,0), 3)
        score_diff_from_1=0


    return score_diff_from_1, binary_image_copy, slope_of_line,pointA,pointB


def fit_a_line_ransac_withrefine(slice_3_layer,binary_image):
    binary_image_copy=np.copy(slice_3_layer)
    pointA=[]
    pointB=[]
    slope_of_line=0
    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    #    print(len(binary_image_non_zero_cord))
    score_diff_from_1=9999999999
    if len(binary_image_non_zero_cord)>50:
        X=binary_image_non_zero_cord[:,1].reshape(-1,1)
        y=binary_image_non_zero_cord[:,0].reshape(-1,1)
        boston_df = pd.DataFrame(y)
        boston_df.columns = ['y']
        z = np.abs(stats.zscore(boston_df))
        threshold = 3
        aa=np.where(z > threshold)
    
        reg = RANSACRegressor(random_state=0,min_samples=50, max_trials=100000,residual_threshold =1).fit(X,y)
        score_diff_from_1= 1-reg.score(X, y)
        X_test=np.arange(0,512).reshape(-1, 1)
        Y_pred = reg.predict(X_test) 
        slope_of_line=(Y_pred[511] - Y_pred[0]) / (X_test[511]-X_test[0])  #np.array([X_test[511],Y_pred[511]])-np.array([X_test[0],Y_pred[0]])
        pointA= np.array([X_test[0][0],Y_pred[0][0]])
        pointB= np.array([X_test[511][0],Y_pred[511][0]])
        for  each_point in binary_image_non_zero_cord : #range(0,512): #
            cv2.circle(binary_image_copy,(each_point[1],each_point[0]),2,(0,200,0),1)
    
        for k in range(len(X_test)):
                cv2.circle(binary_image_copy,(X_test[k],Y_pred[k]),2,(188,0,0),1)
    
    return score_diff_from_1, binary_image_copy, slope_of_line,pointA,pointB

def write_image_mask(directory_image,directory_mask,mask_ext_string,outputdirectory):
    files_image=glob.glob(directory_image+ "/*")
    for file in files_image:
        imagebasename=os.path.basename(file)
        imagebasename_wext=imagebasename.split('.')
        mask_basename=imagebasename_wext[0]+mask_ext_string
        maskfilename=os.path.join(directory_mask,mask_basename)
        image1=cv2.imread(file)
        image2=cv2.imread(maskfilename)
        alpha = 0.3
        beta = (1.0 - alpha)
    ##            filtered_img = cv2.addWeighted(np.uint8(img[:,:,img.shape[2]-i]), alpha, filtered_img, beta, 0.0)
    #    image1_layer= np.zeros([image1.shape[0],image1.shape[1],3])
    #    image1_layer[:,:,0]=image1
    #    image1_layer[:,:,1]=image1
    #    image1_layer[:,:,2]=image1
    #    image2_layer= np.zeros([image2.shape[0],image2.shape[1],3])
    #    image2_layer[:,:,0]=image1
    #    image2_layer[:,:,1]=image1
    #    image2_layer[:,:,2]=image1
        image_combined = cv2.addWeighted(image1, alpha, image2, beta, 0.0)
        cv2.imwrite(os.path.join(outputdirectory,imagebasename_wext[0]+ '_combined.png'),image_combined)
    
def create_histo(data, min_val, max_val, num_bins=None):

    if num_bins is None:
        num_bins = int(np.max(data) - np.min(data))    
    hist, bins = np.histogram(data, num_bins, range=(min_val, max_val))
    fig3, axes3 = plt.subplots(1, 1,figsize=(8, 8))
  
    hist = np.array(hist) 
    axes3.hist(hist, bins = bins) 
    axes3.hist(data.T, cmap="gray", origin="lower")
    
    return fig3 
def contraststretch(data,maxx,minn): 
    epi_img_data_max=maxx #np.max(data)
    epi_img_data_min=minn #np.min(data)
    thisimage=data
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min) * 255
    return thisimage

def normalizeimage(data): 
    epi_img_data_max=np.max(data)
    epi_img_data_min=np.min(data)
    thisimage=data
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min)  * 255
    return thisimage

def normalizeimage0to1(data): 
    epi_img_data_max=np.max(data)
    epi_img_data_min=np.min(data)
    thisimage=data
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min) 
    return thisimage

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def independentimagestack(directoryname,filenameniigz,outputdirectory):
    affine=affine = np.diag([1, 1, 1, 1])
    allfiles= glob.glob(directoryname+"/*.nii.gz")
    image_stack=[]
    for file in allfiles:
        img=nib.load(os.path.join(directoryname,file))
        img_data=img.get_fdata()
        img_data = cv2.resize(img_data, (512, 512))
        image_stack.append(img_data)
        
    numpy2dmat=np.asarray(image_stack)
    filenameniigz=filenameniigz+ ".nii.gz"
    command='rm     ' +  os.path.join(outputdirectory,filenameniigz)
    subprocess.call(command, shell=True)
    array_img = nib.Nifti1Image(numpy2dmat, affine)
    nib.save(array_img, os.path.join(outputdirectory,filenameniigz))
    return "XX"

def subset_brain_stack(image_stack,outputfilenameniigz,outputdirectory,affine):
#    affine= affine
    numpy2dmat=np.asarray(image_stack)
    array_img = nib.Nifti1Image(numpy2dmat, affine)
    filenameniigz=outputfilenameniigz+ ".nii.gz"
    nib.save(array_img, os.path.join(outputdirectory,filenameniigz))
    return "XX"

   
    
    
def numpy2dmat2niigzNOAFFINE(numpy2dmat,directoryname,filenameniigz):
    affine=affine = np.diag([1, 1, 1, 1])
    filenameniigz=filenameniigz+ ".nii.gz"
    command='rm     ' +  os.path.join(directoryname,filenameniigz)
    subprocess.call(command, shell=True)
    array_img = nib.Nifti1Image(numpy2dmat, affine)
    nib.save(array_img, os.path.join(directoryname,filenameniigz))
    return "XX"

def numpy2dmat2niigz(numpy2dmat,affine,directoryname,filenameniigz):
    filenameniigz=filenameniigz+ ".nii.gz"
    command='rm     ' +  os.path.join(directoryname,filenameniigz)
    subprocess.call(command, shell=True)
    array_img = nib.Nifti1Image(numpy2dmat, affine)
    nib.save(array_img, os.path.join(directoryname,filenameniigz))
    return "XX"

def numpy2dmat2h5(numpy2dmat,directoryname,filenameh5):
    filenameh5=filenameh5+ ".h5"
    command='rm     ' +  os.path.join(directoryname,filenameh5)
    subprocess.call(command, shell=True)
    hf = h5py.File(os.path.join(directoryname,filenameh5), 'w')
    hf.create_dataset('image', data=numpy2dmat)
    hf.close()
    return "XX"

def niigz2niiformat(niigzfilename,niifilenametosave):
    filenameniigz=niigzfilename
    niigzdata=nib.load(filenameniigz)
    array_img = nib.Nifti1Image(niigzdata.get_fdata(), header=niigzdata.header)
    nib.save(array_img, niifilenametosave)
    return "XX"
def hdr2niigz(hdrfilename,niigzfilenametosave,headerfiledata):
    filenameniigz=hdrfilename
    analyzedata=nib.AnalyzeImage.from_filename(filenameniigz)
    array_img = nib.Nifti1Image(analyzedata.get_fdata(), affine=headerfiledata.affine, header=headerfiledata.header)
    nib.save(array_img, niigzfilenametosave)
    
    
def hdr2niigz1(hdrfilename,niigzfilenametosave):
    filenameniigz=hdrfilename
    analyzedata=nib.AnalyzeImage.from_filename(filenameniigz)
    array_img = nib.Nifti1Image(analyzedata.get_fdata(), affine=analyzedata.affine, header=analyzedata.header)
    nib.save(array_img, niigzfilenametosave)

def create_raw_images_label_niigz(POINT_LOCATION,TYPE_OF_DATA,RAW_DATA_FOLDER):
    COMMON_DIR=os.path.join(RAW_DATA_FOLDER,POINT_LOCATION) #RAW_DATA_FOLDER #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/Ideal/INBRAIN"
    
    IML_directory=os.path.join(COMMON_DIR,TYPE_OF_DATA) # "/TEST"
    #COMMON_DIR=os.path.join(COMMON_DIR,TYPE_OF_DATA)
    IML_directory1=os.path.join(RAW_DATA_FOLDER,TYPE_OF_DATA)
#    command='rm -r ' + IML_directory1 +'/AV'
#    subprocess.call(command, shell=True)
    command='mkdir -p  ' + IML_directory1 +'/AV'
    subprocess.call(command, shell=True)
#    command='rm -r ' + IML_directory1 +'/V'
#    subprocess.call(command, shell=True)
    command='mkdir -p  ' + IML_directory1 +'/V'
    subprocess.call(command, shell=True)
#    command='rm -r ' + IML_directory1 +'/BV'
#    subprocess.call(command, shell=True)
    command='mkdir -p  ' + IML_directory1 +'/BV'
    subprocess.call(command, shell=True)
    BET_MASK_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/Ideal/grayscale_BET"
    SAVE_RESULT_DIR_GRAY=os.path.join(IML_directory1) #IML_directory + "/grayscale_av"
    counter=0
    data_directories=get_immediate_subdirectories(IML_directory)
    TYPES_OF_SLICE=["AV","V"]
    for TYPE_OF_SLICE in TYPES_OF_SLICE:
        for each_dir in data_directories:
            each_dir_base= os.path.basename(each_dir)
            if "_BONE" in each_dir_base: 
                each_dir_base_splitted=each_dir_base.split("_BONE")
                each_dir_base=each_dir_base_splitted[0]
            ## nii file pattern
            nii_file_name=each_dir_base+".nii"
            ## AV ROI file pattern
            avroi_file_name=each_dir_base+"_AV.hdr"
            ## V ROI file pattern
            vroi_file_name=each_dir_base+"_V.hdr"
            if os.path.exists(os.path.join(each_dir, nii_file_name)) and os.path.exists(os.path.join(each_dir, avroi_file_name)) and os.path.exists(os.path.join(each_dir, vroi_file_name)):
                ##         #load analyze AV:
                avroi_file_name=each_dir_base+"_" + TYPE_OF_SLICE + ".hdr"
                av_img = nib.AnalyzeImage.from_filename(os.path.join(each_dir, avroi_file_name))
                av_img_data = av_img.get_fdata()
                av_img_data.astype(float)
                av_img_data = np.asarray(av_img_data)      
                av_img_data=normalizeimage(av_img_data)
                nii_img = nib.load(os.path.join(each_dir, nii_file_name))
                affine=nii_img.affine
                nii_img_data = nii_img.get_fdata()
                nii_img_data.astype(float)
                nii_img_data = np.asarray(nii_img_data)
                if nii_img_data.shape[0]==512 and nii_img_data.shape[1]==512:
#                    nii_img_data=normalizeimage0to1(nii_img_data)
#                    nii_BET_mask=nib.load(os.path.join(BET_MASK_DIRECTORY, nii_file_name+".gz"))
#                    nii_BET_mask_data = nii_BET_mask.get_fdata()
                    for xx in range(0,av_img_data.shape[2]):
                        av_img_data_slice= av_img_data[:,:,xx]
                        av_img_data_slice = cv2.resize(av_img_data_slice, (IMAGE_SIZE_TO, IMAGE_SIZE_TO))
                        nii_img_data_slice=nii_img_data[:,:,xx]
#                        nii_BET_mask_data_slice=nii_BET_mask_data[:,:,xx]
#                        nii_img_data_slice=nii_img_data_slice*nii_BET_mask_data_slice
                        nii_img_data_slice = cv2.resize(nii_img_data_slice, (IMAGE_SIZE_TO, IMAGE_SIZE_TO))
                        av_img_data_slice_max=np.max(av_img_data_slice)
                        if av_img_data_slice_max > 0: 
                            if counter ==0:
                                print(xx)
                                print(each_dir_base)
                                counter=counter+1
                            bright_points = np.transpose(np.nonzero(av_img_data_slice))
                            if(len(bright_points)==2):
                                image_name= each_dir_base + str(xx)#  + ".h5"  #str(image_number) 
                                
                                SAVE_RESULT_DIR_GRAY_this=os.path.join(SAVE_RESULT_DIR_GRAY,TYPE_OF_SLICE)
                                numpy2dmat2niigz(nii_img_data_slice,affine,SAVE_RESULT_DIR_GRAY_this,image_name)
                        else :
                            SAVE_RESULT_DIR_GRAY_this=os.path.join(SAVE_RESULT_DIR_GRAY,"BV")
                            image_name= each_dir_base + str(xx) #+ ".h5"  #str(image_number) 
                            numpy2dmat2niigz(nii_img_data_slice,affine,SAVE_RESULT_DIR_GRAY_this,image_name)
                    
    return "XX"
def niigzstackasmnistdata_yasheng_data(imagestack,image_size=64):
    x_train=[]
    y_train=[]
    #affine=np.diag([1, 2, 3, 1])
#    min_number_files=10000
#    for thisdirectory in range(len(directoriesname)):
#        filenames=glob.glob(directoriesname[thisdirectory]+"/*.nii.gz")
#        if min_number_files > len(filenames):
#            min_number_files=len(filenames)
##        imsagesuperstack=[]
#       #imagestack=[]
#        counter = 1
       # hf = h5py.File(os.path.join(directoryname,"/" + savemat_name+'.h5'), 'w')
    for i in range(0,imagestack.shape[2]): #filenames:
    #            if file % 5 ==0:
    #            thisfile=os.path.join(directoriesname[thisdirectory],filenames[file])
    #            img = nib.load(thisfile)
    #affine=img.affine
        img_data = cv2.resize(imagestack[:,:,i], (image_size, image_size))
        img_data=exposure.rescale_intensity(img_data, in_range=(0, 200))
        #                img_data[img_data<0]=0
        #                img_data[img_data>1900]=1900
        x_train.append(img_data)
        y_train.append(0)
    #x_train.append(imagestack)
    #print(y_train)   
    return (np.array(x_train),np.array(y_train))

def niigzstackasmnistdata_1(imagestack,image_size=64):
    x_train=[]
    y_train=[]
    #affine=np.diag([1, 2, 3, 1])
#    min_number_files=10000
#    for thisdirectory in range(len(directoriesname)):
#        filenames=glob.glob(directoriesname[thisdirectory]+"/*.nii.gz")
#        if min_number_files > len(filenames):
#            min_number_files=len(filenames)
##        imsagesuperstack=[]
#       #imagestack=[]
#        counter = 1
       # hf = h5py.File(os.path.join(directoryname,"/" + savemat_name+'.h5'), 'w')
    for i in range(0,imagestack.shape[2]): #filenames:
    #            if file % 5 ==0:
    #            thisfile=os.path.join(directoriesname[thisdirectory],filenames[file])
    #            img = nib.load(thisfile)
    #affine=img.affine
        img_data = cv2.resize(imagestack[:,:,i], (image_size, image_size))
        img_data=exposure.rescale_intensity(img_data, in_range=(0, 200))
        #                img_data[img_data<0]=0
        #                img_data[img_data>1900]=1900
        x_train.append(img_data)
        y_train.append(0)
    #x_train.append(imagestack)
    #print(y_train)   
    return (np.array(x_train),np.array(y_train))

def niigzstackasmnistdata(directoriesname,image_size=64):
    x_train=[]
    y_train=[]
    #affine=np.diag([1, 2, 3, 1])
    min_number_files=10000
    for thisdirectory in range(len(directoriesname)):
        filenames=glob.glob(directoriesname[thisdirectory]+"/*.nii.gz")
        if min_number_files > len(filenames):
            min_number_files=len(filenames)
#        imsagesuperstack=[]
       #imagestack=[]
        counter = 1
       # hf = h5py.File(os.path.join(directoryname,"/" + savemat_name+'.h5'), 'w')
        for file in range(0,min_number_files): #filenames:
            if file % 5 ==0:
                thisfile=os.path.join(directoriesname[thisdirectory],filenames[file])
                img = nib.load(thisfile)
                #affine=img.affine
                img_data = cv2.resize(img.get_fdata(), (image_size, image_size))
                img_data=exposure.rescale_intensity(img_data, in_range=(0, 200))
#                img_data[img_data<0]=0
#                img_data[img_data>1900]=1900
                x_train.append(img_data)
                y_train.append(thisdirectory)
        #x_train.append(imagestack)
    #print(y_train)   
    return (np.array(x_train),np.array(y_train))
            #print(img)
       # hf.create_dataset('imagestack', data=imagestack)
       # hf.close()
def niigzstackasmnistdata_call(TYPES_OF_SLICE,TYPES_OF_DATA,RAW_DATA_FOLDER,image_size=64):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    for iii in range(0,len(TYPES_OF_SLICE)):
            for xxx in range(0,len(TYPES_OF_DATA)):
    #            COMMON_DIR=RAW_DATA_FOLDER#RAW_DATA_FOLDER #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/Ideal/INBRAIN"
    #            IML_directory=os.path.join(COMMON_DIR,TYPES_OF_DATA[xxx]) # "/TEST"
    #            COMMON_DIR=os.path.join(COMMON_DIR,TYPES_OF_DATA[xxx])
                IML_directory1=os.path.join(RAW_DATA_FOLDER,TYPES_OF_DATA[xxx]) 
                dir_AV=IML_directory1 +'/AV'
                dir_V=IML_directory1 +'/V'
                dir_BV=IML_directory1 +'/BV'
#                display_h5_matrix(dir_gray)
    #                display_h5_matrix(dir_ant)
    #                display_h5_matrix(dir_post)
#                command="rm   " + os.path.join(IML_directory1,'grayscalestackAV.h5')
#                subprocess.call(command,shell=True)
#                command="rm   " + os.path.join(IML_directory1,'grayscalestackV.h5')
#                subprocess.call(command,shell=True)
#                command="rm   " + os.path.join(IML_directory1,'grayscalestackBV.h5')
#                subprocess.call(command,shell=True)
#                command="rm   " + os.path.join(IML_directory1,'grayscalestackAV.nii.gz')
#                subprocess.call(command,shell=True)
#                command="rm   " + os.path.join(IML_directory1,'grayscalestackV.nii.gz')
#                subprocess.call(command,shell=True)
#                command="rm   " + os.path.join(IML_directory1,'grayscalestackBV.nii.gz')
#                subprocess.call(command,shell=True)
                if TYPES_OF_DATA[xxx]=="TRAIN":
                    (x_train,y_train) = niigzstackasmnistdata((dir_AV,dir_V),image_size)
#                    (x_train,y_train) = niigzstackasmnistdata((dir_AV,dir_V,dir_BV),image_size)
                if TYPES_OF_DATA[xxx]=="TEST":
                    (x_test,y_test) = niigzstackasmnistdata((dir_AV,dir_V),image_size)
#                    (x_test,y_test) = niigzstackasmnistdata((dir_AV,dir_V,dir_BV),image_size)
                    
    return (x_train,y_train), (x_test,y_test)
#                hf = h5py.File(os.path.join(IML_directory1,'grayscalestackAV.h5'), 'w')
#                hf.create_dataset('imagestack', data=imagestack)
#                hf.close()
#                imagestack= niigzstackasmnistdata(dir_V)
#                hf = h5py.File(os.path.join(IML_directory1,'grayscalestackV.h5'), 'w')
#                hf.create_dataset('imagestack', data=imagestack)
#                hf.close()
#                imagestack= niigzstackasmnistdata(dir_BV)
#                hf = h5py.File(os.path.join(IML_directory1,'grayscalestackBV.h5'), 'w')
#                hf.create_dataset('imagestack', data=imagestack)
#                hf.close()  
def display_point_3d(point):
    points = vtk.vtkPoints()
    p = [point[0], point[1], point[2]]
    
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    
    id = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(id)
    
    # Create a polydata object
    point = vtk.vtkPolyData()
    
    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(points)
    point.SetVerts(vertices)
    
    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(point)
    else:
        mapper.SetInputData(point)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(20)
    return actor
def numpy2VTK(img,spacing=[1.0,1.0,1.0]):
    # evolved from code from Stou S.,
    # on http://www.siafoo.net/snippet/314
    importer = vtk.vtkImageImport()
    
    img_data = img.astype('uint8')
    img_string = img_data.tostring() # type short
    dim = img.shape
    
    importer.CopyImportVoidPointer(img_string, len(img_string))
    importer.SetDataScalarType(VTK_UNSIGNED_CHAR)
    importer.SetNumberOfScalarComponents(1)
    
    extent = importer.GetDataExtent()
    importer.SetDataExtent(extent[0], extent[0] + dim[2] - 1,
                           extent[2], extent[2] + dim[1] - 1,
                           extent[4], extent[4] + dim[0] - 1)
    importer.SetWholeExtent(extent[0], extent[0] + dim[2] - 1,
                            extent[2], extent[2] + dim[1] - 1,
                            extent[4], extent[4] + dim[0] - 1)

    importer.SetDataSpacing( spacing[0], spacing[1], spacing[2])
    importer.SetDataOrigin( 0,0,0 )

    return importer

def volumeRender(img, tf=[],spacing=[1.0,1.0,1.0],color_factor=1):
    importer = numpy2VTK(img,spacing)

    # Transfer Functions
    opacity_tf = vtk.vtkPiecewiseFunction()
    color_tf = vtk.vtkColorTransferFunction()

    if len(tf) == 0:
        tf.append([img.min(),0,0,0,0])
        tf.append([img.max(),1,1,1,1])

    for p in tf:
        color_tf.AddRGBPoint(p[0], p[1]*color_factor, p[2], p[3])
        opacity_tf.AddPoint(p[0], p[4])

    # working on the GPU
    volMapper = vtk.vtkGPUVolumeRayCastMapper()
    volMapper.SetInputConnection(importer.GetOutputPort())
    
    # The property describes how the data will look
    volProperty =  vtk.vtkVolumeProperty()
    volProperty.SetColor(color_tf)
    volProperty.SetScalarOpacity(opacity_tf)
    volProperty.ShadeOn()
    volProperty.SetInterpolationTypeToLinear()

#    # working on the CPU
#    volMapper = vtk.vtkVolumeRayCastMapper()
#    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
#    compositeFunction.SetCompositeMethodToInterpolateFirst()
#    volMapper.SetVolumeRayCastFunction(compositeFunction)
#    volMapper.SetInputConnection(importer.GetOutputPort())
#
#    # The property describes how the data will look
#    volProperty =  vtk.vtkVolumeProperty()
#    volProperty.SetColor(color_tf)
#    volProperty.SetScalarOpacity(opacity_tf)
#    volProperty.ShadeOn()
#    volProperty.SetInterpolationTypeToLinear()
    
    # Do the lines below speed things up?
    # pix_diag = 5.0
    # volMapper.SetSampleDistance(pix_diag / 5.0)    
    # volProperty.SetScalarOpacityUnitDistance(pix_diag) 
    

    vol = vtk.vtkVolume()
    vol.SetMapper(volMapper)
    vol.SetProperty(volProperty)
    
    return [vol]


def vtk_basic( actors ):
    """
    Create a window, renderer, interactor, add the actors and start the thing
    
    Parameters
    ----------
    actors :  list of vtkActors
    
    Returns
    -------
    nothing
    """     
    
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(600,600)
    # ren.SetBackground( 1, 1, 1)
 
    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    for a in actors:
        # assign actor to the renderer
        ren.AddActor(a )
    
    # render
    renWin.Render()
   
    # enable user interface interactor
    iren.Initialize()
    iren.Start()

        


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def thresh_callback(grayimage,val,src_gray,xx,point_thresh):

    threshold = val
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    #print("HERE")
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    ellipse=[None]*len(contours)
    centers=[None]*len(contours)
    rects_min=[None]*len(contours)
    #print(len(contours_poly))
    #if len(contours_poly)>10:
    
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
      #  print(contours[i])
       # print(len(c))
        if len(contours[i]) >point_thresh:
       # boundRect[i] = cv.boundingRect(contours_poly[i])
       # centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
            ellipse[i] = cv.fitEllipse(contours[i])
            rects_min[i] = cv2.minAreaRect(contours[i])
           # (x,y), (MA,ma), angle = cv.fitEllipse(contours[i])
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    #drawing = grayimage # np.zeros((grayimage.shape[0], grayimage.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        if len(contours[i]) >point_thresh:
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours_poly, i, color,3)
            #print(np.int0(ellipse[i][0]))
            cv.circle(drawing,(np.int0(ellipse[i][0][0]),np.int0(ellipse[i][0][1])), 3, (0,0,255), -1)
           # cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            #  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            #cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
            cv.ellipse(drawing,ellipse[i],(0,255,0),2)
            x,y=rotate((0,0), (0,ellipse[i][1][0]), math.radians(ellipse[i][2]))
           # cv.circle(drawing,(np.int0(x+ellipse[i][0][0]),np.int0(y+ellipse[i][0][1])), 3, (0,0,255), -1)
            x1,y1=rotate((0,0), (ellipse[i][1][1],0), math.radians(ellipse[i][2]))
            x2,y2=rotate((0,0), (0,-ellipse[i][1][0]), math.radians(ellipse[i][2]))
            x3,y3=rotate((0,0), (-ellipse[i][1][1],0), math.radians(ellipse[i][2])) 
            #cv.circle(drawing,(np.int0(x1+ellipse[i][0][0]),np.int0(y1+ellipse[i][0][1])), 3, (0,0,255), -1)
            cv2.line(drawing,(np.int0(ellipse[i][0][0]),np.int0(ellipse[i][0][1])),(np.int0(x+ellipse[i][0][0]),np.int0(y+ellipse[i][0][1])),(255,0,0),2)
            cv2.line(drawing,(np.int0(ellipse[i][0][0]),np.int0(ellipse[i][0][1])),(np.int0(x1+ellipse[i][0][0]),np.int0(y1+ellipse[i][0][1])),(255,0,0),2)
            cv2.line(drawing,(np.int0(ellipse[i][0][0]),np.int0(ellipse[i][0][1])),(np.int0(x2+ellipse[i][0][0]),np.int0(y2+ellipse[i][0][1])),(255,0,0),2)
            cv2.line(drawing,(np.int0(ellipse[i][0][0]),np.int0(ellipse[i][0][1])),(np.int0(x3+ellipse[i][0][0]),np.int0(y3+ellipse[i][0][1])),(255,0,0),2)

#            box = cv2.boxPoints(rects_min[i])
#            box = np.int0(box/2)
#            cv.drawContours(drawing,[box],0,(0,0,255),2)
#   
    #ellipse[i]
    dst = cv.addWeighted(drawing, 0.5, grayimage, 0.5, 0.0)
#    cv.imwrite(str(xx)+".jpg",dst)
    cv.imshow('Contours', dst)
    
#parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
#parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
#args = parser.parse_args()
def plot_contours(grayimage,src,xx,point_thresh):
#    #cv.samples.findFile(args.input))
    #if src is None:
    #    print('Could not open or find the image:', args.input)
    #    exit(0)
    # Convert image to gray and blur it
    src_gray = src # cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #src_gray = cv.blur(src_gray, (3,3))
    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)
    max_thresh = 255
    thresh = 200 # initial threshold
    cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(grayimage,thresh,src_gray,xx,point_thresh)
    cv.waitKey()
    

def plot_contours_call_1(RAW_DATA_FOLDER,BET_DATA_FOLDER,filename,filename_gray,filename_BET):
    nii_img = nib.load(os.path.join(BET_DATA_FOLDER, filename))
    nii_img_data = nii_img.get_fdata()
    nii_img_data.astype(float)
    nii_img_data = np.asarray(nii_img_data)
    
    nii_img_gray = nib.load(os.path.join(RAW_DATA_FOLDER, filename_gray))
    nii_img_data_gray = nii_img_gray.get_fdata()
    nii_img_data_gray.astype(float)
    nii_img_data_gray = np.asarray(nii_img_data_gray)
    
    nii_img_gray_bet = nib.load(os.path.join(BET_DATA_FOLDER, filename_BET))
    nii_img_data_gray_bet  = nii_img_gray_bet.get_fdata()
    nii_img_data_gray_bet.astype(float)
    nii_img_data_gray_bet = np.asarray(nii_img_data_gray_bet)
    for xx in range(0,nii_img_data.shape[2]):
        
        src = nii_img_data[:,:,xx] #cv.imread("Bounding_Rects_Circles_Source_Image.jpg")
        grayscale_image= nii_img_data_gray[:,:,xx] 
        grayscale_image_bet= nii_img_data_gray_bet[:,:,xx] 
        ## get bones only:
    #    grayscale_image[grayscale_image<np.min(grayscale_image)+1]=0
    #    grayscale_image[grayscale_image>1900]=1900
        grayscale_image=exposure.rescale_intensity(grayscale_image, in_range=(0,200)) #950, 950+200))
        if(np.sum(src)>5):
            src=(src- np.min(src)) /(np.max(src)-np.min(src)) * 255
            grayscale_image=(grayscale_image- np.min(grayscale_image)) /(np.max(grayscale_image)-np.min(grayscale_image)) * 255
            #grayscale_image = cv2.convertScaleAbs(grayscale_image-np.min(grayscale_image), alpha=(255.0 / min(np.max(grayscale_image)-np.min(grayscale_image), 10000)))
            grayscale_image1 = np.zeros(shape=(grayscale_image.shape[0],grayscale_image.shape[1],3)).astype(np.float32)
            grayscale_image1[:,:,0]=grayscale_image
            grayscale_image1[:,:,1]=grayscale_image
            grayscale_image1[:,:,2]=grayscale_image
    #        src=(src- np.min(src)) /(np.max(src)-np.min(src)) * 255
            #img = src#*post_point_mask_reg #plt.imread("test.png")
           # print(np.max(src))
    
    #        fig, ax = plt.subplots(num="MRI_demo")
    #        ax.imshow(src, cmap="gray")
    #        ax.axis('off') 
    #        plt.show()
            kernel = np.ones((3,3),np.uint8)
    
            src=src.astype(np.uint8)
            grayscale_image=grayscale_image.astype(np.uint8)
            grayscale_image1=grayscale_image1.astype(np.uint8)
#            src = cv2.erode(src,kernel,iterations = 3)
#            src = cv2.dilate(src,kernel,iterations = 3)
            #cv.imwrite('src.jpg', src)
            #src=cv.imread()
            point_thresh=25
            plot_contours(grayscale_image1,src,xx,point_thresh)    
    
    
def plot_contours_call(RAW_DATA_FOLDER,filename,filename_gray,filename_BET):
    nii_img = nib.load(os.path.join(RAW_DATA_FOLDER, filename))
    nii_img_data = nii_img.get_fdata()
    nii_img_data.astype(float)
    nii_img_data = np.asarray(nii_img_data)
    
    nii_img_gray = nib.load(os.path.join(RAW_DATA_FOLDER, filename_gray))
    nii_img_data_gray = nii_img_gray.get_fdata()
    nii_img_data_gray.astype(float)
    nii_img_data_gray = np.asarray(nii_img_data_gray)
    
    nii_img_gray_bet = nib.load(os.path.join(RAW_DATA_FOLDER, filename_BET))
    nii_img_data_gray_bet  = nii_img_gray_bet.get_fdata()
    nii_img_data_gray_bet.astype(float)
    nii_img_data_gray_bet = np.asarray(nii_img_data_gray_bet)
    for xx in range(0,nii_img_data.shape[2]):
        
        src = nii_img_data[:,:,xx] #cv.imread("Bounding_Rects_Circles_Source_Image.jpg")
        grayscale_image= nii_img_data_gray[:,:,xx] 
        grayscale_image_bet= nii_img_data_gray_bet[:,:,xx] 
        ## get bones only:
    #    grayscale_image[grayscale_image<np.min(grayscale_image)+1]=0
    #    grayscale_image[grayscale_image>1900]=1900
        grayscale_image=exposure.rescale_intensity(grayscale_image, in_range=(950, 950+200))
        if(np.sum(src)>5):
            src=(src- np.min(src)) /(np.max(src)-np.min(src)) * 255
            grayscale_image=(grayscale_image- np.min(grayscale_image)) /(np.max(grayscale_image)-np.min(grayscale_image)) * 255
            #grayscale_image = cv2.convertScaleAbs(grayscale_image-np.min(grayscale_image), alpha=(255.0 / min(np.max(grayscale_image)-np.min(grayscale_image), 10000)))
            grayscale_image1 = np.zeros(shape=(grayscale_image.shape[0],grayscale_image.shape[1],3)).astype(np.float32)
            grayscale_image1[:,:,0]=grayscale_image
            grayscale_image1[:,:,1]=grayscale_image
            grayscale_image1[:,:,2]=grayscale_image
    #        src=(src- np.min(src)) /(np.max(src)-np.min(src)) * 255
            #img = src#*post_point_mask_reg #plt.imread("test.png")
           # print(np.max(src))
    
    #        fig, ax = plt.subplots(num="MRI_demo")
    #        ax.imshow(src, cmap="gray")
    #        ax.axis('off') 
    #        plt.show()
            kernel = np.ones((3,3),np.uint8)
    
            src=src.astype(np.uint8)
            grayscale_image=grayscale_image.astype(np.uint8)
            grayscale_image1=grayscale_image1.astype(np.uint8)
#            src = cv2.erode(src,kernel,iterations = 3)
#            src = cv2.dilate(src,kernel,iterations = 3)
            #cv.imwrite('src.jpg', src)
            #src=cv.imread()
            point_thresh=25
            plot_contours(grayscale_image1,src,xx,point_thresh)
        
def erodeNdilate(directoryname):
    for eachfile in glob.glob(directoryname+"/*.nii.gz"):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(eachfile)
        image = reader.Execute();
        
        image = sitk.BinaryErode(image!=0, 2)
        image = sitk.BinaryDilate(image!=0, 3)
        filter1=sitk.BinaryFillholeImageFilter()
        filter1.SetForegroundValue(1)
        filter1.SetFullyConnected(True)
        image=filter1.Execute(image)
        image=sitk.VotingBinaryIterativeHoleFilling(image,sitk.VectorUInt32(100) )
#        image = sitk.BinaryErode(image!=0, 3)
#        image = sitk.BinaryDilate(image!=0, 4)
#        image=sitk.BinaryFillhole(image,fullyConnected=True)
#        image=sitk.BinaryFillhole(image,fullyConnected=True)
#        image=sitk.BinaryFillhole(image,fullyConnected=True)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(eachfile)
        writer.Execute(image)
#        sitk.Show(reader)
        print(eachfile)
    return image
def applyBETtoniigz(directoryname,directorynameprediction):
    for eachfile in glob.glob(directoryname+"/*.nii.gz"):
        filebasename=os.path.basename(eachfile)
        mask_filename=os.path.join(directorynameprediction,filebasename)
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(eachfile)
        image = reader.Execute();
        reader1 = sitk.ImageFileReader()
        reader1.SetImageIO("NiftiImageIO")
        reader1.SetFileName(mask_filename)
        image1 = reader.Execute();
        image=image*image1
        writer = sitk.ImageFileWriter()
        writer.SetFileName(eachfile)
        writer.Execute(image)
#        sitk.Show(reader)
        print(eachfile)
    return image
        
   
#directoryname="/media/atul/AC0095E80095BA32/WASHU_WORK/THIRD_PARTY/CT_BET-1/results_folder/_2019717_94641/predictions"
#directorynameprediction="/media/atul/AC0095E80095BA32/WASHU_WORK/THIRD_PARTY/CT_BET-1/results_folder/_2019717_94641/predictions"
#image=erodeNdilate(directoryname)
#display_2dnumpystack(sitk.GetArrayFromImage(image))
#applyBETtoniigz(directoryname,directorynameprediction)
#print(image)
## find the coordinates of all the bright points:

## find the centroid of the points coordinates:

## filter the border: total connected pixel threshold + distance from the centroid threshold

## Get the slice with the bright area

## remove the outer CSF area


## create a bounding box around the bright area

## draw  the midpoint of the bounding box.
    
def call_with_dryasheng_data():
    POINTS_LOCATION=["INBRAIN","ONBONE"]#,
    TYPES_OF_DATA=["TRAIN","TEST"]
    TYPES_OF_SLICE=["AV", "V"]
    RAW_DATA_FOLDER='/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CSF_SEGMENTATION/' # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/BET_RESULT" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/BET_RESULT" 
    AUGMENTED_DATA_FOLDER="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/unetcodes"
    CODE_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/THIRD_PARTY/Selfie_Filters_OpenCV"
    ## READ CSF segmented  image file
    filename='WUSTL_166_05022015_1332_final_seg.nii.gz' #'ct_20130524_1011_101232_j40s2.nii.gz'
    filename_gray='WUSTL_166_05022015_1332.nii.gz' #'ct_20130524_1011_101232_j40s2_gray.nii.gz'
    filename_BET='WUSTL_166_05012015_1332_levelset_bet.nii.gz' # 'WUSTL_166_05012015_1535_final_seg.nii.gz'
    
    ## find unique file ids:
    allfiles =glob.glob(RAW_DATA_FOLDER+ "/*.nii.gz")
    unique_names_files_pattern=set()
    for files in allfiles:
        thisfile=os.path.basename(files)
        thisfile=thisfile.split('.')
        thisfile=thisfile[0].split('_')
        thisfileuniqueid=thisfile[0] + '_' + thisfile[1] + '_' + thisfile[2] +'_' +  thisfile[3] 
        unique_names_files_pattern.add(thisfileuniqueid)
    unique_names_files_pattern=list(unique_names_files_pattern) 
#    print(unique_names_files_pattern)
    
    for each_unique_names_file_pattern in unique_names_files_pattern:
        each_unique_names_file=glob.glob(RAW_DATA_FOLDER+ "/" +each_unique_names_file_pattern+ "*.nii.gz")
        for each_unique_file in each_unique_names_file:
            each_unique_file=os.path.basename(each_unique_file)
            each_unique_names_file_split=each_unique_file[:-7]
    #        print(each_unique_names_file_split)
            filename = each_unique_names_file_split + '_final_seg.nii.gz'
            filename_gray = each_unique_names_file_split + '_levelset.nii.gz'
            filename_BET = each_unique_names_file_split + '_levelset_bet.nii.gz'
            if os.path.exists(os.path.join(RAW_DATA_FOLDER,filename)) and os.path.exists(os.path.join(RAW_DATA_FOLDER,filename_gray)) and os.path.exists(os.path.join(RAW_DATA_FOLDER,filename_BET)):
                print("YES")
#                plot_contours_call(RAW_DATA_FOLDER,filename,filename_gray,filename_BET)
                plot_contours_call(RAW_DATA_FOLDER,filename_BET,filename_gray,filename_BET)




def call_with_zach_data():
    POINTS_LOCATION=["INBRAIN","ONBONE"]#,
    TYPES_OF_DATA=["TRAIN","TEST"]
    TYPES_OF_SLICE=["AV", "V"]
    GRAYSCALE_FOLDER="/media/atul/AC0095E80095BA32/WASHU_WORK/THIRD_PARTY/CT_BET-1/image_data"
    BET_MASK_FOLDER="/media/atul/AC0095E80095BA32/WASHU_WORK/THIRD_PARTY/CT_BET-1/results_folder/unet_CT_SS_2019718_151930/predictions"
#    RAW_DATA_FOLDER='/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CSF_SEGMENTATION/' # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/BET_RESULT" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/BET_RESULT" 
#    AUGMENTED_DATA_FOLDER="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/unetcodes"
#    CODE_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/THIRD_PARTY/Selfie_Filters_OpenCV"
    ## READ CSF segmented  image file
    filename='WUSTL_166_05022015_1332_final_seg.nii.gz' #'ct_20130524_1011_101232_j40s2.nii.gz'
    filename_gray='WUSTL_166_05022015_1332.nii.gz' #'ct_20130524_1011_101232_j40s2_gray.nii.gz'
    filename_BET='WUSTL_166_05012015_1332_levelset_bet.nii.gz' # 'WUSTL_166_05012015_1535_final_seg.nii.gz'
    
    ## find unique file ids:
    allfiles =glob.glob(GRAYSCALE_FOLDER+ "/*.nii.gz")
    for files in allfiles:
        thisfile=os.path.basename(files)
#    unique_names_files_pattern=set()
#    for files in allfiles:
#        thisfile=os.path.basename(files)
#        thisfile=thisfile.split('.')
#        thisfile=thisfile[0].split('_')
#        thisfileuniqueid=thisfile[0] + '_' + thisfile[1] + '_' + thisfile[2] +'_' +  thisfile[3] 
#        unique_names_files_pattern.add(thisfileuniqueid)
#    unique_names_files_pattern=list(unique_names_files_pattern) 
##    print(unique_names_files_pattern)
    
#    for each_unique_names_file_pattern in allfiles : #unique_names_files_pattern:
#        each_unique_names_file=glob.glob(RAW_DATA_FOLDER+ "/" +each_unique_names_file_pattern+ "*.nii.gz")
#        for each_unique_file in each_unique_names_file:
#            each_unique_file=os.path.basename(each_unique_file)
#            each_unique_names_file_split=each_unique_file[:-7]
    #        print(each_unique_names_file_split)
        filename = os.path.join(BET_MASK_FOLDER,thisfile) #each_unique_names_file_split + '_final_seg.nii.gz'
        filename_gray =os.path.join(GRAYSCALE_FOLDER,thisfile)# each_unique_names_file_split + '_levelset.nii.gz'
        filename_BET = os.path.join(BET_MASK_FOLDER,thisfile)#each_unique_names_file_split + '_levelset_bet.nii.gz'
        filename_AV = os.path.join(GRAYSCALE_FOLDER,thisfile[:-7]+"_AV_mask.nii.gz")#each_unique_names_file_split + '_levelset_bet.nii.gz'
        filename_V = os.path.join(GRAYSCALE_FOLDER,thisfile[:-7]+"_AV_mask.nii.gz")#each_unique_names_file_split + '_levelset_bet.nii.gz'
        if os.path.exists(os.path.join(BET_MASK_FOLDER,filename)) and os.path.exists(os.path.join(GRAYSCALE_FOLDER,filename_gray)) and os.path.exists(os.path.join(BET_MASK_FOLDER,filename_BET)):
            print("YES")
#                plot_contours_call(RAW_DATA_FOLDER,filename,filename_gray,filename_BET)
            plot_contours_call_1(GRAYSCALE_FOLDER,BET_MASK_FOLDER,filename_BET,filename_gray,filename_BET)
 
def image_plane_vtk_getplane(numpy_image,i, rgbFrame=None):
    #Build vtkImageData here from the given numpy uint8_t arrays.
    __depthImageData = vtk.vtkImageData()
    depthArray = numpy_support.numpy_to_vtk(numpy_image.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR) 
    # .transpose(2, 0, 1) may be required depending on numpy array order see - https://github.com/quentan/Test_ImageData/blob/master/TestImageData.py
    
    __depthImageData.SetDimensions(numpy_image.shape[0],numpy_image.shape[1],1)
    #assume 0,0 origin and 1,1 spacing.
    __depthImageData.SetSpacing([1,1,1])
    __depthImageData.SetOrigin([0,0,0])
    __depthImageData.GetPointData().SetScalars(depthArray)
    aPlane =  vtkPlaneSource()
#    aPlane.SetXResolution(512);
#    aPlane.SetYResolution(512);
    aPlane.SetCenter(0,0,i)
    aPlane.SetNormal(0,0,1)
    aPlane.Update();
    transform = vtk.vtkTransform()
    transform.Translate(256, 256, 0)
    transform.Scale(512, 512, 1)

    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()

    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    
    #   depthFrame=np.zeros([512,512])
    #   depthFrame[:,1:100]=255
    texture = vtkTexture()  
    #    writer=vtkJPEGWriter()
    
    texture.SetInputData(__depthImageData) #image_data) #Connection(imageSource.GetOutputPort());
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) #appendFilter.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
#    actor.GetProperty().SetPointSize(5)
    actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return transformPD, actor #aPlane, texture #__depthImageData

def image_plane_vtk(numpy_image,i, rgbFrame=None):
    #Build vtkImageData here from the given numpy uint8_t arrays.
    numpy_imagecopy=np.copy(numpy_image)
#    input("ENTER")
#    print("FLIPPING DONE")
#    numpy_imagecopy = cv2.flip( numpy_imagecopy, -1 )
    __depthImageData = vtk.vtkImageData()
    depthArray = numpy_support.numpy_to_vtk(numpy_imagecopy.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR) 
    # .transpose(2, 0, 1) may be required depending on numpy array order see - https://github.com/quentan/Test_ImageData/blob/master/TestImageData.py
    
    __depthImageData.SetDimensions(numpy_imagecopy.shape[0],numpy_imagecopy.shape[1],1)
    #assume 0,0 origin and 1,1 spacing.
    __depthImageData.SetSpacing([1,1,1])
    __depthImageData.SetOrigin([0,0,0])
    __depthImageData.GetPointData().SetScalars(depthArray)
    aPlane =  vtkPlaneSource()
    aPlane.SetXResolution(512);
    aPlane.SetYResolution(512);
    aPlane.SetCenter(0,0,i)
    aPlane.SetNormal(0,0,1)
    aPlane.Update();
    transform = vtk.vtkTransform()
    transform.Translate(256, 256, 0)
    transform.Scale(512, 512, 1)

    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()

    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    
    #   depthFrame=np.zeros([512,512])
    #   depthFrame[:,1:100]=255
    texture = vtkTexture()  
    #    writer=vtkJPEGWriter()
    
    texture.SetInputData(__depthImageData) #image_data) #Connection(imageSource.GetOutputPort());
    
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) #appendFilter.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)
    actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return  actor #aPlane, texture #__depthImageData

def updateFrames(depthFrame,i, rgbFrame=None):
   #Build vtkImageData here from the given numpy uint8_t arrays.
   __depthImageData = vtk.vtkImageData()
   depthArray = numpy_support.numpy_to_vtk(depthFrame.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR) 
   # .transpose(2, 0, 1) may be required depending on numpy array order see - https://github.com/quentan/Test_ImageData/blob/master/TestImageData.py

   __depthImageData.SetDimensions(depthFrame.shape[0],depthFrame.shape[1],1)
  #assume 0,0 origin and 1,1 spacing.
   __depthImageData.SetSpacing([1,1,1])
   __depthImageData.SetOrigin([0,0,0])
   __depthImageData.GetPointData().SetScalars(depthArray)
   aPlane =  vtkPlaneSource()
   aPlane.SetXResolution(512);
   aPlane.SetYResolution(512);
   aPlane.SetCenter(0,0,i+10)
   aPlane.Update();
   depthFrame=np.zeros([512,512])
   depthFrame[:,1:100]=255
   texture = vtkTexture()  

   texture.SetInputData(__depthImageData) #image_data) #Connection(imageSource.GetOutputPort());
   
   mapper = vtk.vtkDataSetMapper()
   mapper.SetInputConnection(aPlane.GetOutputPort()) #appendFilter.GetOutputPort())
    
   actor = vtk.vtkActor()
   actor.SetMapper(mapper)
   actor.GetProperty().SetPointSize(5)
   actor.SetTexture(texture);

   
   return actor #aPlane, texture #__depthImageData

def draw_points(p,renderer,color):
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    
    id = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(id)
    
    point = vtk.vtkPolyData()
    
    point.SetPoints(points)
    point.SetVerts(vertices)
    

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(point)
    else:
        mapper.SetInputData(point)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(20)
    actor.GetProperty().SetColor(color[0],color[1],color[2])
    renderer.AddActor(actor)
    return actor
def draw_points1(p,renderer,color):
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    
    id = points.InsertNextPoint(p)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(id)
    
    point = vtk.vtkPolyData()
    
    point.SetPoints(points)
    point.SetVerts(vertices)
    

    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(point)
    else:
        mapper.SetInputData(point)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(20)
    actor.GetProperty().SetColor(color[0],color[1],color[2])
    renderer.AddActor(actor)
    return actor

def draw_plane_with_points(points):
    aPlane =  vtkPlaneSource()
#    aPlane.SetXResolution(512);
#    aPlane.SetYResolution(512);
    center=(points[0]+ points[1])/2
    aPlane.SetCenter(center[0],center[1],center[2])
    aPlane.SetPoint1(points[2])
    aPlane.SetPoint1(points[3])
    aPlane.Update();
    transform = vtk.vtkTransform()
#    transform.Translate(center[0],center[1],center[2])
    transform.Scale(512, 512, 1)
    
    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    texture = vtkTexture()  
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor #aPlane, texture #__depthImageData
def intersectionofplanes_polydata(input1,input2):
#    input1 = plane1.GetOutput() #sphereSource1.GetOutput()
    sphere1Tri = vtk.vtkTriangleFilter()
    sphere1Tri.SetInputData(input1)
    
#    input2 = plane2.GetOutput()  #sphereSource2.GetOutput()
    sphere2Tri = vtk.vtkTriangleFilter()
    sphere2Tri.SetInputData(input2)

    
    booleanOperation = vtk.vtkIntersectionPolyDataFilter()

    booleanOperation.SetInputConnection(0, sphere1Tri.GetOutputPort())
    booleanOperation.SetInputConnection(1, sphere2Tri.GetOutputPort())

    booleanOperation.Update()
#    
#    booleanOperationMapper = vtk.vtkPolyDataMapper()
#    booleanOperationMapper.SetInputConnection(booleanOperation.GetOutputPort())
#    booleanOperationMapper.ScalarVisibilityOff()
#    
    booleanOperationActor = vtk.vtkActor()
#    booleanOperationActor.SetMapper(booleanOperationMapper)
#    renderer.AddViewProp(booleanOperationActor)
    return  booleanOperation, booleanOperationActor
def cutter_polydata_v1(center,normal,image_plane):
    plane = vtk.vtkPlane()
    plane.SetOrigin(center[0],center[1],center[2])
    plane.SetNormal(normal[0],normal[1],normal[2])
    cutter =vtk.vtkCutter()
    cutter.SetCutFunction(plane);
    cutter.SetInputConnection(image_plane.GetOutputPort());
    cutter.Update();
    cutterMapper = vtk.vtkPolyDataMapper();
    cutterMapper.SetInputConnection( cutter.GetOutputPort()); 
    planeActor =vtk.vtkActor();
    planeActor.GetProperty().SetColor(1.0,1,0);
    planeActor.GetProperty().SetLineWidth(2);
    planeActor.SetMapper(cutterMapper);
    return   planeActor,cutter

def cutter_polydata(center,normal,image_plane):
    plane = vtk.vtkPlane()
    plane.SetOrigin(center[0],center[1],center[2])
    plane.SetNormal(normal[0],normal[1],normal[2])
    cutter =vtk.vtkCutter()
    cutter.SetCutFunction(plane);
    cutter.SetInputConnection(image_plane.GetOutputPort());
    cutter.Update();
    cutterMapper = vtk.vtkPolyDataMapper();
    cutterMapper.SetInputConnection( cutter.GetOutputPort()); 
    planeActor =vtk.vtkActor();
    planeActor.GetProperty().SetColor(1.0,1,0);
    planeActor.GetProperty().SetLineWidth(2);
    planeActor.SetMapper(cutterMapper);
    return   planeActor

def intersectionofplanes(plane1,plane2):
    input1 = plane1.GetOutput() #sphereSource1.GetOutput()
    sphere1Tri = vtk.vtkTriangleFilter()
    sphere1Tri.SetInputData(input1)
    
    input2 = plane2.GetOutput()  #sphereSource2.GetOutput()
    sphere2Tri = vtk.vtkTriangleFilter()
    sphere2Tri.SetInputData(input2)

    
    booleanOperation = vtk.vtkIntersectionPolyDataFilter()

    booleanOperation.SetInputConnection(0, sphere1Tri.GetOutputPort())
    booleanOperation.SetInputConnection(1, sphere2Tri.GetOutputPort())

    booleanOperation.Update()
    
    booleanOperationMapper = vtk.vtkPolyDataMapper()
    booleanOperationMapper.SetInputConnection(booleanOperation.GetOutputPort())
    booleanOperationMapper.ScalarVisibilityOff()
    
    booleanOperationActor = vtk.vtkActor()
    booleanOperationActor.SetMapper(booleanOperationMapper)
#    renderer.AddViewProp(booleanOperationActor)
    return booleanOperation, booleanOperationActor
    
    

def create_plane(center,normal,N="Z"):
    aPlane =  vtkPlaneSource()
    aPlane.SetXResolution(512);
    aPlane.SetYResolution(512);
    aPlane.SetCenter(0,0,0)
    aPlane.SetNormal(normal[0],normal[1],normal[2])
    aPlane.Update();
    transform = vtk.vtkTransform()
    transform.Translate(center[0],center[1],center[2])
    if N=="Z":
        transform.Scale(512, 512, 1)
    if N=="X":
        transform.Scale(1, 512, 512)
    if N=="Y":
        transform.Scale(512, 1, 512)
    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
    actor.GetProperty().SetPointSize(10)
#    plane1 = vtk.vtkPlaneSource()# vtk.vtkSphereSource()
##    sphereSource2.SetCenter(256,256,15)
##    sphereSource2.SetNormal(1.0, 0, 0.0)
    print('transformPD.GetOutput().GetNumberOfPoints()')
    print(transformPD.GetOutput().GetNumberOfPoints())
##    vtkPlaneSource2.SetPoint1(transformPD.GetOutput().GetPoint(1)[0],transformPD.GetOutput().GetPoint(1)[1],transformPD.GetOutput().GetPoint(1)[2])
##    vtkPlaneSource2.SetPoint2(transformPD.GetOutput().GetPoint(2)[0],transformPD.GetOutput().GetPoint(2)[1],transformPD.GetOutput().GetPoint(2)[2])
#    vtkPlaneSource2.Update()
    
    #   actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return transformPD, actor #actor #aPlane, texture #__depthImageData
def draw_plane_3(center,normal,renderer,scale_factor,N="Z"):

    aPlane =  vtkPlaneSource()
    aPlane.SetXResolution(512);
    aPlane.SetYResolution(512);
    aPlane.SetCenter(0,0,0)
    aPlane.SetNormal(normal[0],normal[1],normal[2])
    aPlane.Update();
    transform = vtk.vtkTransform()
#    transform.SetMatrix(matrix)
    transform.Translate(center[0],center[1],center[2])
    transform.Scale(scale_factor, scale_factor, scale_factor)


##    if N=="Z":
##    transform.Scale(512, 512, 1)
##    if N=="X":
##        transform.Scale(1, 512, 512)
##    if N=="Y":
##        transform.Scale(512, 1, 512)
#    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
#    renderer.AddActor(actor)
    #   actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return transformPD, actor #aPlane, texture #__depthImageData

def draw_plane_2(point1,point2,point3,renderer,scale_factor,N="Z"):

    aPlane =  vtkPlaneSource()
#    aPlane.SetXResolution(512);
#    aPlane.SetYResolution(512);
    aPlane.SetPoint1(point1[0],point1[1],point1[2])
    aPlane.SetPoint2(point2[0],point2[1],point2[2])
    aPlane.SetOrigin(point3[0],point3[1],point3[2])
#    aPlane.SetCenter(0,0,0)
#    aPlane.SetNormal(normal[0],normal[1],normal[2])
    aPlane.Update();
    center=aPlane.GetCenter()
    transform = vtk.vtkTransform()
#    transform.SetMatrix(matrix)
    transform.Translate(center[0],center[1],center[2])
    transform.Scale(scale_factor, scale_factor, scale_factor)
    transform.Translate(-center[0],-center[1],-center[2])

##    if N=="Z":
##    transform.Scale(512, 512, 1)
##    if N=="X":
##        transform.Scale(1, 512, 512)
##    if N=="Y":
##        transform.Scale(512, 1, 512)
#    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
#    renderer.AddActor(actor)
    #   actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return transformPD, actor #aPlane, texture #__depthImageData

def draw_plane_1(point1,point2,point3,N="Z"):
#    center=(point1+point2+point3)/3
    A = point2-point1
    B= point3-point1
    C=np.cross(A,B)
    D=np.cross(C,A)
    A=A/np.linalg.norm(A)
    B=B/np.linalg.norm(B)
    C=C/np.linalg.norm(C)
    D=D/np.linalg.norm(D) # A,C and D
    print(A)
    print(B)
    print(C)
    center=point1
    matrix=vtk.vtkMatrix4x4()
    matrix.SetElement(0,0,A[0])
    matrix.SetElement(0,1,A[1])
    matrix.SetElement(0,2,A[2])
    matrix.SetElement(0,3,center[0])
    
    matrix.SetElement(1,0,D[0])
    matrix.SetElement(1,1,D[1])
    matrix.SetElement(1,2,D[2])
    matrix.SetElement(1,3,center[1])
    
    matrix.SetElement(2,0,C[0])
    matrix.SetElement(2,1,C[1])
    matrix.SetElement(2,2,C[2])
    matrix.SetElement(2,3,center[2])
    
    matrix.SetElement(3,0,0)
    matrix.SetElement(3,1,0)
    matrix.SetElement(3,2,0)
    matrix.SetElement(3,3,1)
    aPlane =  vtkPlaneSource()
    aPlane.SetXResolution(512);
    aPlane.SetYResolution(512);
    aPlane.SetPoint1(512,0,0)
    aPlane.SetPoint2(0,512,0)
    aPlane.SetCenter(0,0,0)
#    aPlane.SetCenter(0,0,0)
#    aPlane.SetNormal(normal[0],normal[1],normal[2])
    aPlane.Update();
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
#    transform.Translate(center[0],center[1],center[2])
#    if N=="Z":
#    transform.Scale(512, 512, 1)
#    if N=="X":
#        transform.Scale(1, 512, 512)
#    if N=="Y":
#        transform.Scale(512, 1, 512)
    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
    #   actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return actor #aPlane, texture #__depthImageData
    


def draw_plane(center,normal,N="Z"):
    aPlane =  vtkPlaneSource()
    aPlane.SetXResolution(512);
    aPlane.SetYResolution(512);
    aPlane.SetCenter(0,0,0)
    aPlane.SetNormal(normal[0],normal[1],normal[2])
    aPlane.Update();
    transform = vtk.vtkTransform()
    transform.Translate(center[0],center[1],center[2])
    if N=="Z":
        transform.Scale(512, 512, 1)
    if N=="X":
        transform.Scale(1, 512, 512)
    if N=="Y":
        transform.Scale(512, 1, 512)
    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
    #   actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return actor #aPlane, texture #__depthImageData
def draw_plane_with_size(center,normal,plane_size , N="Z"):
    aPlane =  vtkPlaneSource()
    aPlane.SetXResolution(plane_size);
    aPlane.SetYResolution(plane_size);
    aPlane.SetCenter(0,0,0)
    aPlane.SetNormal(normal[0],normal[1],normal[2])
    aPlane.Update();
    transform = vtk.vtkTransform()
    transform.Translate(center[0],center[1],center[2])
    if N=="Z":
        transform.Scale(plane_size, plane_size, 1)
    if N=="X":
        transform.Scale(1, plane_size, plane_size)
    if N=="Y":
        transform.Scale(plane_size, 1, plane_size)
    # Transform the polydata
    transformPD = vtk.vtkTransformPolyDataFilter()
    
    transformPD.SetInputConnection(aPlane.GetOutputPort())
    transformPD.SetTransform(transform)
    transformPD.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(transformPD.GetOutputPort()) 
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
    #   actor.SetTexture(texture);
    
    # Create a renderer, render window, and interactor
    
    return actor #aPlane, texture #__depthImageData


def make_image_mask(image_directory,train_gray_filename,train_bin_filename,train_or_test, stretch_thresh):


    train_gray=nib.load(os.path.join(image_directory,train_gray_filename))
    train_grayI=exposure.rescale_intensity(train_gray.get_fdata(),in_range=(stretch_thresh[0],stretch_thresh[1]))
    train_bin=nib.load(os.path.join(image_directory,train_bin_filename))
    count=0
    for i in range(0,train_gray.get_fdata().shape[2]):
        if np.sum(train_bin.get_fdata()[:,:,i]) >0:
            image=np.zeros([train_gray.get_fdata().shape[0],train_gray.get_fdata().shape[1],3])
            image[:,:,0]= normalizeimage0to1(train_grayI[:,:,i]) * 255
            image[:,:,1]= normalizeimage0to1(train_grayI[:,:,i]) * 255
            image[:,:,2]= normalizeimage0to1(train_grayI[:,:,i]) * 255
            
            mask=np.zeros([train_gray.get_fdata().shape[0],train_gray.get_fdata().shape[1],3])
            mask[:,:,0]= train_bin.get_fdata()[:,:,i] 
            mask[:,:,1]= train_bin.get_fdata()[:,:,i]
            mask[:,:,2]= train_bin.get_fdata()[:,:,i]
#                if train_or_test.lower() == 'train' : 
            command='mkdir -p ' + image_directory + 'IMAGEFORUNET/' + train_or_test.lower() + '/image/'
            subprocess.call(command,shell=True)
            command='mkdir -p ' + image_directory + 'IMAGEFORUNET/' + train_or_test.lower() +'/mask/'
            subprocess.call(command,shell=True)
            cv2.imwrite(os.path.join(image_directory + 'IMAGEFORUNET/' + train_or_test.lower() + '/image/',str(count)+'.tif'), image.astype(int))
            cv2.imwrite(os.path.join(image_directory + 'IMAGEFORUNET/' + train_or_test.lower() +'/mask/' ,str(count)+'.tif'), train_bin.get_fdata()[:,:,i].astype(int))
            count= count +1

def sequence_to_niigz(sourcedirectory, destinationdirectory,nii_file_gray,imsize):
    niifile=nib.load(nii_file_gray)
    filesinsource= glob.glob(sourcedirectory+ "/*")
    image=np.zeros([imsize[0],imsize[1],len(filesinsource)])
    filebasename=os.path.basename(filesinsource[0])
    filebasenamesplit=filebasename.split('.')
    filebasenamesplit1=filebasenamesplit[0].rsplit('_',1)
    for i in range(1,len(filesinsource)+1):
        thisfilename= filebasenamesplit1[0] + '_' + "{:02d}".format(i) + '.' +  filebasenamesplit[1]
        img=cv.imread(os.path.join(sourcedirectory,thisfilename)).astype(float)
        img[img<200]=0
        cv.transpose(img,img );
#        img=cv.flip(img , 0);
#        show_slice((img[:,:,0] + img[:,:,1] + img[:,:,2])/3)
        
#        show_slice(exposure.rescale_intensity(niifile.get_fdata()[:,:,i-1], in_range=(1000, 1200)))
#        show_slices(niifile.get_fdata())
#    slice_num=int(filebasenamesplit1[len(filebasenamesplit1)-1])
        
        image[:,:,i-1]= (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
        print(np.min((img[:,:,0] + img[:,:,1] + img[:,:,2])/3))
#      filebasenamesplit1[0]  
    nii_mask_file= nib.Nifti1Image(image.astype(int), niifile.affine, niifile.header)
    nii_mask_file.to_filename(os.path.join(destinationdirectory,filebasenamesplit1[0]  + '.nii.gz'))
    return "YY"

def plot_im(im,filesaveas="testplot.jpg", dpi=80):
    px,py,pz = im.shape # depending of your matplotlib.rc you may 
#                              have to use py,px instead
    #px,py = im[:,:,0].shape # if image has a (x,y,z) shape 
    size = (py/np.float(dpi), px/np.float(dpi)) # note the np.float()

    fig = plt.figure(figsize=size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    # Customize the axis
    # remove top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # turn off ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.imshow(im)
    fig.savefig(filesaveas, format='jpg', dpi=400)
    plt.close(fig)
    return "XX"
#    plt.show()
def latex_start_table2c(filename):
    print("latex_start_table2c")
    print(filename)
    file1 = open(filename,"a")
    file1.writelines("\\begin{center}\n")
    file1.writelines("\\begin{tabular}{ c c  }\n")
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
def latex_start_beamer(filename):
    file1 = open(filename,"w")
    file1.writelines("\\documentclass[hyperref={bookmarks=false},aspectratio=169]{beamer}\n")
#    file1.writelines("\\usepackage[utf8]{inputenc}\n")
#    file1.writelines("\\usepackage{tikz}\n")
    
#    % ---------------  Define theme and color scheme  -----------------
#    file1.writelines("\\usetheme[sidebarleft]{Caltech}  % 3 options: minimal, sidebarleft, sidebarright\n")
#    file1.writelines("\\documentclass{beamer}\n")
#    file1.writelines("\\begin{document}\n")
    return file1
def latex_begin_document(filename):
    file1 = open(filename,"a")
    file1.writelines("\\begin{document}\n")
    return file1                                                                                                                                                                                                     
    
def latex_start_1(filename):
    file1 = open(filename,"w")
    file1.writelines("\\documentclass{article}\n")
    file1.writelines("\\usepackage[margin=0.01in]{geometry}\n")
    file1.writelines("\\usepackage{graphicx}\n")   
    file1.writelines("\\usepackage[T1]{fontenc} \n")
    file1.writelines("\\begin{document}\n")
    return file1    
def latex_start(filename):
    file1 = open(filename,"w")
    file1.writelines("\\documentclass{article}\n")
    file1.writelines("\\usepackage[margin=0.01in]{geometry}\n")
    file1.writelines("\\usepackage{graphicx}\n")   
    file1.writelines("\\usepackage[T1]{fontenc} \n")
#    file1.writelines("\\begin{document}\n")
    return file1
def latex_end(filename):
    file1 = open(filename,"a")
    file1.writelines("\\end{document}\n")
    file1.close()
    return "X"
def frame_start(filename,frametitle="ATUL"):
    file1 = open(filename,"a")
    file1.writelines("\\begin{frame}\n")
    file1.writelines("\\frametitle{" + frametitle + "}\n")

    return file1
def frame_end(filename):
    file1 = open(filename,"a")
    file1.writelines("\\end{frame}\n")
    file1.close()
    return "X"
def latex_insert_line(filename,text="ATUL KUMAR"):
    file1 = open(filename,"a")
    file1.writelines(text)
    file1.writelines("\n")
    return file1
def latex_insert_text(filename,text="ATUL KUMAR",texttype="section"):
    file1 = open(filename,"a")
    if texttype=="section":
        file1.writelines("\\section{}\n")
        file1.writelines(text)
    if texttype=="item":
        file1.writelines("\\begin{itemize}\n")
        file1.writelines("\\item\n")
        file1.writelines(text)
        file1.writelines("\\end{itemize}\n")
    file1.writelines("\n")
    return file1

def latex_insert_image(filename,image_dir="/home/atul/Pictures/",image="lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\begin{figure}\n")
    file1.writelines("\\includegraphics[scale=" + str(imagescale) + "]{" + image_dir + image + "}\n")
    file1.writelines("\\caption{" + caption + "}\n")
    file1.writelines("\\end{figure}\n")
    return file1
def latex_insert_image1(filename,image_source="/home/atul/Pictures/lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\begin{figure}\n")
    file1.writelines("\\includegraphics[scale=" + str(imagescale) + "]{" + image_source + "}\n")
    file1.writelines("\\caption{" + caption + "}\n")
    file1.writelines("\\end{figure}\n")
    return file1
def latex_begin_image(filename):
    file1 = open(filename,"a")
    file1.writelines("\\begin{figure}\n")
    return file1
def latex_include_image1(filename,image_source="/home/atul/Pictures/lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\includegraphics[scale=" + str(imagescale) + "]{" + image_source + "}\n")
#    file1.writelines("\\caption{" + caption + "}\n")
    return file1
def latex_include_image(filename,image_dir="/home/atul/Pictures/",image="lion.jpg",caption="ATUL",imagescale=0.5):
    file1 = open(filename,"a")
    file1.writelines("\\includegraphics[scale=" + str(imagescale) + "]{" + image_dir + image + "}\n")
#    file1.writelines("\\caption{" + caption + "}\n")
    return file1
def latex_end_image(filename):
    file1 = open(filename,"a")
    file1.writelines("\\end{figure}\n")
    return file1
def latex_begin_columns(filename):
    file1 = open(filename,"a")
    file1.writelines("\\begin{columns}[c]\n")
    return file1
def latex_end_columns(filename):
    file1 = open(filename,"a")
    file1.writelines("\\end{columns}\n")
    return file1
def latex_file_build(filename,output_directory=""):
    if len(output_directory)>2:
        command  =   "pdflatex -aux-directory=" + output_directory + "   " +  "-output-directory=" + output_directory + "  " + filename 
    else:
        command = "pdflatex   " + filename
        
    subprocess.call(command,shell=True)
#    pro = subprocess.Popen(command, stdout=subprocess.PIPE, 
#                       shell=True, preexec_fn=os.setsid)
#    os.killpg(os.getpgid(pro.pid), signal.SIGTERM) 
    return "yy"
def subtract_binary(binary_imageBig,binary_imageSmall):
#    print("I am here")
    binary_image_non_zero_cord_t=np.nonzero(binary_imageSmall>0)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
#    print("binary_image_non_zero_cord")
#    print(binary_image_non_zero_cord)
    for each_point in binary_image_non_zero_cord:
        binary_imageBig[each_point[0],each_point[1],each_point[2]]=0
    return binary_imageBig

def get_bet_area(binary_imageBig,binary_imageSmall):
    print("I am here")
    binary_image_non_zero_cord_t=np.nonzero(binary_imageSmall==0)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    print("binary_image_non_zero_cord")
    print(binary_image_non_zero_cord)
    for each_point in binary_image_non_zero_cord:
        binary_imageBig[each_point[0],each_point[1],each_point[2]]=0
    return binary_imageBig

def get_mask_area(binary_imageBig,binary_imageSmall):
    binary_imageBigCopy=np.copy(binary_imageBig)
#    print("I am here")
    binary_image_non_zero_cord_t=np.nonzero(binary_imageSmall==0)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
#    print("binary_image_non_zero_cord")
#    print(binary_image_non_zero_cord)
    for each_point in binary_image_non_zero_cord:
        binary_imageBigCopy[each_point[0],each_point[1],each_point[2]]=0
    return binary_imageBigCopy


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
        
def calculate_volume_1(nii_loaded,mask_data_np):
#    mask_data=myfunctions.analyze_stack(mask_file)
#    img = nib.load(nii_file)
    resol= np.prod(np.array(nii_loaded.header["pixdim"][1:4]))
    mask_data_flatten= mask_data_np.flatten()
    num_pixel_gt_0=mask_data_flatten[np.where(mask_data_flatten>0)]
    print(num_pixel_gt_0)
    return (resol * num_pixel_gt_0.size)/1000

def calculate_volume(nii_file,mask_file):
    mask_data=myfunctions.analyze_stack(mask_file)
    img = nib.load(nii_file)
    resol= np.prod(np.array(img.header["pixdim"][1:4]))
    mask_data_flatten= mask_data.flatten()
    num_pixel_gt_0=mask_data_flatten[np.where(mask_data_flatten>0)]
    print(num_pixel_gt_0)
    return (resol * num_pixel_gt_0.size)/1000
def create_subj_name_datetime_helsinki(file_name):
    filename= os.path.basename(file_name)
    filename_splitted0=filename.split("-")
    filename_splitted=filename_splitted0[1].split("_")
    participantid= filename_splitted[0]# + "_" + filename_splitted[1]
    DateTime= datetime.strptime(filename_splitted[2]+ filename_splitted[3],'%m%d%Y%H%M')
    return filename_splitted0[0],participantid, DateTime
def create_subj_name_datetime(file_name):
    filename= os.path.basename(file_name)
    filename_splitted=filename.split("_")
    participantid=  filename_splitted[1]
    DateTime= datetime.strptime(filename_splitted[2]+ filename_splitted[3],'%m%d%Y%H%M')
    return filename_splitted[0],participantid, DateTime
def dir_filelist(thisfile):
    thisdir=os.path.dirname(thisfile)
    filename,file_extension=os.path.splitext(thisfile)
    dir_pattern=thisdir +'/*'+ file_extension
    return glob.glob(dir_pattern)

def write_csv_for_hem(mask_files,gray_files,csv_filename):
    dict_for_csv=[]
    files_not_suitable=[]
    for thismaskfile in mask_files:
        thismaskfile1=os.path.basename(thismaskfile)
        thismaskfilename_split=thismaskfile1.split("_")
        thismaskfilename = thismaskfilename_split[0]+thismaskfilename_split[1]+thismaskfilename_split[2]+thismaskfilename_split[3]
        for thisgrayfile in gray_files:
            thisgrayfile1 = os.path.basename(thisgrayfile)
            thisgrayfilename_split=thisgrayfile1.split("_")
            thisgrayfilename = thisgrayfilename_split[0]+thisgrayfilename_split[1]+thisgrayfilename_split[2]+thisgrayfilename_split[3]
    #        print(thismaskfilename)
    #        print(thisgrayfilename)
            if thismaskfilename== thisgrayfilename:
                try:
                    print(thisgrayfile)
                    print(thismaskfile)
                    print("yes")
                    this_Site, this_participantid,this_datetime=create_subj_name_datetime(thisgrayfile)
                    this_volume=9999     
                    this_volume=calculate_volume(thisgrayfile,thismaskfile)
                    this_Site_Number= this_Site + "_" + this_participantid
                    this_dict={"Site_Number": this_Site_Number,"participantid":this_participantid,"DateTimeHem":this_datetime,"Hem_Vol":this_volume}
                    dict_for_csv.append(this_dict)
                except:
                    print("Could not work on file:" , os.path.basename(thismaskfile))
                    files_not_suitable.append(os.path.basename(thismaskfile))
                    pass
               #this_dict={"participantid":this_participantid,"DateTime":this_datetime,"hem_vol":this_volume}
    csvfile_with_vol=csv_filename+'.csv'
    csv_columns=['Site_Number','participantid','DateTimeHem','Hem_Vol']
    write_csv(csvfile_with_vol,csv_columns,dict_for_csv)
    notsuitablefiles = csv_filename + 'not_suitable.csv' 
    with open(notsuitablefiles,'w') as f:
        writer = csv.writer(f)
        writer.writerows([files_not_suitable])


def analyze_stack(pathname):
    img_data = nib.AnalyzeImage.from_filename(pathname)
    img_data = img_data.get_fdata()
    arr = np.asarray(img_data)
    arr.astype(float)
    return  arr

def nii_stack(pathname):
    img = nib.load(pathname)
    img_data = img.get_fdata()
    arr = np.asarray(img_data)
    arr.astype(float)
    return arr

def np3darray_show_slice(images):

    for i  in range(0,images.shape[2]):
        fig, ax = plt.subplots(num="CT_demo")
        ax.imshow(images[:,:,i], cmap="gray")
        ax.axis('off')
        plt.show()
def np3darray_histogram(x,nbins=50):
    hist, bins = np.histogram(x.flatten(), bins=nbins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(num="CT_demo")
    ax.bar(center, hist, align='center', width=width)
    plt.show()
        
def show_slices(slices):
 #""" Function to display row of image slices """
#    fig, axes = plt.subplots(1, len(slices))
#    for i, slice in enumerate(slices):
#        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    for  i in range(slices.shape[2]):
        fig, axes = plt.subplots(1, 1)
        axes.imshow(slices[:,:,i].T, cmap="gray", origin="lower")
def show_slice(slices):
 #""" Function to display row of image slices """
    fig, axes = plt.subplots(1, 1)
    axes.imshow(slices, cmap="gray", origin="lower")

def show_slice_colored(slices):
 #""" Function to display row of image slices """
    fig, axes = plt.subplots(1, 1)
    axes.imshow(slices,origin="lower")   
def show_slice_withpoint(slices,points):
 #""" Function to display row of image slices """
    fig, axes = plt.subplots(1, 1)
    axes.imshow(slices, cmap="gray", origin="lower")
    if len(points.shape)==1:
        axes.scatter(points[0],points[1]) #, c='r', s=40)
    else:
        axes.scatter(points[:,0],points[:,1]) #, c='r', s=40)
def show_slice_withaline(slices,points):
 #""" Function to display row of image slices """
    fig, axes = plt.subplots(1, 1)
    axes.imshow(slices, cmap="gray", origin="lower")
    x1, y1 = [points[0][0], points[1][0]], [points[0][1], points[1][1]]
    axes.plot(x1, y1, marker = 'o') #, c='r', s=40)
    
    
def show_slice_with_multiple_lines(slices,lines):
     #""" Function to display row of image slices """
    fig, axes = plt.subplots(1, 1)
    axes.imshow(slices, cmap="gray", origin="lower")
    x1, y1 = [points[0][0], points[1][0]], [points[0][1], points[1][1]]
    axes.plot(x1, y1, marker = 'o') #, c='r', s=40)
    
def show_slice_colored_withpoint(slices,points):
 #""" Function to display row of image slices """
    fig, axes = plt.subplots(1, 1)
    axes.imshow(slices, origin="lower")
    if len(points.shape)==1:
        axes.scatter(points[0],points[1]) #, c='r', s=40)
    else:
        axes.scatter(points[:,0],points[:,1]) #, c='r', s=40)
    
# Method to process the red band of the image
kernel = np.ones((5,5),np.uint8)
def normalizeRed(intensity):
    iI = intensity

    minI = 86

    maxI = 230

    minO = 0

    maxO = 255

    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)

    return iO


# Method to process the green band of the image

def normalizeGreen(intensity):
    iI = intensity

    minI = 90

    maxI = 225

    minO = 0

    maxO = 255

    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)

    return iO


# Method to process the blue band of the image

def normalizeBlue(intensity):
    iI = intensity

    minI = 100

    maxI = 210

    minO = 0

    maxO = 255

    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)

    return iO
def rgb2gray(filename):
    image = cv.imread(filename)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.cvtColor(gray_image, cv.COLOR_GRAY2RGB)
    filename=filename.replace('.png', '.jpg')
    print(filename)
    cv.imwrite(filename, gray_image)

def niitotiff(imagestack,pathname,imagename):
    img = nib.load(imagestack)
    img_data = img.get_fdata()
    arr = np.asarray(img_data)
    sizearr=arr.shape
    arr[arr > 80] = 0
    arr[arr < 20] = 0
    #subp.call(["mkdir", imagename], shell=False)
    for i in range (0,sizearr[2]):
        # img = PIL.Image.fromarray(arr[:,:,i])
        # img = img.resize((256, 256),  PIL.Image.ANTIALIAS)
        # img = img.convert('L')
        # filenumber=["%03d" % i]
        filenumber=[i]
        filename=  str(filenumber[0]) +'.tif'
        filename= os.path.join(pathname, filename)
        gray_image= arr[:,:,i] #cv.merge(arr[:,:,i],arr[:,:,i],arr[:,:,i])
        img2 = np.zeros(shape=(gray_image.shape[0],gray_image.shape[1],3))
        # img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image

        # gray_image = cv.cvtColor(arr[:,:,i], cv.COLOR_GRAY2RGB)
        cv.imwrite(filename, img2)
    #subp.call(["mv",'*.tiff' , imagename], shell=False)


def analyzetotiff(imagestack,pathname,imagename):
    img_data = nib.AnalyzeImage.from_filename(imagestack)
    img_data = img_data.get_fdata()
    arr = np.asarray(img_data)
    arr.astype(float)
    sizearr=arr.shape
   # subp.call(["mkdir", imagename], shell=False)
    for i in range (0,sizearr[2]):
        # img = arr[:,:,i]
        # img = img[:, :,0]
        # img = PIL.Image.fromarray(img)
        # img = img.resize((256, 256), PIL.Image.ANTIALIAS)
        # img = img.convert('L')
        # filenumber=["%03d" % i]
        filenumber=[i]
        filename= str(filenumber[0]) +'.tif'
        filename = os.path.join(pathname, filename)
        gray_image= arr[:,:,i][0] #cv.merge(arr[:,:,i],arr[:,:,i],arr[:,:,i])
        img2 = np.zeros(shape=(gray_image.shape[0], gray_image.shape[1], 3))
        # img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image

        # gray_image = cv.cvtColor(arr[:,:,i], cv.COLOR_GRAY2RGB)
        cv.imwrite(filename, img2)
        # img.save(filename)
    #subp.call(["mv",'*.tiff' , imagename], shell=False)


def npstacktotiff(npstacktotiff,pathname,imagename):
    arr=npstacktotiff
    sizearr=arr.shape
   # subp.call(["mkdir", imagename], shell=False)
    for i in range (0,sizearr[2]):
        # img = arr[:,:,i]
        # # img = img[:, :]
        # img = PIL.Image.fromarray(img)
        # img = img.resize((256, 256), PIL.Image.ANTIALIAS)
        # img = img.convert('L')
        # filenumber=["%03d" % i]
        fillgap=arr[:,:,i]

#img_data=exposure.rescale_intensity(img_data, in_range=(0, 200))
###############################
def create_histo(data, min_val, max_val, num_bins=None):

    if num_bins is None:
        num_bins = int(np.max(data) - np.min(data))    
    hist, bins = np.histogram(data, num_bins, range=(min_val, max_val))
    fig3, axes3 = plt.subplots(1, 1,figsize=(8, 8))
  
    hist = np.array(hist) 
    axes3.hist(hist, bins = bins) 
    axes3.hist(data.T, cmap="gray", origin="lower")
    
    return fig3 
def contraststretch(data,maxx,minn): 
    epi_img_data_max=maxx #np.max(data)
    epi_img_data_min=minn #np.min(data)
    thisimage=data
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min) * 255
    return thisimage

def normalizeimage(data): 
    epi_img_data_max=np.max(data)
    epi_img_data_min=np.min(data)
    thisimage=data
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min)  * 255
    return thisimage

def normalizeimage0to1(data): 
    epi_img_data_max=np.max(data)
    epi_img_data_min=np.min(data)
    thisimage=np.copy(data)
    thisimage=(thisimage-epi_img_data_min)/(epi_img_data_max-epi_img_data_min) 
    return thisimage

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def resample_single_image(image,width, height):
    dim = (width, height)
# resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    
    return resized

def show_allimages_in_one(image_np,z_position='first'):
    if z_position=='last':
        image_num=image_np.shape[2]
        print("Number of slices")
        print(image_num)
        fig, axs = plt.subplots(int(image_num/5) + 1, 5)
        row=0
        col=0
        for i in range(image_num):
#            print([row,col])
            axs[row,col].imshow(image_np[:,:,i], cmap="gray", origin="lower")
            if col%4==0 and col!=0:
                row=row+1
                col=0
            else:
                col=col+1
    if z_position=='first':
        image_num=image_np.shape[0]
        print("Number of slices")
        print(image_num)
        fig, axs = plt.subplots(int(image_num/5) + 1, 5)
        row=0
        col=0
        for i in range(image_num):
#            print([row,col])
            axs[row,col].imshow(image_np[i,:,:], cmap="gray", origin="lower")
            if col%4==0 and col!=0:
                row=row+1
                col=0
            else:
                col=col+1
                
def angle_bet_two_vector(v1,v2):
#    angle = np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2)) 
    angle =(np.arctan2(v2[1], v2[0]) -  np.arctan2(v1[1], v1[0]))* 180 / np.pi
    return angle
def angle_bet_two_vectorRad(v1,v2):
#    angle = np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2)) 
    angle =np.arctan2(v2[1], v2[0]) -  np.arctan2(v1[1], v1[0])
    return angle
#def angle_bet_two_vector(v1, v2):
#    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
#    cosang = np.dot(v1, v2)
#    sinang = la.norm(np.cross(v1, v2))
#    return np.arctan2(sinang, cosang) * 180 / np.pi
def rotate_image(img,center1=[0,0],angle=0):
    (h,w)= (img.shape[0],img.shape[1])
    scale = 1.0
    # calculate the center of the image
#    center = (w / 2, h / 2) 
    center = (center1[0], center1[1]) 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotatedimg = cv2.warpAffine(img, M, (h, w), flags= cv2.INTER_NEAREST) 
    return rotatedimg

def declare_rendwindow():
    colors = vtk.vtkNamedColors()
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    return renderer,renderWindow,renderWindowInteractor,colors
def add_vol_to_renderer(filename,renderer,intensity_range,color_factor=1):
    img = sitk.ReadImage( filename ) # SimpleITK object
    data =  sitk.GetArrayFromImage( img ).astype('float') # numpy array
    images_data=exposure.rescale_intensity(data,in_range=(intensity_range[0],intensity_range[1]))
#    data *= 255 / data.max()
    
    from scipy.stats.mstats import mquantiles
    q = mquantiles(data.flatten(),[0.7,0.98])
    q[0]=max(q[0],1)
    q[1] = max(q[1],1)
    tf=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],1,1,1,0.5],[data.max(),1,1,1,1]]
#    color_factor=1
    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(),color_factor=color_factor)
    for a in actor_list:
    # assign actor to the renderer
        renderer.AddActor(a)
        
def add_numpyvol_to_renderer(data,renderer,spacing):
#    img = sitk.ReadImage( filename ) # SimpleITK object
#    data =  sitk.GetArrayFromImage( img ).astype('float') # numpy array
#    images_data=exposure.rescale_intensity(data,in_range=(intensity_range[0],intensity_range[1]))
#    data *= 255 / data.max()
    
    from scipy.stats.mstats import mquantiles
    q = mquantiles(data.flatten(),[0.7,0.98])
    q[0]=max(q[0],1)
    q[1] = max(q[1],1)
    tf=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],1,1,1,0.5],[data.max(),1,1,1,1]]
    color_factor=1
    actor_list = volumeRender(data, tf=tf, spacing=spacing,color_factor=1)
    for a in actor_list:
    # assign actor to the renderer
        renderer.AddActor(a)
def start_rendering(renderer,renderWindow,renderWindowInteractor,colors):
    renderer.ResetCamera()
    renderer.SetBackground(colors.GetColor3d("SlateGray"))
    
    # Render and interact

    renderWindow.Render()
    renderWindowInteractor.Start()
    
def flip_3D_image():
    return "XX"


def resample_with_nilearn(inputfile,outputfile,target_shape):
# inputfile='/home/atul/Downloads/CLINICALMASTER/scct_stripped.nii'
# outputfile='/home/atul/Downloads/CLINICALMASTER/scct_strippedREG.nii.gz'
##orig_nii = nb.load("../input/ixi-example/ixi002-guys-0828-t1.nii/IXI002-Guys-0828-T1.nii")
#template_image_for_resampling="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_660/files_Head_Spiral_3.0_J40s_2_20180503102601_2.nii"
##template_image_for_resampling_nil= image.load_img(template_image_for_resampling)
    orig_nii = nb.load(inputfile)
#np.round(orig_nii.affine)
#downsampled_nii = resample_img(orig_nii, target_affine=np.eye(3)*2., interpolation='nearest')
#print(downsampled_nii.affine)
#target_shape = np.array((512,512,50))
    target_affine=np.eye(4)#*0.5 #orig_nii.affine
#    target_affine[0][0]=1.
#    target_affine[1][1]=1.
#    target_affine[2][2]=1.
#    target_affine[3][3]=1.
    target_affine[0][3]= -256 #-175. #
    target_affine[1][3]=-256 #-75.  # 
    target_affine[2][3]=-40   # -50
    resampled_image = resample_img(orig_nii,target_affine=target_affine,target_shape=target_shape, interpolation='linear')
    nib.save(resampled_image, outputfile)
def resample_with_nilearnForMNI(inputfile,outputfile,target_shape):
# inputfile='/home/atul/Downloads/CLINICALMASTER/scct_stripped.nii'
# outputfile='/home/atul/Downloads/CLINICALMASTER/scct_strippedREG.nii.gz'
##orig_nii = nb.load("../input/ixi-example/ixi002-guys-0828-t1.nii/IXI002-Guys-0828-T1.nii")
#template_image_for_resampling="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_660/files_Head_Spiral_3.0_J40s_2_20180503102601_2.nii"
##template_image_for_resampling_nil= image.load_img(template_image_for_resampling)
    orig_nii = nb.load(inputfile)
#np.round(orig_nii.affine)
#downsampled_nii = resample_img(orig_nii, target_affine=np.eye(3)*2., interpolation='nearest')
#print(downsampled_nii.affine)
#target_shape = np.array((512,512,50))
    target_affine=np.eye(4)#*0.5 #orig_nii.affine
#    target_affine[0][0]=1.
#    target_affine[1][1]=1.
#    target_affine[2][2]=1.
#    target_affine[3][3]=1.
    target_affine[0][3]= -256 #-175. #
    target_affine[1][3]=-256 #-75.  # 
    target_affine[2][3]=-40   # -50
    resampled_image = resample_img(orig_nii,target_affine=target_affine,target_shape=target_shape, interpolation='linear')
    nib.save(resampled_image, outputfile)

def bet_withoutskull_regrow(directory_name,filename):
    file = os.path.join(directory_name,filename)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=700 )
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
    #for i in range(sitk.GetArrayFromImage(bet_itk_array).shape[0]):
    # show_slice(bet_image[i,:,:])
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=-1000 #np.min(sitk.GetArrayFromImage(img_gray))
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    
    return eht_image_itk,seg_explicit_thresholds # seg_explicit_thresholds,img_T1

def write_BET_image(NECT_filename):
    file_gray=NECT_filename
    img_T1=sitk.ReadImage(file_gray)
    img_T1_np=sitk.GetArrayFromImage(img_T1)
    img_T1_np_itk=sitk.GetImageFromArray(img_T1_np)
    img_T1_np_itk.CopyInformation(img_T1)
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(img_T1, sigma=.5)
    speed_img = sitk.BoundedReciprocal(feature_img)
    fm_filter = sitk.FastMarchingBaseImageFilter()
    seed = (256,256,int(sitk.GetArrayFromImage(feature_img).shape[0]/2))
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(1400)
    fm_img = fm_filter.Execute(speed_img)
    fm_img_numpy=sitk.GetArrayFromImage(fm_img)
    fm_img_numpy[fm_img_numpy==np.max(fm_img_numpy)]=0
    fm_img_numpy[fm_img_numpy>0]=255
    fmimg_itk=sitk.GetImageFromArray(fm_img_numpy)
    fmimg_itk.SetDirection(fm_img.GetDirection())
    fmimg_itk.SetSpacing(fm_img.GetSpacing())
    fmimg_itk.SetOrigin(fm_img.GetOrigin())
    file_gray_bet=NECT_filename[:-7]+'_BET_TEST.nii.gz'
    sitk.WriteImage(fmimg_itk,file_gray_bet,True)
    
    

def bet_withskull_regrow(directory_name,filename):
    file = os.path.join(directory_name,filename)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
    for i in range(sitk.GetArrayFromImage(bet_image).shape[0]):
        show_slice(bet_image[i,:,:])
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=-1000 #np.min(sitk.GetArrayFromImage(img_gray))
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    
    return eht_image_itk,seg_explicit_thresholds # seg_explicit_thresholds,img_T1
def bet_withskull_regrowv1(file):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=int(np.min(sitk.GetArrayFromImage(img_T1))/2), upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
#    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=int(200.0))
    
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
    for i in range(bet_image.shape[0]):
     show_slice(bet_image[i,:,:])
     print(np.min(sitk.GetArrayFromImage(img_T1)))
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=np.min(img_gray_numpy) #sitk.GetArrayFromImage(
    img_gray_numpy[img_gray_numpy<100]=np.min(img_gray_numpy)   
    img_gray_numpy[img_gray_numpy>np.min(img_gray_numpy)]=np.max(img_gray_numpy)
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    for i in range(img_gray_numpy.shape[0]):
     show_slice(img_gray_numpy[i,:,:])    
    return eht_image_itk,seg_explicit_thresholds # seg_explicit_thresholds,img_T1

def seg_calcified_Struct(file):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=int(np.min(sitk.GetArrayFromImage(img_T1))/2), upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
#    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=int(200.0))
    
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
    for i in range(bet_image.shape[0]):
     show_slice(bet_image[i,:,:])
     print(np.min(sitk.GetArrayFromImage(img_T1)))
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=np.min(img_gray_numpy) #sitk.GetArrayFromImage(
    img_gray_numpy[img_gray_numpy<100]=np.min(img_gray_numpy)   
    img_gray_numpy[img_gray_numpy>np.min(img_gray_numpy)]=np.max(img_gray_numpy)
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    for i in range(img_gray_numpy.shape[0]):
     show_slice(img_gray_numpy[i,:,:]) 
    plt.close('all')
    return eht_image_itk,seg_explicit_thresholds # seg_explicit_thresholds,img_T1

def seg_brain_with_skull(file): #    filetosave=file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    if np.min(sitk.GetArrayFromImage(img_T1))< 0:
        seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=1, upper=int(np.max(sitk.GetArrayFromImage(img_T1))))

    else:
        seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=1000, upper=int(np.max(sitk.GetArrayFromImage(img_T1))))

#        seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=int(np.min(sitk.GetArrayFromImage(img_T1))/4), upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
#    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=int(200.0))
    
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
#    for i in range(bet_image.shape[0]):
#     show_slice(bet_image[i,:,:])
#     print(np.min(sitk.GetArrayFromImage(img_T1)))
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=np.min(img_gray_numpy) #sitk.GetArrayFromImage(
#    img_gray_numpy[img_gray_numpy<100]=np.min(img_gray_numpy)   
#    img_gray_numpy[img_gray_numpy>np.min(img_gray_numpy)]=np.max(img_gray_numpy)
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    filetosave=file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
    sitk.WriteImage(eht_image_itk,filetosave)
#    for i in range(img_gray_numpy.shape[0]):
#     show_slice(img_gray_numpy[i,:,:]) 
#    plt.close('all')
    return eht_image_itk,seg_explicit_thresholds # seg_explicit_thresholds,img_T1

def seg_skullbone(file): #filenametosave=file.split(".nii")[0]+"_SKULL.nii.gz"
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=0, upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
#    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=int(200.0)) #int(np.min(sitk.GetArrayFromImage(img_T1))/2)
    
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
#    for i in range(bet_image.shape[0]):
#     show_slice(bet_image[i,:,:])
#     print(np.min(sitk.GetArrayFromImage(img_T1)))
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=np.min(img_gray_numpy) #sitk.GetArrayFromImage(
    img_gray_numpy[img_gray_numpy<200]=np.min(img_gray_numpy)   
    img_gray_numpy[img_gray_numpy>np.min(img_gray_numpy)]=np.max(img_gray_numpy)
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    vectorRadius=(11,11,11)
    kernel=sitk.sitkBall
    seg_implicit_thresholds_clean = sitk.BinaryMorphologicalClosing(sitk.Cast(eht_image_itk, sitk.sitkUInt8) , 
                                                                vectorRadius,
                                                                kernel)
    
    filenametosave=file.split(".nii")[0]+"_SKULL.nii.gz"
    sitk.WriteImage(eht_image_itk,filenametosave)
#    for i in range(img_gray_numpy.shape[0]):
#     show_slice(img_gray_numpy[i,:,:]) 
#    plt.close('all')
    return eht_image_itk#,seg_explicit_thresholds # seg_explicit_thresholds,img_T1

def bet_withskull_regrowv1_1(file):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=0, upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
#    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))], lower=-120, upper=int(200.0))
    
    bet_image=sitk.GetArrayFromImage(seg_explicit_thresholds)
    bet_image[bet_image>0]=255
    for i in range(bet_image.shape[0]):
     show_slice(bet_image[i,:,:])
    img_gray_numpy=sitk.GetArrayFromImage(img_T1)
    img_gray_numpy[bet_image==0]=-1000 #np.min(sitk.GetArrayFromImage(img_gray))
    eht_image_itk=sitk.GetImageFromArray(img_gray_numpy)
    eht_image_itk.CopyInformation(img_T1)
    
    return eht_image_itk,seg_explicit_thresholds # seg_explicit_thresholds,img_T1



def bet_with_levelset(directory_name,filename):
    file = os.path.join(directory_name,filename)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    
    seed = (256,256,25)
    
    seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(img_T1)
    seg[seed] = 1
    seg = sitk.BinaryDilate(seg, 3)
#    stats = sitk.LabelStatisticsImageFilter()
#    stats.Execute(img_T1, seg)
    
#    factor = 3.5
    lower_threshold = - int(np.max(sitk.GetArrayFromImage(img_T1))/2)  # int(np.min(sitk.GetArrayFromImage(img_T1)))# stats.GetMean(1)-factor*stats.GetSigma(1)
    upper_threshold =int(np.max(sitk.GetArrayFromImage(img_T1))) # stats.GetMean(1)+factor*stats.GetSigma(1)
    init_ls = sitk.SignedMaurerDistanceMap(seg, insideIsPositive=True, useImageSpacing=True)
    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lower_threshold)
    lsFilter.SetUpperThreshold(upper_threshold)
    lsFilter.SetMaximumRMSError(5) #0.02)
    lsFilter.SetNumberOfIterations(1000)
    lsFilter.SetCurvatureScaling(.5)
    lsFilter.SetPropagationScaling(1)
    lsFilter.ReverseExpansionDirectionOn()
    ls = lsFilter.Execute(init_ls, sitk.Cast(img_T1, sitk.sitkFloat32))
    print(lsFilter)
#    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(256,256,25)], lower=0, upper=int(np.max(sitk.GetArrayFromImage(img_T1))))
    return ls , img_T1 #"seg_explicit_thresholds,img_T1"
def write_mat_to_nii_gz(gray_mask_image_itk,img_gray_ctp,outputfilename):
    mask_image_itk1 = sitk.Resample(gray_mask_image_itk,img_gray_ctp.GetSize() )
    mask_image_itk1 = sitk.Flip( mask_image_itk1, [True,False,False] )
    mask_image_itk1.SetSpacing(img_gray_ctp.GetSpacing())
    mask_image_itk1.SetOrigin(img_gray_ctp.GetOrigin())
    #mask_image_itk1.SetDirection(img_gray_ctp.GetDirection())
    #mask_image_itk1.SetMetaData(img_gray_ctp.GetMetaData())
    #mask_image_itk1.CopyInformation(img_gray_ctp)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputfilename) #)
    writer.Execute(mask_image_itk1)
def resample3D(file_nii,ref_nii,filenametosave):
    grid_image=sitk.ReadImage(file_nii)
#    file_nii_data=nib.load(file_nii)
    ref_nii_data=nib.load(ref_nii).get_fdata()
    new_size = [ref_nii_data.shape[0], ref_nii_data.shape[1],ref_nii_data.shape[2]]
    reference_image = sitk.Image(new_size, grid_image.GetPixelIDValue())
    reference_image.SetOrigin(grid_image.GetOrigin())
    reference_image.SetDirection(grid_image.GetDirection())
    reference_image.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, grid_image.GetSize(), grid_image.GetSpacing())])
    # Resample without any smoothing.
    resampled_image=sitk.Resample(grid_image, reference_image)
    sitk.WriteImage(resampled_image,filenametosave)
    
def register_twoimages(fixed,moving,type_of_transform= 'Rigid'):
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
#    fixed = ants.resample_image(fixed, (64,64), 1, 0)
#    moving = ants.resample_image(moving, (64,64), 1, 0)
    mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = type_of_transform,reg_iterations=(2000,1000,1000),aff_sampling=512,syn_sampling=512)
    # Get the mask and transform it: 
    return mytx,fixed
def register_twoimages_test(fixed,moving,type_of_transform= 'Rigid'):
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
    mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # Get the mask and transform it: 

    return mytx,fixed

def register_twoimages_N_Save(fixed_file,moving_file,type_of_transform= 'Rigid'):
    
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)
    mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # Get the mask and transform it: 
    mywarpedimage=apply_transformation_1(fixed, moving,mytx)
    moving_file_split=moving_file.split(".nii")
    reg_output_filename1=moving_file_split[0]+ "REGTO" + os.path.basename(fixed_file).split(".nii")[0] +".nii.gz"
    ants.image_write(mywarpedimage,reg_output_filename1) 

    return mytx,fixed_file,reg_output_filename1

def resample_register_twoimages_N_Save(fixed_file,moving_file,type_of_transform= 'Rigid'):
    
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)
    mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # Get the mask and transform it: 
    mywarpedimage=apply_transformation_1(fixed, moving,mytx)
    moving_file_split=moving_file.split(".nii")
    reg_output_filename1=moving_file_split[0]+ "REGTO" + os.path.basename(fixed_file).split(".nii")[0] +".nii.gz"
    ants.image_write(mywarpedimage,reg_output_filename1) 

    return mytx,fixed_file,reg_output_filename1

def register_twoimages_test_v1(fixed,moving,type_of_transform= 'ElasticSyN'):
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
    mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # Get the mask and transform it: 

    return mytx,fixed

def register_rigid_twoimages_savefile_py():
    static_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Helsinki2000_1024_3_10242014_0950_Head_4.0_e1RS_WRTWUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2BETgray.nii.gz" #sys.argv[1]
    moving_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Helsinki2000_1024_3_10242014_0950_Head_4.0_e1RS_WRTWUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii.gz" #sys.argv[2]
    type_of_transform="Rigid" #sys.argv[3]
    static_file_basename_split=os.path.basename(static_file).split("_")
    filenam_suffix="REG"+ static_file_basename_split[0] + static_file_basename_split[1] + static_file_basename_split[2] + static_file_basename_split[3] + ".nii.gz"
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
#    nameebeforenii=moving.split(".nii")
    static_file_bet=static_file.split(".nii")[0]+ "BETgray.nii.gz" 
    moving_file_bet=moving_file.split(".nii")[0]+ "BETgray.nii.gz"
    
    print(static_file_bet)
    print(static_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )

    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # Get the mask and transform it: 
    mywarpedimage=apply_transformation_1(fixed_bet, moving,mytx)
    moving_file_split=moving_file.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    
#    hf = h5py.File(moving_file_split[0]+'TX.h5', 'w')
#    hf.create_dataset('mytx', data=mytx)
#    hf.close()
#    ants.write_transform(mytx, moving_file_split[0]+'TX.mat')
#    print(mytx)
#    command="cp " + mytx['fwdtransforms'][0] + "  " + moving_file_split[0]+'TX0.mat'
#    subprocess.call(command,shell=True)
#    command="cp " + mytx['fwdtransforms'][1] + "  " + moving_file_split[0]+'TX1.mat'
#    subprocess.call(command,shell=True)
#    mytx['fwdtransforms'][0]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX0.mat'
#    mytx['fwdtransforms'][1]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX1.mat'
#    np.save(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2' +'.npy', {mytx['fwdtransforms']}) 
#    hf = h5py.File(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2.h5', 'w')
#    hf.create_dataset('mytx', data=mytx['fwdtransforms'])
#    hf.close()
    return mytx,fixed_bet

def register_rigid_twoimages_savefile_sh():
    print(" I AM HERE")

    static_file=sys.argv[2]
    moving_file=sys.argv[1] 
    type_of_transform=sys.argv[3]
    static_file_basename_split=os.path.basename(static_file).split("_")
    filenam_suffix="REG"+ static_file_basename_split[0] + static_file_basename_split[1] + static_file_basename_split[2] + static_file_basename_split[3] + ".nii.gz"
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
#    nameebeforenii=moving.split(".nii")
    static_file_bet=static_file.split(".nii")[0]+ "BETFgray.nii.gz" 
    moving_file_bet=moving_file.split(".nii")[0]+ "BETFgray.nii.gz"
    
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )

    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    print("Registration was done")
    # Get the mask and transform it: 
    mywarpedimage=apply_transformation_1(fixed_bet, moving,mytx)
    moving_file_split=moving_file.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    print("Image  1 is saved at:")
    print(reg_output_filename1)
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    print("Image 2 is saved at:")
    print(reg_output_filename1)
#    hf = h5py.File(moving_file_split[0]+'TX.h5', 'w')
#    hf.create_dataset('mytx', data=mytx)
#    hf.close()
#    ants.write_transform(mytx, moving_file_split[0]+'TX.mat')
#    print(mytx)
#    command="cp " + mytx['fwdtransforms'][0] + "  " + moving_file_split[0]+'TX0.mat'
#    subprocess.call(command,shell=True)
#    command="cp " + mytx['fwdtransforms'][1] + "  " + moving_file_split[0]+'TX1.mat'
#    subprocess.call(command,shell=True)
#    mytx['fwdtransforms'][0]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX0.mat'
#    mytx['fwdtransforms'][1]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX1.mat'
#    np.save(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2' +'.npy', {mytx['fwdtransforms']}) 
#    hf = h5py.File(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2.h5', 'w')
#    hf.create_dataset('mytx', data=mytx['fwdtransforms'])
#    hf.close()
    return mytx,fixed_bet

def register_twoimages_test_savefile(filenam_suffix="CTnolesion.nii.gz", type_of_transform= 'Rigid'):
    static_file=sys.argv[1]
    moving_file=sys.argv[2]
    static_file_basename_split=os.path.basename(static_file).split("_")
    filenam_suffix="REG"+ static_file_basename_split[0] + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
#    nameebeforenii=moving.split(".nii")
    static_file_bet=static_file[:-7]+ "BETgray.nii.gz" 
    moving_file_bet=moving_file.split(".nii")[0]+ "BETgray.nii.gz"
    
    print(static_file_bet)
    print(static_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )

    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # Get the mask and transform it: 
    mywarpedimage=apply_transformation_1(fixed_bet, moving,mytx)
    moving_file_split=moving_file.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
#    hf = h5py.File(moving_file_split[0]+'TX.h5', 'w')
#    hf.create_dataset('mytx', data=mytx)
#    hf.close()
#    ants.write_transform(mytx, moving_file_split[0]+'TX.mat')
#    print(mytx)
#    command="cp " + mytx['fwdtransforms'][0] + "  " + moving_file_split[0]+'TX0.mat'
#    subprocess.call(command,shell=True)
#    command="cp " + mytx['fwdtransforms'][1] + "  " + moving_file_split[0]+'TX1.mat'
#    subprocess.call(command,shell=True)
#    mytx['fwdtransforms'][0]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX0.mat'
#    mytx['fwdtransforms'][1]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX1.mat'
#    np.save(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2' +'.npy', {mytx['fwdtransforms']}) 
#    hf = h5py.File(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2.h5', 'w')
#    hf.create_dataset('mytx', data=mytx['fwdtransforms'])
#    hf.close()
    return mytx,fixed_bet

def register_twoimages_transformoneslice(filenam_suffix="REGTO664.nii.gz", type_of_transform= 'Rigid'):
    static_file=sys.argv[1]
    moving_file=sys.argv[2]
    moving_file_singleslice=sys.argv[3]
    static_file_basename_split=os.path.basename(static_file).split("_")
    filenam_suffix="REG"+ static_file_basename_split[0] + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
#    nameebeforenii=moving.split(".nii")
    static_file_bet=static_file[:-7]+ "BET.nii.gz" 
    moving_file_bet=moving_file[:-7]+ "BETgray.nii.gz"
    
    print(static_file_bet)
    print(static_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
#    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving_file_singleslice_data=ants.image_read(moving_file_singleslice)
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
#    print(mytx['fwdtransforms'][1] )
    # Get the mask and transform it: 
    mywarpedimage=apply_transformation_1(fixed_bet, moving_file_singleslice_data,mytx)
    moving_file_split=moving_file_singleslice.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    return mytx,fixed_bet

def apply_transformation(fixed, moving1,mytx):
#    fixed=fixed.resample_image_to_target(moving1)
    mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving1,interpolator='linear', #'nearestNeighbor',
                                               transformlist=mytx['fwdtransforms'] )
    return mywarpedimage 
def apply_transformation_1(fixed, moving1,mytx):
#    fixed=fixed.resample_image_to_target(moving1)
    mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving1,interpolator='linear',
                                               transformlist=mytx['fwdtransforms'] )
    return mywarpedimage 
def apply_transformation_2(fixed, moving1,mytx):
#    fixed=fixed.resample_image_to_target(moving1)
    mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving1,interpolator='genericLabel',
                                               transformlist=mytx['fwdtransforms'] )
    return mywarpedimage 
    
def writefilewithsitk(filename,sitk_image):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(sitk_image)    

def BET_levelset_itk():
    
    inputFileName ="fixed_image.nii.gz" # "WUSTL_667_05102018_0218_part320gray.jpg" #"BrainProtonDensitySlice6.png" #arguments[1]
    outputFileName ="fixed_image_BET.jpg'" #nii.gz" # "outputBrainProtonDensitySlice6.png" #arguments[2]
    image=itk.imread(inputFileName)
    image_numpy=itk.GetArrayFromImage(image)
    #arguments=(0,0,0,81, 114, 5.0, 1.0, -0.5 ,3.0, 2.0,1000)
    arguments=(0,0,0,256, 256, 100.0, 1.0, -0.3, 2.0, 10.0,1000)
    seedPosX = int(arguments[3])
    seedPosY = int(arguments[4])
    
    initialDistance = float(arguments[5])
    sigma = float(arguments[6])*5
    alpha = float(arguments[7])
    beta = float(arguments[8])
    propagationScaling = float(arguments[9])
    numberOfIterations = int(arguments[10])
    seedValue = -initialDistance
    
    Dimension = 2# 2
    
    InputPixelType = itk.F
    OutputPixelType = itk.UC
    
    InputImageType = itk.Image[InputPixelType, Dimension]
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    
    ReaderType = itk.ImageFileReader[InputImageType]
    WriterType = itk.ImageFileWriter[OutputImageType]
    
#    reader = ReaderType.New()
#    reader.SetFileName(inputFileName)
    for i in range(20,21) : #image_numpy.shape[0]):
        SmoothingFilterType = itk.CurvatureAnisotropicDiffusionImageFilter[
            InputImageType, InputImageType]
        smoothing = SmoothingFilterType.New()
        smoothing.SetTimeStep(0.125)
        smoothing.SetNumberOfIterations(5)
        smoothing.SetConductanceParameter(9.0)
        smoothing.SetInput(itk.GetImageFromArray(exposure.rescale_intensity(  image_numpy[i,:,:], in_range=(0, 200))))
                
#                image_numpy[i,:,:])) # reader.GetOutput())
        
        GradientFilterType = itk.GradientMagnitudeRecursiveGaussianImageFilter[
            InputImageType, InputImageType]
        gradientMagnitude = GradientFilterType.New()
        gradientMagnitude.SetSigma(sigma)
        gradientMagnitude.SetInput(smoothing.GetOutput())
        
        SigmoidFilterType = itk.SigmoidImageFilter[InputImageType, InputImageType]
        sigmoid = SigmoidFilterType.New()
        sigmoid.SetOutputMinimum(0.0)
        sigmoid.SetOutputMaximum(1.0)
        sigmoid.SetAlpha(alpha)
        sigmoid.SetBeta(beta)
        sigmoid.SetInput(gradientMagnitude.GetOutput())
        
        FastMarchingFilterType = itk.FastMarchingImageFilter[
            InputImageType, InputImageType]
        fastMarching = FastMarchingFilterType.New()
        
        GeoActiveContourFilterType = itk.GeodesicActiveContourLevelSetImageFilter[
            InputImageType, InputImageType, InputPixelType]
        geodesicActiveContour = GeoActiveContourFilterType.New()
        geodesicActiveContour.SetPropagationScaling(propagationScaling)
        geodesicActiveContour.SetCurvatureScaling(1.0)
        geodesicActiveContour.SetAdvectionScaling(1.0)
        geodesicActiveContour.SetMaximumRMSError(0.02)
        geodesicActiveContour.SetNumberOfIterations(numberOfIterations)
        geodesicActiveContour.SetInput(fastMarching.GetOutput())
        geodesicActiveContour.SetFeatureImage(sigmoid.GetOutput())
        
        ThresholdingFilterType = itk.BinaryThresholdImageFilter[
            InputImageType, OutputImageType]
        thresholder = ThresholdingFilterType.New()
        thresholder.SetLowerThreshold(-1000.0)
        thresholder.SetUpperThreshold(0.0)
        thresholder.SetOutsideValue(itk.NumericTraits[OutputPixelType].min())
        thresholder.SetInsideValue(itk.NumericTraits[OutputPixelType].max())
        thresholder.SetInput(geodesicActiveContour.GetOutput())
        
        seedPosition = itk.Index[Dimension]()
        seedPosition[0] = seedPosX
        seedPosition[1] = seedPosY
#        seedPosition[2] = 11
        
        node = itk.LevelSetNode[InputPixelType, Dimension]()
        node.SetValue(seedValue)
        node.SetIndex(seedPosition)
        
        seeds = itk.VectorContainer[
            itk.UI, itk.LevelSetNode[InputPixelType, Dimension]].New()
        seeds.Initialize()
        seeds.InsertElement(0, node)
        
        fastMarching.SetTrialPoints(seeds)
        fastMarching.SetSpeedConstant(1.0)
        
        CastFilterType = itk.RescaleIntensityImageFilter[
            InputImageType, OutputImageType]
        
        caster1 = CastFilterType.New()
        caster2 = CastFilterType.New()
        caster3 = CastFilterType.New()
        caster4 = CastFilterType.New()
        
        writer1 = WriterType.New()
        writer2 = WriterType.New()
        writer3 = WriterType.New()
        writer4 = WriterType.New()
        
        caster1.SetInput(smoothing.GetOutput())
        writer1.SetInput(caster1.GetOutput())
        writer1.SetFileName("GeodesicActiveContourImageFilterOutput1.png")
        caster1.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
        caster1.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
        writer1.Update()
        
        caster2.SetInput(gradientMagnitude.GetOutput())
        writer2.SetInput(caster2.GetOutput())
        writer2.SetFileName("GeodesicActiveContourImageFilterOutput2.png")
        caster2.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
        caster2.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
        writer2.Update()
        
        caster3.SetInput(sigmoid.GetOutput())
        writer3.SetInput(caster3.GetOutput())
        writer3.SetFileName("GeodesicActiveContourImageFilterOutput3.png")
        caster3.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
        caster3.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
        writer3.Update()
        
        caster4.SetInput(fastMarching.GetOutput())
        writer4.SetInput(caster4.GetOutput())
        writer4.SetFileName("GeodesicActiveContourImageFilterOutput4.png")
        caster4.SetOutputMinimum(itk.NumericTraits[OutputPixelType].min())
        caster4.SetOutputMaximum(itk.NumericTraits[OutputPixelType].max())
        writer4.Update()
        fastMarching.SetOutputSize( itk.GetImageFromArray(image_numpy[i,:,:]).GetBufferedRegion().GetSize())
#            reader.GetOutput().GetBufferedRegion().GetSize())
        
        writer = WriterType.New()
        writer.SetFileName(outputFileName)
        writer.SetInput(thresholder.GetOutput())
        writer.Update()
        
        #print(
        #    "Max. no. iterations: " +
        #    str(geodesicActiveContour.GetNumberOfIterations()) + "\n")
        #print(
        #    "Max. RMS error: " +
        #    str(geodesicActiveContour.GetMaximumRMSError()) + "\n")
        #print(
        #    "No. elpased iterations: " +
        #    str(geodesicActiveContour.GetElapsedIterations()) + "\n")
        #print("RMS change: " + str(geodesicActiveContour.GetRMSChange()) + "\n")
        #
        #writer4.Update()
        #
        #InternalWriterType = itk.ImageFileWriter[InputImageType]
        #
        #mapWriter = InternalWriterType.New()
        #mapWriter.SetInput(fastMarching.GetOutput())
        #mapWriter.SetFileName("GeodesicActiveContourImageFilterOutput4.mha")
        #mapWriter.Update()
        #
        #speedWriter = InternalWriterType.New()
        #speedWriter.SetInput(sigmoid.GetOutput())
        #speedWriter.SetFileName("GeodesicActiveContourImageFilterOutput3.mha")
        #speedWriter.Update()
        #
        #gradientWriter = InternalWriterType.New()
        #gradientWriter.SetInput(gradientMagnitude.GetOutput())
        #gradientWriter.SetFileName("GeodesicActiveContourImageFilterOutput2.mha")
        #gradientWriter.Update()
            
            
def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])


def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


def rotate_around_point_lowperf(point, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "low performance" version since it's recalculating
    the same values more than once [cos(radians), sin(radians), x-ox, y-oy).
    It's more readable than the next function, though.
    """
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy


def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
     
        
def dicomseries_to_nifti(image_directory,outputfilename):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_directory) # sys.argv[1] )
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    writer = sitk.ImageFileWriter()
    writer.SetImageIO('NiftiImageIO')
    writer.SetFileName(outputfilename)
    writer.Execute(image)
    
def extractMax(input): 
  
     # get a list of all numbers separated by  
     # lower case characters  
     # \d+ is a regular expression which means 
     # one or more digit 
     # output will be like ['100','564','365'] 
     numbers = re.findall('\d+',input) 
  
     # now we need to convert each number into integer 
     # int(string) converts string into integer 
     # we will map int() function onto all elements  
     # of numbers list 
     numbers = map(int,numbers) 
  
#     print (max(numbers) )
     return numbers
 
    
def normalize_itk_0to1(a_itk):
    a=sitk.GetArrayFromImage(a_itk)
    b = (a - np.min(a))/np.ptp(a)
    b_itk=sitk.GetImageFromArray(b)
    b_itk.CopyInformation(a_itk)
    b_itk=sitk.Cast(b_itk,sitk.sitkInt8)
    return b_itk

def perpendicular_bisector(pointAnp,pointBnp):
    p1=pointAnp
    p2=pointBnp
    mid_point=(p1+p2)/2
    slope=np.divide((p2[1]-p1[1]),(p2[0]-p1[0]))
    perp_slope= -1/slope
    # when x=0
    y0= perp_slope*(-mid_point[0])+mid_point[1]
    # when x=512 
    y512= perp_slope*(512-mid_point[0])+mid_point[1]
    return slope, perp_slope, (0,y0), (512,y512)
def whichsideofline(line_pointA,line_pointB,point_todecide):
    
    return (point_todecide[0]-line_pointA[0])*(line_pointB[1]-line_pointA[1])  -  (point_todecide[1]-line_pointA[1])*(line_pointB[0]-line_pointA[0])
    
    
    
def perpendicular_throughapoint(pointAnp,pointBnp,pointtopass):
    p1=pointAnp
    p2=pointBnp
    mid_point=pointtopass #(p1+p2)/2
    slope=np.divide((p2[1]-p1[1]),(p2[0]-p1[0]))
    perp_slope= -1/slope
    # when x=0
    y0= perp_slope*(-mid_point[0])+mid_point[1]
    # when x=512 
    y512= perp_slope*(512-mid_point[0])+mid_point[1]
    point1=(0,y0)
    point2=(512,y512) 
    if slope==0:
        point1=(p1[1],p1[0])
        point2=(p2[1],p2[0])
        
        
    return slope, perp_slope,point1,point2 # (0,y0), (512,y512)   
    
def getPerpCoord(aX, aY, bX, bY, length):
    vX = bX-aX
    vY = bY-aY
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)   
def lower_slice_midline_refinement_v1(image_gray_bet,image_gray,pointA,pointB,anglethreshold=10):
        # between 0 to 255
#image_gray_bet=np.uint8(img_gray_bet.get_fdata()[:,:,min_slope_diff_topN_bottom[3]]*255)
#image_gray=np.uint8(img[:,:,min_slope_diff_topN_bottom[3]]*255)
#anglethreshold=10
# 
    imgray=image_gray_bet
    imgray1=image_gray 
    ret, thresh = cv2.threshold(imgray, 127, 255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_3_layer= np.zeros([imgray1.shape[0],imgray.shape[1],3])
    slice_3_layer[:,:,0]=imgray1
    slice_3_layer[:,:,1]=imgray1
    slice_3_layer[:,:,2]=imgray1
    slice_3_layer=np.uint8(slice_3_layer)
#    cv2.drawContours(slice_3_layer, contours, -1, (0,255,0), 3)
#    pointA=(int(256.0),int(0.0))
#    pointB=(int(256.0),int(512.0))
    slice_3_layer=cv2.line(slice_3_layer, ( int(pointA[0]),int(pointA[1])),(int(pointB[0]),int(pointB[1])), (255,255,0), 2)
    
    # mid point of the line:
    pointAnp=np.array(pointA)
    pointBnp=np.array(pointB)
    mean_point_contour=[]
    contourNitslength=[]
    for each_contour in contours:
#        print(len(each_contour))
        each_contourNitslenght=[each_contour,len(each_contour)]
        contourNitslength.append(each_contourNitslenght)
        mean_point_contour.append(np.mean(each_contour,axis=0)[0])
    contourNitslength= np.array(contourNitslength)
    largest_contour=contourNitslength[np.argmax(contourNitslength[:,1]),0]
    mean_point_contournp_mean=np.mean(np.array(largest_contour),axis=0)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_throughapoint(pointAnp,pointBnp,mean_point_contournp_mean[0])
    ## separate the points into upper and lower half:
    upperhalf=[]
    lowerhalf=[]
    for each_point in largest_contour:
        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
        if xx >0:
            upperhalf.append(each_point[0])
        if xx <0:
            lowerhalf.append(each_point[0])
            
    
#    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
#    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
#    X=binary_image_non_zero_cord[:,1].reshape(-1,1)
#    gabor_image_pixels_cord=np.zeros([len(X),2])
#    #    print(len(binary_image_non_zero_cord))
#    if len(binary_image_non_zero_cord)>300:
#        X=binary_image_non_zero_cord[:,1].reshape(-1,1)
#        y=binary_image_non_zero_cord[:,0].reshape(-1,1)
#        gabor_image_pixels_cord[:,0]=np.transpose(X)
#        gabor_image_pixels_cord[:,1]=np.transpose(y)
##    print(largest_contour)
#    for each_point in gabor_image_pixels_cord:
#        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point)
#        if xx > 0:
#            upperhalf.append(each_point)
#        if xx < 0:
#            lowerhalf.append(each_point)
##    for each_point in largest_contour:
##        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
##        if xx >0:
##            upperhalf.append(each_point[0])
##        if xx <0:
##            lowerhalf.append(each_point[0])
#        
#    upperhalf=np.array(upperhalf)
#    lowerhalf=np.array(lowerhalf)
    upperhalf_20deg=[]
    lowerhalf_20deg=[]
#    for each_point in upperhalf:
#        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
#    for each_point in lowerhalf:
#        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,0),1)
    for each_point in upperhalf:
    #    print(each_point[0])
        v1= np.array(pointAnp) - mean_point_contournp_mean[0]
        v2= each_point - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1,v2))
        if angle1>270:
            print(angle1)
            angle1=360-angle1
        if angle1 <anglethreshold :
            upperhalf_20deg.append(each_point)
            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    upperhalf_20deg=np.array(upperhalf_20deg)
    for each_point in lowerhalf:
    #    print(each_point[0])
        v1_1= np.array(pointBnp) - mean_point_contournp_mean[0]
        v2= each_point - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1_1,v2))
        print(angle1)
        if angle1>270:
            print(angle1)
            angle1=360-angle1
        if angle1 <anglethreshold :
            lowerhalf_20deg.append(each_point)
            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    
    lowerhalf_20deg=np.array(lowerhalf_20deg)
    upperhalf_20deg_dist=np.sqrt(np.sum(np.square(upperhalf_20deg-mean_point_contournp_mean[0]),axis=1))
    lowerhalf_20deg_dist=np.sqrt(np.sum(np.square(lowerhalf_20deg-mean_point_contournp_mean[0]),axis=1))
    upperhalf_20deg_dist_min_id= np.argmin(upperhalf_20deg_dist)
    lowerhalf_20deg_dist_min_id= np.argmin(lowerhalf_20deg_dist)
    midline_upper_point= upperhalf_20deg[upperhalf_20deg_dist_min_id]
    midline_lower_point= lowerhalf_20deg[lowerhalf_20deg_dist_min_id]
    
    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,255,0), 2)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_bisector(pointAnp,pointBnp)
    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,0,0), 2)
    
    slice_3_layer=cv2.line(slice_3_layer,( int(midline_upper_point[0]),int(midline_upper_point[1])),(int(midline_lower_point[0]),int(midline_lower_point[1])) , (255,0,0), 5)
    
    
    
#    cv2.imshow('Contours', slice_3_layer) 
##    cv2.imshow('Grayscale', imgray1) 
##    cv2.waitKey(0)
#    cv2.waitKey(0) 
#    cv2.destroyAllWindows() 
    PointA=( int(midline_upper_point[0]),int(midline_upper_point[1]))
    PointB=(int(midline_lower_point[0]),int(midline_lower_point[1]))

    return PointA,PointB
def lower_slice_midline_refinement_v1_2(image_gray_bet,image_gray,pointA,pointB,anglethreshold=10,gray_image_filepath="",RESULT_DIR="",filter_type="",slicenumber=1):
    
    DATA_DIRECTORY= os.path.dirname(gray_image_filepath)
    DATA_DIRECTORY_BASENAME=os.path.basename(DATA_DIRECTORY)
    print('gray_image_filepath')
    print(gray_image_filepath)
    file_base_name= os.path.basename(gray_image_filepath) # filesindir[file])
    imgray=image_gray_bet
    imgray1=image_gray 
    ret, thresh = cv2.threshold(imgray, 127, 255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_3_layer= np.zeros([imgray1.shape[0],imgray.shape[1],3])
    slice_3_layer[:,:,0]=imgray1
    slice_3_layer[:,:,1]=imgray1
    slice_3_layer[:,:,2]=imgray1
    slice_3_layer=np.uint8(slice_3_layer)
#    cv2.drawContours(slice_3_layer, contours, -1, (0,255,0), 3)
#    pointA=(int(256.0),int(0.0))
#    pointB=(int(256.0),int(512.0))
    slice_3_layer=cv2.line(slice_3_layer, ( int(pointA[0]),int(pointA[1])),(int(pointB[0]),int(pointB[1])), (255,255,0), 2)
    
    # mid point of the line:
    pointAnp=np.array(pointA)
    pointBnp=np.array(pointB)
    mean_point_contour=[]
    contourNitslength=[]
    for each_contour in contours:
#        print(len(each_contour))
        each_contourNitslenght=[each_contour,len(each_contour)]
        contourNitslength.append(each_contourNitslenght)
        mean_point_contour.append(np.mean(each_contour,axis=0)[0])
    contourNitslength= np.array(contourNitslength)
    largest_contour=contourNitslength[np.argmax(contourNitslength[:,1]),0]
    mean_point_contournp_mean=np.mean(np.array(largest_contour),axis=0)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_throughapoint(pointAnp,pointBnp,mean_point_contournp_mean[0])
    ## separate the points into upper and lower half:
#    for each_point in largest_contour:
#        cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
        
    upperhalf=[]
    lowerhalf=[]
    for each_point in largest_contour:
        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
        if xx >0:
            upperhalf.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,0),1)
        if xx <0:
            lowerhalf.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
            

    upperhalf_20deg=[]
    lowerhalf_20deg=[]
    upperhalf_20degBP=[]
    lowerhalf_20degBP=[]
    upperhalf_20degPx=[] 
    lowerhalf_20degPx=[] 
#    show_slice_withaline(imgray1,np.array([pointAnp,pointBnp]))
#    show_slice_withaline(imgray1,np.array([per_bisec_A,per_bisec_B]))
    
    for each_point in upperhalf:
        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    for each_point in lowerhalf:
        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,0),1)
    for each_point in largest_contour:
    #    print(each_point[0])
        v1= np.array(pointBnp) - mean_point_contournp_mean[0]
        v2= each_point[0] - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1,v2))
        print(angle1)
        if angle1<=anglethreshold:
            line1=np.array([per_bisec_A,per_bisec_B])
            line2=np.array([pointA,pointB])
            P= project_point_online(each_point[0],line1)
            Px= project_point_online(each_point[0],line2)
            upperhalf_20deg.append(P)
            upperhalf_20degPx.append(Px)
            upperhalf_20degBP.append(each_point[0])
            cv2.circle(slice_3_layer,(int(P[0]),int(P[1])),2,(255,0,255),1)
            cv2.circle(slice_3_layer,(int(Px[0]),int(Px[1])),2,(255,0,255),1)
            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
        if angle1>90:
            angle1=np.abs(180-angle1)
            if  angle1<=anglethreshold:
                line1=np.array([per_bisec_A,per_bisec_B])
                line2=np.array([pointA,pointB])
                P= project_point_online(each_point[0],line1)
                Px= project_point_online(each_point[0],line2)
                lowerhalf_20deg.append(P)
                lowerhalf_20degPx.append(Px)
                lowerhalf_20degBP.append(each_point[0])
                cv2.circle(slice_3_layer,(int(P[0]),int(P[1])),2,(255,0,0),1)
                cv2.circle(slice_3_layer,(int(Px[0]),int(Px[1])),2,(255,0,0),1)
                cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,0),1)




    upperhalf_20deg=np.array(upperhalf_20deg)
    upperhalf_20degPx=np.array(upperhalf_20degPx)
    upperhalf_20degBP=np.array(upperhalf_20degBP)
    
    lowerhalf_20deg=np.array(lowerhalf_20deg)
    lowerhalf_20degPx=np.array(lowerhalf_20degPx)
    lowerhalf_20degBP=np.array(lowerhalf_20degBP)
    
#    upperhalf_id=0
#    lowerhalf_id=0
#    minimum_distance=999999999
#    current_uh_id=0
#    distances=[]
#    for each_point_upper in upperhalf_20deg: 
##        current_lh_id=0
#        current_distance=[]
#        for each_point_lower in lowerhalf_20degBP:
#            distance= np.sqrt(np.sum((each_point_upper - each_point_lower) * (each_point_upper - each_point_lower)))
#            current_distance.append(distance)
#        distances.append(np.min(np.array(current_distance)))
#    upperhalf_20deg_dist_min_id= np.argmin(np.array(distances))
#    current_distance=[]
#    for each_point_lower in lowerhalf_20degBP:
#        distance= np.sqrt(np.sum((upperhalf_20deg[upperhalf_20deg_dist_min_id] - each_point_lower) * (upperhalf_20deg[upperhalf_20deg_dist_min_id] - each_point_lower)))
#        current_distance.append(distance)
#    lowerhalf_20deg_dist_min_id= np.argmin(np.array(current_distance))
#    
##            if distance < minimum_distance:
##                upperhalf_id=current_uh_id
##                lowerhalf_id=current_lh_id
##            current_lh_id= current_lh_id+1
##        current_uh_id=current_uh_id+1
            
        
#    
#    
#    
    upperhalf_20deg_dist=np.sqrt(np.sum(np.square(upperhalf_20degBP-mean_point_contournp_mean[0]),axis=1))
    lowerhalf_20deg_dist=np.sqrt(np.sum(np.square(lowerhalf_20degBP-mean_point_contournp_mean[0]),axis=1))
#    upperhalf_20deg_distx=np.sqrt(np.sum(np.square(upperhalf_20degPx-mean_point_contournp_mean[0]),axis=1))
#    lowerhalf_20deg_distx=np.sqrt(np.sum(np.square(lowerhalf_20degPx-mean_point_contournp_mean[0]),axis=1))
#    upperhalf_20deg_dist=upperhalf_20deg_distx #upperhalf_20deg_dist*
#    lowerhalf_20deg_dist=lowerhalf_20deg_distx #lowerhalf_20deg_dist*
    
    upperhalf_20deg_dist_min_id= np.argmin(upperhalf_20deg_dist)
    lowerhalf_20deg_dist_min_id= np.argmin(lowerhalf_20deg_dist)
    midline_upper_point= upperhalf_20degBP[upperhalf_20deg_dist_min_id]
    midline_lower_point= lowerhalf_20degBP[lowerhalf_20deg_dist_min_id]
    
    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,255,0), 2)
#    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_bisector(pointAnp,pointBnp)
#    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,0,0), 2)
    
    slice_3_layer=cv2.line(slice_3_layer,( int(midline_upper_point[0]),int(midline_upper_point[1])),(int(midline_lower_point[0]),int(midline_lower_point[1])) , (255,0,0), 5)
    
    
    
#    cv2.imshow('Contours', slice_3_layer) 
##    cv2.imshow('Grayscale', imgray1) 
##    cv2.waitKey(0)
#    cv2.waitKey(0) 
#    cv2.destroyAllWindows() 
    PointA=( int(midline_upper_point[0]),int(midline_upper_point[1]))
    PointB=(int(midline_lower_point[0]),int(midline_lower_point[1]))
    PointAnp=np.array(PointA)
    PointBnp=np.array(PointB)
    if PointAnp.size==0 or PointBnp.size==0:
        PointA=( int(pointA[0]),int(pointA[1]))
        PointB=(int(pointB[0]),int(pointB[1]))
    i=slicenumber
    cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_gray" + str(i)+'.jpg'),slice_3_layer)
#    cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_Gabor" + str(i)+'.jpg'),filtered_img)
#    cv2.imwrite(os.path.join(RESULT_DIR,DATA_DIRECTORY_BASENAME+filter_type,file_base_name[:-7]+ "_slice_" + str(i)+'.jpg'),binary_image_copy)
    return PointA,PointB


def lower_slice_midline_refinement_v1_1(image_gray_bet,image_gray,pointA,pointB,anglethreshold=10,RESULT_DIR="",filter_type="",slicenumber=1):
        # between 0 to 255
#image_gray_bet=np.uint8(img_gray_bet.get_fdata()[:,:,min_slope_diff_topN_bottom[3]]*255)
#image_gray=np.uint8(img[:,:,min_slope_diff_topN_bottom[3]]*255)
#anglethreshold=10
#   
    SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/RESULTS_FOR_FIGURE"
    imgray=image_gray_bet
    imgray1=image_gray 
    ret, thresh = cv2.threshold(imgray, 127, 255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_3_layer= np.zeros([imgray1.shape[0],imgray.shape[1],3])
    slice_3_layer[:,:,0]=imgray1
    slice_3_layer[:,:,1]=imgray1
    slice_3_layer[:,:,2]=imgray1
    slice_3_layer=np.uint8(slice_3_layer)
#    cv2.drawContours(slice_3_layer, contours, -1, (0,255,0), 3)
#    pointA=(int(256.0),int(0.0))
#    pointB=(int(256.0),int(512.0))
    
    # mid point of the line:
    pointAnp=np.array(pointA)
    pointBnp=np.array(pointB)
    mean_point_contour=[]
    contourNitslength=[]
    for each_contour in contours:
#        print(len(each_contour))
        each_contourNitslenght=[each_contour,len(each_contour)]
        contourNitslength.append(each_contourNitslenght)
        mean_point_contour.append(np.mean(each_contour,axis=0)[0])
    contourNitslength= np.array(contourNitslength)
    largest_contour=contourNitslength[np.argmax(contourNitslength[:,1]),0]
    mean_point_contournp_mean=np.mean(np.array(largest_contour),axis=0)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_throughapoint(pointAnp,pointBnp,mean_point_contournp_mean[0])
    ## separate the points into upper and lower half:
    for each_point in largest_contour:
        cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
    

    cv2.circle(slice_3_layer,(int(mean_point_contournp_mean[0][0]),int(mean_point_contournp_mean[0][1])),2,(0,0,255),2)
    cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Centroid.png"),slice_3_layer)
    slice_3_layer=cv2.line(slice_3_layer, ( int(pointA[0]),int(pointA[1])),(int(pointB[0]),int(pointB[1])), (255,255,0), 2)
    cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse.png"),slice_3_layer)
    
    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,255,0), 2)
#    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_bisector(pointAnp,pointBnp)
#    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,0,0), 2)
    cv2.circle(slice_3_layer,(int(mean_point_contournp_mean[0][0]),int(mean_point_contournp_mean[0][1])),2,(0,0,255),2)
    cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse3.png"),slice_3_layer)
    upperhalf=[]
    lowerhalf=[]
    for each_point in largest_contour:
        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
        if xx >0:
            upperhalf.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,0),1)
        if xx <0:
            lowerhalf.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
            

    upperhalf_20deg=[]
    lowerhalf_20deg=[]
    upperhalf_20degBP=[]
    lowerhalf_20degBP=[]
    upperhalf_20degPx=[] 
    lowerhalf_20degPx=[] 
#    show_slice_withaline(imgray1,np.array([pointAnp,pointBnp]))
#    show_slice_withaline(imgray1,np.array([per_bisec_A,per_bisec_B]))
    
    for each_point in upperhalf:
        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    for each_point in lowerhalf:
        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,0),1)
        

    cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse1.png"),slice_3_layer)

    for each_point in largest_contour:
    #    print(each_point[0])
        v1= np.array(pointBnp) - mean_point_contournp_mean[0]
        v2= each_point[0] - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1,v2))
        print(angle1)
        if angle1<=anglethreshold:
            line1=np.array([per_bisec_A,per_bisec_B])
            line2=np.array([pointA,pointB])
            P= project_point_online(each_point[0],line1)
            Px= project_point_online(each_point[0],line2)
            upperhalf_20deg.append(P)
            upperhalf_20degPx.append(Px)
            upperhalf_20degBP.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(P[0]),int(P[1])),2,(255,0,255),1)
#            cv2.circle(slice_3_layer,(int(Px[0]),int(Px[1])),2,(255,0,255),1)
            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
        if angle1>90:
            angle1=np.abs(180-angle1)
            if  angle1<=anglethreshold:
                line1=np.array([per_bisec_A,per_bisec_B])
                line2=np.array([pointA,pointB])
                P= project_point_online(each_point[0],line1)
                Px= project_point_online(each_point[0],line2)
                lowerhalf_20deg.append(P)
                lowerhalf_20degPx.append(Px)
                lowerhalf_20degBP.append(each_point[0])
#                cv2.circle(slice_3_layer,(int(P[0]),int(P[1])),2,(255,0,0),1)
                cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse2_1.png"),slice_3_layer)
#                cv2.circle(slice_3_layer,(int(Px[0]),int(Px[1])),2,(255,0,0),1)
                cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse2_2.png"),slice_3_layer)
                cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,0),1)
                cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse2_3.png"),slice_3_layer)


    cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse2.png"),slice_3_layer)

    upperhalf_20deg=np.array(upperhalf_20deg)
    upperhalf_20degPx=np.array(upperhalf_20degPx)
    upperhalf_20degBP=np.array(upperhalf_20degBP)
    
    lowerhalf_20deg=np.array(lowerhalf_20deg)
    lowerhalf_20degPx=np.array(lowerhalf_20degPx)
    lowerhalf_20degBP=np.array(lowerhalf_20degBP)
    
#    upperhalf_id=0
#    lowerhalf_id=0
#    minimum_distance=999999999
#    current_uh_id=0
#    distances=[]
#    for each_point_upper in upperhalf_20deg: 
##        current_lh_id=0
#        current_distance=[]
#        for each_point_lower in lowerhalf_20degBP:
#            distance= np.sqrt(np.sum((each_point_upper - each_point_lower) * (each_point_upper - each_point_lower)))
#            current_distance.append(distance)
#        distances.append(np.min(np.array(current_distance)))
#    upperhalf_20deg_dist_min_id= np.argmin(np.array(distances))
#    current_distance=[]
#    for each_point_lower in lowerhalf_20degBP:
#        distance= np.sqrt(np.sum((upperhalf_20deg[upperhalf_20deg_dist_min_id] - each_point_lower) * (upperhalf_20deg[upperhalf_20deg_dist_min_id] - each_point_lower)))
#        current_distance.append(distance)
#    lowerhalf_20deg_dist_min_id= np.argmin(np.array(current_distance))
#    
##            if distance < minimum_distance:
##                upperhalf_id=current_uh_id
##                lowerhalf_id=current_lh_id
##            current_lh_id= current_lh_id+1
##        current_uh_id=current_uh_id+1
            
        
#    
#    
#    
    upperhalf_20deg_dist=np.sqrt(np.sum(np.square(upperhalf_20degBP-mean_point_contournp_mean[0]),axis=1))
    lowerhalf_20deg_dist=np.sqrt(np.sum(np.square(lowerhalf_20degBP-mean_point_contournp_mean[0]),axis=1))
#    upperhalf_20deg_distx=np.sqrt(np.sum(np.square(upperhalf_20degPx-mean_point_contournp_mean[0]),axis=1))
#    lowerhalf_20deg_distx=np.sqrt(np.sum(np.square(lowerhalf_20degPx-mean_point_contournp_mean[0]),axis=1))
#    upperhalf_20deg_dist=upperhalf_20deg_distx #upperhalf_20deg_dist*
#    lowerhalf_20deg_dist=lowerhalf_20deg_distx #lowerhalf_20deg_dist*
    
    upperhalf_20deg_dist_min_id= np.argmin(upperhalf_20deg_dist)
    lowerhalf_20deg_dist_min_id= np.argmin(lowerhalf_20deg_dist)
    midline_upper_point= upperhalf_20degBP[upperhalf_20deg_dist_min_id]
    midline_lower_point= lowerhalf_20degBP[lowerhalf_20deg_dist_min_id]
    

    slice_3_layer=cv2.line(slice_3_layer,( int(midline_upper_point[0]),int(midline_upper_point[1])),(int(midline_lower_point[0]),int(midline_lower_point[1])) , (255,0,0), 5)
    
    cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,"Ellipse4.png"),slice_3_layer)   
    
#    cv2.imshow('Contours', slice_3_layer) 
##    cv2.imshow('Grayscale', imgray1) 
##    cv2.waitKey(0)
#    cv2.waitKey(0) 
#    cv2.destroyAllWindows() 
    PointA=( int(midline_upper_point[0]),int(midline_upper_point[1]))
    PointB=(int(midline_lower_point[0]),int(midline_lower_point[1]))
    PointAnp=np.array(PointA)
    PointBnp=np.array(PointB)
    if PointAnp.size==0 or PointBnp.size==0:
        PointA=( int(pointA[0]),int(pointA[1]))
        PointB=(int(pointB[0]),int(pointB[1]))

    return PointA,PointB

def lower_slice_midline_refinement(image_gray_bet,image_gray,pointA,pointB,anglethreshold=20):
    # between 0 to 255
    imgray=np.uint8(image_gray_bet)
    imgray1=np.uint8(image_gray)
    ret, thresh = cv2.threshold(imgray, 127, 255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_3_layer= np.zeros([imgray1.shape[0],imgray.shape[1],3])
    slice_3_layer[:,:,0]=imgray1
    slice_3_layer[:,:,1]=imgray1
    slice_3_layer[:,:,2]=imgray1
    slice_3_layer=np.uint8(slice_3_layer)
    cv2.drawContours(slice_3_layer, contours, 8, (0,255,0), 3)
#    pointA=(int(256.0),int(0.0))
#    pointB=(int(256.0),int(512.0))
    slice_3_layer=cv2.line(slice_3_layer, pointA,pointB , (255,255,0), 2)
    
    # mid point of the line:
    pointAnp=np.array(pointA)
    pointBnp=np.array(pointB)
    mean_point_contour=[]
    contourNitslength=[]
    for each_contour in contours:
        print(len(each_contour))
        each_contourNitslenght=[each_contour,len(each_contour)]
        contourNitslength.append(each_contourNitslenght)
        mean_point_contour.append(np.mean(each_contour,axis=0)[0])
    contourNitslength= np.array(contourNitslength)
    largest_contour=contourNitslength[np.argmax(contourNitslength[:,1]),0]
    mean_point_contournp_mean=np.mean(np.array(largest_contour),axis=0)
    
    
    
    
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_throughapoint(pointAnp,pointBnp,mean_point_contournp_mean[0])
    ## separate the points into upper and lower half:
    upperhalf=[]
    lowerhalf=[]
    for each_point in largest_contour:
        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
        if xx >0:
            upperhalf.append(each_point[0])
        if xx <0:
            lowerhalf.append(each_point[0])
        
    upperhalf=np.array(upperhalf)
    lowerhalf=np.array(lowerhalf)
    upperhalf_20deg=[]
    lowerhalf_20deg=[]
    for each_point in upperhalf:
    #    print(each_point[0])
        v1= np.array(pointAnp) - mean_point_contournp_mean[0]
        v2= each_point - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v2,v1))
        if angle1 <anglethreshold :
            upperhalf_20deg.append(each_point)
            cv2.circle(slice_3_layer,(each_point[0],each_point[1]),2,(0,0,255),1)
    upperhalf_20deg=np.array(upperhalf_20deg)
    for each_point in lowerhalf:
    #    print(each_point[0])
        v1_1= np.array(pointBnp) - mean_point_contournp_mean[0]
        v2= each_point - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v2,v1_1))
        if angle1 <anglethreshold :
            lowerhalf_20deg.append(each_point)
            cv2.circle(slice_3_layer,(each_point[0],each_point[1]),2,(0,0,255),1)
    lowerhalf_20deg=np.array(lowerhalf_20deg)
    upper_lower_pixels=np.array([])
    if lowerhalf_20deg.shape[0]>0 and upperhalf_20deg.shape[0]>0:
        upper_lower_pixels=np.concatenate((upperhalf_20deg, lowerhalf_20deg), axis=0)
    else:
        if lowerhalf_20deg.shape[0]>0 and upperhalf_20deg.shape[0]==0:
            lowerhalf_20deg=lowerhalf_20deg
        elif lowerhalf_20deg.shape[0]==0 and upperhalf_20deg.shape[0]>0:
            upper_lower_pixels=upperhalf_20deg
    
    upperhalf_20deg_dist=np.sqrt(np.sum(np.square(upperhalf_20deg-mean_point_contournp_mean[0]),axis=1))
    lowerhalf_20deg_dist=np.sqrt(np.sum(np.square(lowerhalf_20deg-mean_point_contournp_mean[0]),axis=1))
    upperhalf_20deg_dist_min_id= np.argmin(upperhalf_20deg_dist)
    lowerhalf_20deg_dist_min_id= np.argmin(lowerhalf_20deg_dist)
    midline_upper_point= upperhalf_20deg[upperhalf_20deg_dist_min_id]
    midline_lower_point= lowerhalf_20deg[lowerhalf_20deg_dist_min_id]
    ### angle between the perpendicular line and the contour:
    #segment_contour=[]
    #for each_point in largest_contour:
    ##    print(each_point[0])
    #    v1= np.array(pointAnp) - mean_point_contournp_mean
    #    v1_1= np.array(pointBnp) - mean_point_contournp_mean
    #    v2= each_point[0] - mean_point_contournp_mean
    #    angle1=np.abs(angle_bet_two_vector(v1[0],v2[0]))
    #    if angle1>180:
    #        angle1=angle1-180
    #    angle2=np.abs(angle_bet_two_vector(v1_1[0],v2[0]))
    #    if angle2>180:
    #        angle2=angle2-180
    #    if angle1 <20 or angle2 <20:
    #        print (angle1)
    #        print("\n")
    #        print(angle2)
    #        cv2.circle(slice_3_layer,(each_point[0][0],each_point[0][1]),2,(0,0,255),1)
    #
    #        
    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,255,0), 2)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_bisector(pointAnp,pointBnp)
    #slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,0,0), 2)
    
    slice_3_layer=cv2.line(slice_3_layer,( int(midline_upper_point[0]),int(midline_upper_point[1])),(int(midline_lower_point[0]),int(midline_lower_point[1])) , (255,0,0), 5)
    
    
    
#    cv2.imshow('Contours', slice_3_layer) 
##    cv2.imshow('Grayscale', imgray1) 
##    cv2.waitKey(0)
#    cv2.waitKey(0) 
#    cv2.destroyAllWindows() 
    return midline_upper_point, midline_lower_point

def upper_slice_midline_refinement(image_gray_bet,image_gray,gabor_image,pointA,pointB,anglethreshold=20):
#    upper_slice_midline_refinement(np.uint8(img_gray_bet.get_fdata()[:,:,i]*255),gray,filtered_img,(pointA[0],pointA[1]),(pointB[0],pointB[1]) )
#
#    image_gray_bet= np.uint8(img_gray_bet.get_fdata()[:,:,i]*255)
#    image_gray= gray
#    gabor_image= filtered_img

    # between 0 to 255
    binary_image=gabor_image
    binary_image_copy=np.copy(binary_image)
    slope_of_line=100000
    score_diff_from_1= 100000
    imgray=image_gray_bet
    imgray1=image_gray 
    ret, thresh = cv2.threshold(imgray, 127, 255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_3_layer= np.zeros([imgray1.shape[0],imgray.shape[1],3])
    slice_3_layer[:,:,0]= binary_image #imgray1
    slice_3_layer[:,:,1]=binary_image #imgray1
    slice_3_layer[:,:,2]=binary_image# imgray1
    slice_3_layer=np.uint8(slice_3_layer)
#    cv2.drawContours(slice_3_layer, contours, -1, (0,255,0), 3)
#    pointA=(int(256.0),int(0.0))
#    pointB=(int(256.0),int(512.0))
    slice_3_layer=cv2.line(slice_3_layer, ( int(pointA[0]),int(pointA[1])),(int(pointB[0]),int(pointB[1])), (255,255,0), 2)
    
    # mid point of the line:
    pointAnp=np.array(pointA)
    pointBnp=np.array(pointB)
    mean_point_contour=[]
    contourNitslength=[]
    for each_contour in contours:
#        print(len(each_contour))
        each_contourNitslenght=[each_contour,len(each_contour)]
        contourNitslength.append(each_contourNitslenght)
        mean_point_contour.append(np.mean(each_contour,axis=0)[0])
    contourNitslength= np.array(contourNitslength)
    largest_contour=contourNitslength[np.argmax(contourNitslength[:,1]),0]
    mean_point_contournp_mean=np.mean(np.array(largest_contour),axis=0)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_throughapoint(pointAnp,pointBnp,mean_point_contournp_mean[0])
    slice_3_layer=cv2.line(slice_3_layer, ( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])), (255,255,0), 2)
    ## separate the points into upper and lower half:
    upperhalf=[]
    lowerhalf=[]

    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    X=binary_image_non_zero_cord[:,1].reshape(-1,1)
    gabor_image_pixels_cord=np.zeros([len(X),2])
    #    print(len(binary_image_non_zero_cord))
    if len(binary_image_non_zero_cord)>300:
        X=binary_image_non_zero_cord[:,1].reshape(-1,1)
        y=binary_image_non_zero_cord[:,0].reshape(-1,1)
        gabor_image_pixels_cord[:,0]=np.transpose(X)
        gabor_image_pixels_cord[:,1]=np.transpose(y)
#    print(largest_contour)
    for each_point in gabor_image_pixels_cord:
        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point)
        if xx > 0:
            upperhalf.append(each_point)
        if xx < 0:
            lowerhalf.append(each_point)
#    for each_point in largest_contour:
#        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
#        if xx >0:
#            upperhalf.append(each_point[0])
#        if xx <0:
#            lowerhalf.append(each_point[0])
        
    upperhalf=np.array(upperhalf)
    lowerhalf=np.array(lowerhalf)
#    for each_point in upperhalf:
#        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
#    for each_point in lowerhalf:
#        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,0),1)
    upperhalf_20deg=[]
    lowerhalf_20deg=[]
    for each_point in upperhalf:
    #    print(each_point[0])
        v1= np.array(pointBnp) - mean_point_contournp_mean[0]
        v2= each_point - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1,v2))
        if angle1 <anglethreshold :
            upperhalf_20deg.append(each_point)
            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    upperhalf_20deg=np.array(upperhalf_20deg)
    for each_point in lowerhalf:
    #    print(each_point[0])
        v1= np.array(pointAnp) - mean_point_contournp_mean[0]
        v2= each_point - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1,v2))
        if angle1>270:
#            print(360-angle1)
            angle1=360-angle1
        if angle1 <anglethreshold :
            lowerhalf_20deg.append(each_point)
            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
#    for each_point in lowerhalf:
#    #    print(each_point[0])
#        v1_1= np.array(pointAnp) - mean_point_contournp_mean[0]
#        v2= each_point - mean_point_contournp_mean[0]
#        angle1=np.abs(angle_bet_two_vector(v1_1,v2))
#        print(angle1)
#        if angle1 <anglethreshold :
#            lowerhalf_20deg.append(each_point)
#            cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    lowerhalf_20deg=np.array(lowerhalf_20deg)
#    print('lowerhalf_20deg -----------------------------------------------------------------------------------')
#    print(lowerhalf_20deg)
#    print(upperhalf_20deg)
    upper_lower_pixels=np.array([])
    if lowerhalf_20deg.shape[0]>0 and upperhalf_20deg.shape[0]>0:
        upper_lower_pixels=np.concatenate((upperhalf_20deg, lowerhalf_20deg), axis=0)
    else:
        if lowerhalf_20deg.shape[0]>0 and upperhalf_20deg.shape[0]==0:
            lowerhalf_20deg=lowerhalf_20deg
        elif lowerhalf_20deg.shape[0]==0 and upperhalf_20deg.shape[0]>0:
            upper_lower_pixels=upperhalf_20deg
                
    if np.array(upper_lower_pixels).shape[0]>100:
        reg = RANSACRegressor(random_state=0,min_samples=100,residual_threshold=3, max_trials=10000).fit(upper_lower_pixels[:,0].reshape(-1, 1),upper_lower_pixels[:,1].reshape(-1, 1))
#        print(1-reg.score(X, y))
        score_diff_from_1= 1-reg.score(X, y)
        X_test=np.arange(0,512).reshape(-1, 1)
        Y_pred = reg.predict(X_test) 
        slope_of_line=(Y_pred[511] - Y_pred[0]) / (X_test[511]-X_test[0])  #np.array([X_test[511],Y_pred[511]])-np.array([X_test[0],Y_pred[0]])
        pointA= np.array([X_test[0][0],Y_pred[0][0]])
        pointB= np.array([X_test[511][0],Y_pred[511][0]])
        for  each_point in upper_lower_pixels: # binary_image_non_zero_cord : #range(0,512): #
            if len(each_point)==2:
#                print(each_point)
                cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(0,200,0),1)           
        for k in range(len(X_test)):
            cv2.circle(slice_3_layer,(X_test[k],Y_pred[k]),2,(188,0,0),1)    
#    upperhalf_20deg_dist=np.sqrt(np.sum(np.square(upperhalf_20deg-mean_point_contournp_mean[0]),axis=1))
#    lowerhalf_20deg_dist=np.sqrt(np.sum(np.square(lowerhalf_20deg-mean_point_contournp_mean[0]),axis=1))
#    upperhalf_20deg_dist_min_id= np.argmin(upperhalf_20deg_dist)
#    lowerhalf_20deg_dist_min_id= np.argmin(lowerhalf_20deg_dist)
#    midline_upper_point= upperhalf_20deg[upperhalf_20deg_dist_min_id]
#    midline_lower_point= lowerhalf_20deg[lowerhalf_20deg_dist_min_id]      
#    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,0,0), 2)
#    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_bisector(pointAnp,pointBnp)
#    #slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,0,0), 2)
#    
#    slice_3_layer=cv2.line(slice_3_layer,( int(midline_upper_point[0]),int(midline_upper_point[1])),(int(midline_lower_point[0]),int(midline_lower_point[1])) , (255,0,0), 5)
#    
#    
    
#    cv2.imshow('Contours', slice_3_layer) 
##    cv2.imshow('Grayscale', imgray1) 
##    cv2.waitKey(0)
#    cv2.waitKey(0) 
#    cv2.destroyAllWindows() 
    return score_diff_from_1 ,binary_image_copy, slope_of_line,pointA,pointB

def get_midline_pointswithregto664_1(static_file,moving_file,moving_file_intact):
    dirname=os.path.basename(os.path.dirname(static_file))
    result_directory=os.path.dirname(moving_file_intact)
    template_directory=os.path.dirname(static_file) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664"
    test_directory=os.path.dirname(moving_file) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_744" 
    moving_filename=os.path.basename(moving_file) #"WUSTL_744_12152018_1940_part1_Head_Spiral_3.0_J40s_2_2018121519404BETgray.nii.gz"
    fixed_filename=os.path.basename(static_file) #"WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
    
    fixed = nib.load(os.path.join(template_directory,fixed_filename))  # ants.get_ants_data('r16') )
    moving = nib.load(os.path.join(test_directory,moving_filename))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    #
    #
    fixed = ants.image_read(static_file)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    typeofreg="Affine"
    mytx,fixed_new= register_twoimages_test(fixed,moving,type_of_transform= typeofreg)
    ###ants.image_write(fixed_new,'fixed_image2d.jpg')
    moving1 = ants.image_read(moving_file_intact) #os.path.join(result_directory,"moving_slice.nii.gz")) #.resample_image(img_gray_ctp.GetSize())
    mywarpedimage=apply_transformation(fixed, moving1,mytx)
    ##
    reg_output_filename=os.path.join(result_directory,os.path.basename(moving_file_intact)[:-7]+"REG"+typeofreg+dirname+".nii.gz")
    ants.image_write(mywarpedimage,reg_output_filename)  #'movingR.nii.gz')
    
    moving1 = ants.image_read(moving_file) #os.path.join(result_directory,"moving_slice.nii.gz")) #.resample_image(img_gray_ctp.GetSize())
    mywarpedimage=apply_transformation(fixed, moving1,mytx)
    ##
    reg_output_filename1=os.path.join(result_directory,os.path.basename(moving_file_intact)[:-7]+"GrayREG"+typeofreg+dirname+".nii.gz")
    ants.image_write(mywarpedimage,reg_output_filename1)  #'movingR.nii.gz')
    
    ## Get the registered midine mask file:
    midline_nifti_file= reg_output_filename #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midline664.nii.gz"
    midline_nifti=nib.load(midline_nifti_file)
    midline_nifti_np=midline_nifti.get_fdata()
    fixed1 = nib.load(static_file)
    fixed1_np=fixed1.get_fdata()
    #gray_image_nib_np=np.copy(midline_nifti_np)
    #gray_image_nib_np[gray_image_nib_np>=0]=0
    #gray_image_nib_np[gray_image_nib_np<0]=0
    slope_of_lines=[] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
    image_list=[]
    for xx in range(midline_nifti_np.shape[2]):
        this_slice=midline_nifti_np[:,:,xx]
        if np.sum(this_slice)>0:
            print(xx)
            non_zero_cord=np.nonzero(this_slice)
            coefficients = np.polyfit(non_zero_cord[1], non_zero_cord[0], 1)
            polynomial = np.poly1d(coefficients)
    #        x_axis = np.linspace(0,511,511)
    #        y_axis = polynomial(x_axis)
            x_axis_1 = np.linspace(0,511,2)
            y_axis_1 = polynomial(x_axis_1)
    #        for x in x_axis:
    #            for y in y_axis:
    #                gray_image_nib_np[int(y),int(x),xx]=255
            pointA_lower=[x_axis_1[0],y_axis_1[0]]
            pointB_lower=[x_axis_1[1],y_axis_1[1]]
#            show_slice_withaline(np.uint8(fixed1_np[:,:,xx]),np.array([pointA_lower,pointB_lower])) 
#            if xx == 0:
            slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
#            image_list.append(fixed[:,:,xx])
#            if xx==midline_nifti_np.shape[2]-1:
#                slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
    return slope_of_lines


def get_midline_pointswithregto664(static_file,moving_file,moving_file_intact):
    dirname=os.path.basename(os.path.dirname(static_file))
    result_directory=os.path.dirname(moving_file_intact)
    template_directory=os.path.dirname(static_file) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664"
    test_directory=os.path.dirname(moving_file) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_744" 
    moving_filename=os.path.basename(moving_file) #"WUSTL_744_12152018_1940_part1_Head_Spiral_3.0_J40s_2_2018121519404BETgray.nii.gz"
    fixed_filename=os.path.basename(static_file) #"WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
    
    fixed = nib.load(os.path.join(template_directory,fixed_filename))  # ants.get_ants_data('r16') )
    moving = nib.load(os.path.join(test_directory,moving_filename))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    #
    #
    fixed = ants.image_read(static_file)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    typeofreg="Affine"
    mytx,fixed_new= register_twoimages_test(fixed,moving,type_of_transform= typeofreg)
    ###ants.image_write(fixed_new,'fixed_image2d.jpg')
    moving1 = ants.image_read(moving_file_intact) #os.path.join(result_directory,"moving_slice.nii.gz")) #.resample_image(img_gray_ctp.GetSize())
    mywarpedimage=apply_transformation(fixed, moving1,mytx)
    ##
    reg_output_filename=os.path.join(result_directory,os.path.basename(moving_file_intact)[:-7]+"REG"+typeofreg+dirname+".nii.gz")
    ants.image_write(mywarpedimage,reg_output_filename)  #'movingR.nii.gz')
    
    moving1 = ants.image_read(moving_file) #os.path.join(result_directory,"moving_slice.nii.gz")) #.resample_image(img_gray_ctp.GetSize())
    mywarpedimage=apply_transformation(fixed, moving1,mytx)
    ##
    reg_output_filename1=os.path.join(result_directory,os.path.basename(moving_file_intact)[:-7]+"GrayREG"+typeofreg+dirname+".nii.gz")
    ants.image_write(mywarpedimage,reg_output_filename1)  #'movingR.nii.gz')
    
    ## Get the registered midine mask file:
    midline_nifti_file= reg_output_filename #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midline664.nii.gz"
    midline_nifti=nib.load(midline_nifti_file)
    midline_nifti_np=midline_nifti.get_fdata()
    fixed1 = nib.load(static_file)
    fixed1_np=fixed1.get_fdata()
    #gray_image_nib_np=np.copy(midline_nifti_np)
    #gray_image_nib_np[gray_image_nib_np>=0]=0
    #gray_image_nib_np[gray_image_nib_np<0]=0
    slope_of_lines=[] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
    first_xx=0
    last_xx=0
    for xx in range(midline_nifti_np.shape[2]):
        this_slice=midline_nifti_np[:,:,xx]
        if np.sum(this_slice)>0:
            print(xx)
            non_zero_cord=np.nonzero(this_slice)
            coefficients = np.polyfit(non_zero_cord[1], non_zero_cord[0], 1)
            polynomial = np.poly1d(coefficients)
    #        x_axis = np.linspace(0,511,511)
    #        y_axis = polynomial(x_axis)
            x_axis_1 = np.linspace(0,511,2)
            y_axis_1 = polynomial(x_axis_1)
    #        for x in x_axis:
    #            for y in y_axis:
    #                gray_image_nib_np[int(y),int(x),xx]=255
            pointA_lower=[x_axis_1[0],y_axis_1[0]]
            pointB_lower=[x_axis_1[1],y_axis_1[1]]

            if np.sum(np.array(pointA_lower)) > 0 and np.sum(np.array(pointB_lower)) > 0 : 
                
#            show_slice_withaline(np.uint8(fixed1_np[:,:,xx]),np.array([pointA_lower,pointB_lower])) 
#            if xx == 0:
                slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
                print(" point 1 selected")
                break
    for xx in reversed(range(midline_nifti_np.shape[2])):
        this_slice=midline_nifti_np[:,:,xx]
        if np.sum(this_slice)>0:
            print(xx)
            non_zero_cord=np.nonzero(this_slice)
            coefficients = np.polyfit(non_zero_cord[1], non_zero_cord[0], 1)
            polynomial = np.poly1d(coefficients)
    #        x_axis = np.linspace(0,511,511)
    #        y_axis = polynomial(x_axis)
            x_axis_1 = np.linspace(0,511,2)
            y_axis_1 = polynomial(x_axis_1)
    #        for x in x_axis:
    #            for y in y_axis:
    #                gray_image_nib_np[int(y),int(x),xx]=255
            pointA_lower=[x_axis_1[0],y_axis_1[0]]
            pointB_lower=[x_axis_1[1],y_axis_1[1]]

            if np.sum(np.array(pointA_lower)) > 0 and np.sum(np.array(pointB_lower)) > 0 : 
                
    #            show_slice_withaline(np.uint8(fixed1_np[:,:,xx]),np.array([pointA_lower,pointB_lower])) 
    #            if xx == 0:
    #            slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
    #            print(" point 1 selected")
    #            break
                
                
    #            if xx==midline_nifti_np.shape[2]-1:
                slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
                print(" point 2 selected")
                print(midline_nifti_np.shape[2])
                break
    return slope_of_lines

def betbinaryfrombetgray(inputdirectory,outputdirectory,betgrayfileext):
    ## take the grayscalefiles in the inputdirectory
    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)

    for eachgrayfiles in allgrayfiles:
        niifilenametosave=os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")
        print('eachgrayfiles')
        print(eachgrayfiles)
        gray_nifti=nib.load(eachgrayfiles)
        gray_nifti_data=gray_nifti.get_fdata()
        gray_nifti_data[gray_nifti_data>0]=1
        array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
        nib.save(array_img, niifilenametosave)
        
    ## convert it into binary
    ## save it as nii.gz file
def betbinaryfrombetgray_v1(inputdirectory,outputdirectory,betgrayfileext):
    ## take the grayscalefiles in the inputdirectory
    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)

    for eachgrayfiles in allgrayfiles:
        niifilenametosave=os.path.join(outputdirectory,os.path.basename(eachgrayfiles).split(".nii")[0]+"BET.nii.gz")
        print('eachgrayfiles')
        print(eachgrayfiles)
        gray_nifti=nib.load(eachgrayfiles)
        gray_nifti_data=gray_nifti.get_fdata()
        gray_nifti_data[gray_nifti_data>np.min(gray_nifti_data)]=1
        gray_nifti_data[gray_nifti_data<=np.min(gray_nifti_data)]=0
        array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
        nib.save(array_img, niifilenametosave)
        
    ## convert it into binary
    ## save it as nii.gz file
   
def get_midline_pointswithregto664_2(static_file,moving_file,moving_file_intact):
    dirname=os.path.basename(os.path.dirname(static_file))
    result_directory=os.path.dirname(moving_file_intact)
    template_directory=os.path.dirname(static_file) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664"
    test_directory=os.path.dirname(moving_file) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_744" 
    moving_filename=os.path.basename(moving_file) #"WUSTL_744_12152018_1940_part1_Head_Spiral_3.0_J40s_2_2018121519404BETgray.nii.gz"
    fixed_filename=os.path.basename(static_file) #"WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
    
    fixed = nib.load(os.path.join(template_directory,fixed_filename))  # ants.get_ants_data('r16') )
    moving = nib.load(os.path.join(test_directory,moving_filename))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    #
    #
    fixed = ants.image_read(static_file)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    typeofreg="Affine" #"ElasticSyN" #
    mytx,fixed_new= register_twoimages_test(fixed,moving,type_of_transform= typeofreg)
    ###ants.image_write(fixed_new,'fixed_image2d.jpg')
    moving1 = ants.image_read(moving_file_intact) #os.path.join(result_directory,"moving_slice.nii.gz")) #.resample_image(img_gray_ctp.GetSize())
    mywarpedimage=apply_transformation_1(fixed, moving1,mytx)
    ##
    reg_output_filename=os.path.join(result_directory,os.path.basename(moving_file_intact)[:-7]+"REG"+typeofreg+dirname+".nii.gz")
    ants.image_write(mywarpedimage,reg_output_filename)  #'movingR.nii.gz')
    
    moving1 = ants.image_read(moving_file) #os.path.join(result_directory,"moving_slice.nii.gz")) #.resample_image(img_gray_ctp.GetSize())
    mywarpedimage=apply_transformation_1(fixed, moving1,mytx)
    ##
    reg_output_filename1=os.path.join(result_directory,os.path.basename(moving_file_intact)[:-7]+"GrayREG"+typeofreg+dirname+".nii.gz")
    ants.image_write(mywarpedimage,reg_output_filename1)  #'movingR.nii.gz')
    
    ## Get the registered midine mask file:
    midline_nifti_file= reg_output_filename #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midline664.nii.gz"
    midline_nifti=nib.load(midline_nifti_file)
    midline_nifti_np=midline_nifti.get_fdata()
    fixed1 = nib.load(static_file)
    fixed1_np=fixed1.get_fdata()
    #gray_image_nib_np=np.copy(midline_nifti_np)
    #gray_image_nib_np[gray_image_nib_np>=0]=0
    #gray_image_nib_np[gray_image_nib_np<0]=0
    slope_of_lines=[] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
    first_xx=0
    last_xx=0
    numberofslices=1
    if len(midline_nifti_np.shape)>2:
        numberofslices=midline_nifti_np.shape[2]
    else:
        numberofslices=1
        
    for xx in range(numberofslices): #midline_nifti_np.shape[2]):
        
        if numberofslices>1:
            this_slice=midline_nifti_np[:,:,xx]
        else:
            this_slice=midline_nifti_np #[:,:,xx]
        if np.sum(this_slice)>0:
            print(xx)
            non_zero_cord=np.nonzero(this_slice)
            coefficients = np.polyfit(non_zero_cord[1], non_zero_cord[0], 1)
            polynomial = np.poly1d(coefficients)
    #        x_axis = np.linspace(0,511,511)
    #        y_axis = polynomial(x_axis)
            x_axis_1 = np.linspace(0,511,2)
            y_axis_1 = polynomial(x_axis_1)
    #        for x in x_axis:
    #            for y in y_axis:
    #                gray_image_nib_np[int(y),int(x),xx]=255
            pointA_lower=[x_axis_1[0],y_axis_1[0]]
            pointB_lower=[x_axis_1[1],y_axis_1[1]]

            if np.sum(np.array(pointA_lower)) > 0 and np.sum(np.array(pointB_lower)) > 0 : 
                
#            show_slice_withaline(np.uint8(fixed1_np[:,:,xx]),np.array([pointA_lower,pointB_lower])) 
#            if xx == 0:
                slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
                print(" point 1 selected")
                break
    for xx in reversed(range(numberofslices)): #midline_nifti_np.shape[2])):
        if numberofslices>1:
            this_slice=midline_nifti_np[:,:,xx]
        else:
            this_slice=midline_nifti_np #[:,:,xx]
        if np.sum(this_slice)>0:
            print(xx)
            non_zero_cord=np.nonzero(this_slice)
            coefficients = np.polyfit(non_zero_cord[1], non_zero_cord[0], 1)
            polynomial = np.poly1d(coefficients)
    #        x_axis = np.linspace(0,511,511)
    #        y_axis = polynomial(x_axis)
            x_axis_1 = np.linspace(0,511,2)
            y_axis_1 = polynomial(x_axis_1)
    #        for x in x_axis:
    #            for y in y_axis:
    #                gray_image_nib_np[int(y),int(x),xx]=255
            pointA_lower=[x_axis_1[0],y_axis_1[0]]
            pointB_lower=[x_axis_1[1],y_axis_1[1]]

            if np.sum(np.array(pointA_lower)) > 0 and np.sum(np.array(pointB_lower)) > 0 : 
                
    #            show_slice_withaline(np.uint8(fixed1_np[:,:,xx]),np.array([pointA_lower,pointB_lower])) 
    #            if xx == 0:
    #            slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
    #            print(" point 1 selected")
    #            break
                
                
    #            if xx==midline_nifti_np.shape[2]-1:
                slope_of_lines.append([0,np.array(pointA_lower),np.array(pointB_lower),xx])
                print(" point 2 selected")
                print(numberofslices) #midline_nifti_np.shape[2])
                break
    return slope_of_lines

def saveoneslicein3Dpy(grayfile,slicenum,niifilenametosave):
#    grayfile=sys.argv[1]
#    slicenum=int(sys.argv[2])
    ## take the grayscalefiles in the inputdirectory
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#
#    for eachgrayfiles in allgrayfiles:
#    niifilenametosave=os.path.join(grayfile.split(".nii")[0]+"SLICENUM"+ str(slicenum)+".nii.gz")
    print('grayfile')
    print(grayfile)
    gray_nifti=nib.load(grayfile)
    gray_nifti_data=gray_nifti.get_fdata()
    for x in range(gray_nifti_data.shape[2]):
        if x != slicenum:
            gray_nifti_data[:,:,x]=np.min(gray_nifti_data)
            
#    gray_nifti_data[gray_nifti_data>0]=1
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    
def saveoneslicein2Dpy(grayfile,slicenum,niifilenametosave):
#    grayfile=sys.argv[1]
#    slicenum=int(sys.argv[2])
    ## take the grayscalefiles in the inputdirectory
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#
#    for eachgrayfiles in allgrayfiles:
#    niifilenametosave=os.path.join(grayfile.split(".nii")[0]+"SLICENUM"+ str(slicenum)+".nii.gz")
    print('grayfile')
    print(grayfile)
    gray_nifti=nib.load(grayfile)
    gray_nifti_data=gray_nifti.get_fdata()
    slicetosave=gray_nifti_data[:,:,slicenum]
#    for x in range(gray_nifti_data.shape[2]):
#        if x != slicenum:
#            gray_nifti_data[:,:,x]=np.min(gray_nifti_data)
            
#    gray_nifti_data[gray_nifti_data>0]=1
    array_img = nib.Nifti1Image(slicetosave, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
def betgrayfrombetbinary(inputfile,betgrayfile):
    ## take the grayscalefiles in the inputdirectory
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)

#    for eachgrayfiles in allgrayfiles:
    eachgrayfiles=inputfile
    niifilenametosave=inputfile.split(".nii")[0] + "BETgray.nii.gz" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")
    print('eachgrayfiles')
    print(eachgrayfiles)
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    bet_nifti=nib.load(betgrayfile)
    bet_nifti_data=bet_nifti.get_fdata()
    gray_nifti_data[bet_nifti_data<1]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    return niifilenametosave

def saveoneslicein3D():
    grayfile=sys.argv[1]
    slicenum=int(sys.argv[2])
    ## take the grayscalefiles in the inputdirectory
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)
#
#    for eachgrayfiles in allgrayfiles:
    niifilenametosave=os.path.join(grayfile.split(".nii")[0]+"SLICENUM"+ str(slicenum)+".nii.gz")
    print('grayfile')
    print(grayfile)
    gray_nifti=nib.load(grayfile)
    gray_nifti_data=gray_nifti.get_fdata()
    for x in range(gray_nifti_data.shape[2]):
        if x != slicenum:
            gray_nifti_data[:,:,x]=np.min(gray_nifti_data)
            
#    gray_nifti_data[gray_nifti_data>0]=1
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
def betgrayfrombetbinary1(inputfile,betgrayfile):
    ## take the grayscalefiles in the inputdirectory betgrayfile== betbinary
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)

#    for eachgrayfiles in allgrayfiles:
    eachgrayfiles=inputfile
    niifilenametosave=inputfile.split(".nii")[0] + "BETgray.nii" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")
    print('eachgrayfiles')
    print(eachgrayfiles)
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    bet_nifti=nib.load(betgrayfile)
    bet_nifti_data=bet_nifti.get_fdata()
    gray_nifti_data[bet_nifti_data<np.max(bet_nifti_data)]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    return niifilenametosave

def betgrayfrombetbinary1_sh():
    inputfile=sys.argv[1]
    betgrayfile=sys.argv[2]
    ## take the grayscalefiles in the inputdirectory betgrayfile== betbinary
#    allgrayfiles=glob.glob(inputdirectory+ "/*" + betgrayfileext)

#    for eachgrayfiles in allgrayfiles:
    eachgrayfiles=inputfile
    niifilenametosave=inputfile.split(".nii")[0] + "BETgray.nii.gz" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")
    print('eachgrayfiles')
    print(eachgrayfiles)
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    bet_nifti=nib.load(betgrayfile)
    bet_nifti_data=bet_nifti.get_fdata()
    gray_nifti_data[bet_nifti_data<np.max(bet_nifti_data)]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    return niifilenametosave

def fronthalf_backhalfpy(eachgrayfiles):
#    eachgrayfiles=sys.argv[1]
#    seg_skullbone(niifilenametosave)
    niifilenametosave=eachgrayfiles.split(".nii")[0] + "FrontHalf.nii.gz" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")

    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    for x in range(gray_nifti_data.shape[2]):
        gray_nifti_data[:,0:255,x]=np.min(gray_nifti_data)
        
#    bet_nifti=nib.load(betgrayfile)
#    bet_nifti_data=bet_nifti.get_fdata()
#    gray_nifti_data[bet_nifti_data<1]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)

    niifilenametosave=eachgrayfiles.split(".nii")[0] + "BackHalf.nii.gz" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")

    
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    for x in range(gray_nifti_data.shape[2]):
        gray_nifti_data[:,255:gray_nifti_data.shape[0],x]=np.min(gray_nifti_data)
        
#    bet_nifti=nib.load(betgrayfile)
#    bet_nifti_data=bet_nifti.get_fdata()
#    gray_nifti_data[bet_nifti_data<1]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
#    seg_skullbone(niifilenametosave)
   
def fronthalf_backhalf():
    eachgrayfiles=sys.argv[1]
    niifilenametosave=eachgrayfiles.split(".nii")[0] + "FrontHalf.nii.gz" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")

    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    for x in range(gray_nifti_data.shape[2]):
        gray_nifti_data[:,0:255,x]=np.min(gray_nifti_data)
        
#    bet_nifti=nib.load(betgrayfile)
#    bet_nifti_data=bet_nifti.get_fdata()
#    gray_nifti_data[bet_nifti_data<1]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    seg_skullbone(niifilenametosave)
    niifilenametosave=eachgrayfiles.split(".nii")[0] + "BackHalf.nii.gz" #outputfilename #os.path.join(outputdirectory,os.path.basename(eachgrayfiles)[:-7]+"_bet.nii.gz")

    
    gray_nifti=nib.load(eachgrayfiles)
    gray_nifti_data=gray_nifti.get_fdata()
    for x in range(gray_nifti_data.shape[2]):
        gray_nifti_data[:,255:gray_nifti_data.shape[0],x]=np.min(gray_nifti_data)
        
#    bet_nifti=nib.load(betgrayfile)
#    bet_nifti_data=bet_nifti.get_fdata()
#    gray_nifti_data[bet_nifti_data<1]=np.min(gray_nifti_data)
    array_img = nib.Nifti1Image(gray_nifti_data, affine=gray_nifti.affine, header=gray_nifti.header)
    nib.save(array_img, niifilenametosave)
    seg_skullbone(niifilenametosave)
def matchfrontpy(staticfile,movingfile,slicenumbertomatch,startingslice):
    staticfile_skull=staticfile.split(".nii")[0]+"BETgray.nii.gz" #_SKULL
    if not os.path.exists(staticfile_skull):
#        seg_skullbone(staticfile)
        create_BETgray(staticfile)
        
    movingfile_skull=movingfile.split(".nii")[0]+"BETgray.nii.gz" #_SKULL
    if not os.path.exists(movingfile_skull):
#        seg_skullbone(movingfile)
        create_BETgray(movingfile)
        


    staticfile_fronthalf=staticfile_skull.split(".nii")[0] + "FrontHalf.nii.gz" #
    if not os.path.exists(staticfile_fronthalf):
        fronthalf_backhalfpy(staticfile_skull)
    

    movingfile_fronthalf=movingfile_skull.split(".nii")[0] + "FrontHalf.nii.gz" #
    if not os.path.exists(movingfile_fronthalf):
        fronthalf_backhalfpy(movingfile_skull)
        
#    slicenumbertomatch=int(sys.argv[3])
    staticfile=staticfile_fronthalf
    movingfile=movingfile_fronthalf
    niifilenametosave=staticfile.split(".nii")[0]+ "SLICENUM" + str(slicenumbertomatch) + ".nii.gz"
    if not os.path.exists(niifilenametosave):
        saveoneslicein2Dpy(staticfile,slicenumbertomatch,niifilenametosave)
#    gray_nifti_data=nib.load(movingfile).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
#    z_dim=nib.load(movingfile).header["pixdim"][3]
    z_dim_ud= 1 #int(z_dim/3) + 1
    for x in range(int(nib.load(movingfile).get_fdata().shape[2]/2),int(nib.load(movingfile).get_fdata().shape[2])):  #range(startingslice-z_dim_ud,startingslice+z_dim_ud): #range(int(nib.load(movingfile).get_fdata().shape[2]/3),int(nib.load(movingfile).get_fdata().shape[2]/2)): #range(startingslice-z_dim_ud,startingslice+z_dim_ud) : #range(0,int(gray_nifti_data.shape[2]/2)):
        niifilenametosave_moving=movingfile.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
        ants.image_write(mywarpedimage,reg_output_filename1) 
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select
def findmatching_eyeslice_using_reg(staticfile,movingfile,slicenumbertomatch,startingslice):
#    staticfile_skull=staticfile.split(".nii")[0]+"BETgray.nii.gz" #_SKULL
#    if not os.path.exists(staticfile_skull):
##        seg_skullbone(staticfile)
#        create_BETgray(staticfile)
#        
#    movingfile_skull=movingfile.split(".nii")[0]+"BETgray.nii.gz" #_SKULL
#    if not os.path.exists(movingfile_skull):
##        seg_skullbone(movingfile)
#        create_BETgray(movingfile)
#        
#
#
#    staticfile_fronthalf=staticfile_skull.split(".nii")[0] + "FrontHalf.nii.gz" #
#    if not os.path.exists(staticfile_fronthalf):
#        fronthalf_backhalfpy(staticfile_skull)
#    
#
#    movingfile_fronthalf=movingfile_skull.split(".nii")[0] + "FrontHalf.nii.gz" #
#    if not os.path.exists(movingfile_fronthalf):
#        fronthalf_backhalfpy(movingfile_skull)
#        
##    slicenumbertomatch=int(sys.argv[3])
#    staticfile=staticfile_fronthalf
#    movingfile=movingfile_fronthalf
    niifilenametosave=staticfile.split(".nii")[0]+ "SLICENUM" + str(slicenumbertomatch) + ".nii.gz"
    if not os.path.exists(niifilenametosave):
        saveoneslicein2Dpy(staticfile,slicenumbertomatch,niifilenametosave)
#    gray_nifti_data=nib.load(movingfile).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
#    z_dim=nib.load(movingfile).header["pixdim"][3]
    z_dim_ud= 3 #int(z_dim/3) + 1
    for x in range(startingslice-z_dim_ud,startingslice+z_dim_ud): #range(int(nib.load(movingfile).get_fdata().shape[2]/3),int(nib.load(movingfile).get_fdata().shape[2]/2)): #range(startingslice-z_dim_ud,startingslice+z_dim_ud) : #range(0,int(gray_nifti_data.shape[2]/2)):
        niifilenametosave_moving=movingfile.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
        ants.image_write(mywarpedimage,reg_output_filename1) 
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if np.abs(0-error) < minimum_error:
            minimum_error=np.abs(0-error) 
            slice_to_select=x
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select
def matchfront():
    seg_skullbone(sys.argv[1])
    seg_skullbone(sys.argv[2])
    staticfile=sys.argv[1].split(".nii")[0]+"_SKULL.nii.gz"
    movingfile=sys.argv[2].split(".nii")[0]+"_SKULL.nii.gz"
    fronthalf_backhalfpy(staticfile)
    fronthalf_backhalfpy(movingfile)
    staticfile=staticfile.split(".nii")[0] + "FrontHalf.nii.gz" #

    movingfile=movingfile.split(".nii")[0] + "FrontHalf.nii.gz" #

    slicenumbertomatch=int(sys.argv[3])
    niifilenametosave=staticfile.split(".nii")[0]+ "SLICENUM" + sys.argv[3] + ".nii.gz"
    saveoneslicein2Dpy(staticfile,slicenumbertomatch,niifilenametosave)
    gray_nifti_data=nib.load(movingfile).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
    for x in range(0,int(gray_nifti_data.shape[2]/2)):
        niifilenametosave_moving=staticfile.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
        ants.image_write(mywarpedimage,reg_output_filename1) 
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select
def topslice_matching_refine_py(staticfile_gray_bet,movingfile_gray_bet,slicenumbertomatch,startingslice):
#    seg_skullbone(staticfile) 
#    matched_top_slice=topslice_matching_refine_py(moving_file,niigzfilenametosave,slicenumbertomatch,matched_upper_slice)
#    staticfile_gray_bet=moving_file
#    movingfile_gray_bet=niigzfilenametosave
#    startingslice=matched_upper_slice
#    seg_skullbone(movingfile)
#    staticfile=staticfile.split(".nii")[0]+"_SKULL.nii.gz"
#    movingfile=movingfile.split(".nii")[0]+"_SKULL.nii.gz"
#    fronthalf_backhalfpy(staticfile)
#    fronthalf_backhalfpy(movingfile)
#    staticfile=staticfile.split(".nii")[0] + "BackHalf.nii.gz" #
#
#    movingfile=movingfile.split(".nii")[0] + "BackHalf.nii.gz" #

#    slicenumbertomatch=int(sys.argv[3])
    niifilenametosave=staticfile_gray_bet.split(".nii")[0]+ "SLICENUM" + str(slicenumbertomatch) + ".nii.gz"
    saveoneslicein2Dpy(staticfile_gray_bet,slicenumbertomatch,niifilenametosave)
#    gray_nifti_data=nib.load(movingfile).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
    for x in range(startingslice-1,startingslice+1): #int(gray_nifti_data.shape[2]/2)):
        niifilenametosave_moving=staticfile_gray_bet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile_gray_bet,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile_gray_bet.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
        ants.image_write(mywarpedimage,reg_output_filename1) 
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select



def matchbackpy(staticfile,movingfile,slicenumbertomatch,startingslice):
    staticfile_skull=staticfile.split(".nii")[0]+"BETgray.nii.gz" #_SKULL
    if not os.path.exists(staticfile_skull):
        create_BETgray(staticfile)
#        seg_skullbone(staticfile)
        
    movingfile_skull=movingfile.split(".nii")[0]+"BETgray.nii.gz" #_SKULL
    if not os.path.exists(movingfile_skull):
        create_BETgray(movingfile)
#        seg_skullbone(movingfile)
        


    staticfile_backhalf=staticfile_skull.split(".nii")[0] + "BackHalf.nii.gz" #
    if not os.path.exists(staticfile_backhalf):
        fronthalf_backhalfpy(staticfile_skull)
    
    movingfile_backhalf=movingfile_skull.split(".nii")[0] + "BackHalf.nii.gz" #
    if not os.path.exists(movingfile_backhalf):
        fronthalf_backhalfpy(movingfile_skull)
        
#    slicenumbertomatch=int(sys.argv[3])
    staticfile=staticfile_backhalf
    movingfile=movingfile_backhalf
    niifilenametosave=staticfile.split(".nii")[0]+ "SLICENUM" + str(slicenumbertomatch) + ".nii.gz"
    if not os.path.exists(niifilenametosave):
        saveoneslicein2Dpy(staticfile,slicenumbertomatch,niifilenametosave)
#    gray_nifti_data=nib.load(movingfile).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
#    z_dim=nib.load(movi ngfile).header["pixdim"][3]
    z_dim_ud = 1 #int(z_dim/3) + 1
    for x in range(int(nib.load(movingfile).get_fdata().shape[2]/2),int(nib.load(movingfile).get_fdata().shape[2])):  #range(startingslice-z_dim_ud,startingslice+z_dim_ud): #range(int(nib.load(movingfile).get_fdata().shape[2]/3),int(nib.load(movingfile).get_fdata().shape[2]/2)): #range(startingslice-z_dim_ud,startingslice+z_dim_ud): #
        niifilenametosave_moving=movingfile.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
            moving_file_split=movingfile.split(".nii")
            reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
            ants.image_write(mywarpedimage,reg_output_filename1) 
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select
    
def matchback():
    seg_skullbone(sys.argv[1])
    seg_skullbone(sys.argv[2])
    staticfile=sys.argv[1].split(".nii")[0]+"_SKULL.nii.gz"
    movingfile=sys.argv[2].split(".nii")[0]+"_SKULL.nii.gz"
    fronthalf_backhalfpy(staticfile)
    fronthalf_backhalfpy(movingfile)
    staticfile=staticfile.split(".nii")[0] + "BackHalf.nii.gz" #

    movingfile=movingfile.split(".nii")[0] + "BackHalf.nii.gz" #

    slicenumbertomatch=int(sys.argv[3])
    niifilenametosave=staticfile.split(".nii")[0]+ "SLICENUM" + sys.argv[3] + ".nii.gz"
    saveoneslicein2Dpy(staticfile,slicenumbertomatch,niifilenametosave)
    gray_nifti_data=nib.load(movingfile).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
    for x in range(0,int(gray_nifti_data.shape[2]/2)):
        niifilenametosave_moving=staticfile.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
        ants.image_write(mywarpedimage,reg_output_filename1) 
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select
def midline_regmethod_oneachslice0200(static_file,moving_file,moving_file_intact):
    NECT_HET_filename=static_file[:-7]+ "BET.nii.gz"
    if "levelset" in os.path.basename(static_file):
        NECT_HET_filename=static_file[:-7]+ "_bet.nii.gz"

    BET_GRAY_FILE_SAVE=static_file.split(".nii")[0]+"BETgray.nii.gz"
    if not os.path.exists(BET_GRAY_FILE_SAVE):
        img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
        img_gray_bet=nib.load(NECT_HET_filename) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
        img_gray_data=img_gray.get_fdata()
        img_gray_bet_data=img_gray_bet.get_fdata()
        img_gray_data[img_gray_bet_data<1]=np.min(img_gray_data)
        array_img = nib.Nifti1Image(img_gray_data, affine=img_gray.affine, header=img_gray.header)
        nib.save(array_img,BET_GRAY_FILE_SAVE )
    pairs1=[]
    pairs2=[]
    file_gray=BET_GRAY_FILE_SAVE#static_file#moving_file_intact
    slope_of_lines= get_midline_pointswithregto664_2(BET_GRAY_FILE_SAVE,moving_file,moving_file_intact)
    pairs1.append(slope_of_lines[1])
    pairs2.append(slope_of_lines[0])             
    pair1=np.mean(pairs1,axis=0) #slope_of_lines[0] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
    pair2=np.mean(pairs2,axis=0) #slope_of_lines[1] #(pair2_1[1],pair2_1[4],pair2_1[5],pair2_1[0])
    filename_gray_data_np=nib.load(file_gray).get_fdata()
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetPoint1(pair1[1][1], pair1[1][0], pair1[3])
    planeSource.SetPoint2(pair1[2][1], pair1[2][0], pair1[3])
    planeSource.SetOrigin((pair2[1][1]+pair2[2][1])/2, (pair2[1][0]+pair2[2][0])/2, pair2[3])
    planeSource.Update()
    normal_p=planeSource.GetNormal()
    center_p=planeSource.GetCenter() #[(pair1[1][1]+pair1[2][1])/2, (pair1[1][0]+pair1[2][0])/2, pair1[3]] #planeSource.GetCenter(
    renderer = vtk.vtkRenderer()
    transformPD0, actor0= draw_plane_2((pair1[1][1], pair1[1][0], pair1[3]),(pair2[1][1], pair2[1][0], pair2[3]),(pair1[2][1], pair1[2][0], pair1[3]),renderer,scale_factor=5, N="Z")
    filename_gray_data_np=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    numpy_image=normalizeimage0to1(filename_gray_data_np)*255
    slice_points=[]
    for img_idx in range(numpy_image.shape[2]):
        transformPD1,actor1= image_plane_vtk_getplane(numpy_image[:,:,img_idx],img_idx, rgbFrame=None)
        act,intersection_line= cutter_polydata_v1(center_p,normal_p,transformPD1)
        points=np.array([[intersection_line.GetOutput().GetPoint(0)[1],intersection_line.GetOutput().GetPoint(0)[0]],[intersection_line.GetOutput().GetPoint(1)[1],intersection_line.GetOutput().GetPoint(1)[0]]])
        points_copy=np.copy(points)
        slice_points.append([img_idx,points_copy])
        print([img_idx,points_copy])
    return slice_points

def find_point_ofinterest():
    static_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Helsinki2000_2_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1.nii"
    moving_file_intact="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midlineWUSTL_664.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_711/WUSTL_711_08142018_2322_part2_Head_Spiral_3.0_J40s_2_20180815003926_2.nii"
    moving_file= "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
    files=glob.glob(os.path.join(os.path.dirname(static_file),"*.nii"))
    NECT_HET_filename=static_file[:-7]+ "BET.nii.gz"
    if "levelset" in os.path.basename(static_file):
        NECT_HET_filename=static_file[:-7]+ "_bet.nii.gz"
    img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(NECT_HET_filename) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    img_gray_data=img_gray.get_fdata()
    img_gray_bet_data=img_gray_bet.get_fdata()
    img_gray_data[img_gray_bet_data<1]=np.min(img_gray_data)
    img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
    img_gray_data=normalizeimage0to1(img_gray_data)*255
    for i in range(1):
        static_file1=files[i]
        slice_points=midline_regmethod_oneachslice0200(static_file1,moving_file,moving_file_intact)
    
#        slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_upper) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
#        show_slice_withaline(np.uint8(img[:,:,matched_image_id_upper]*255),np.array([pointA_upper,pointB_upper]))
#        slice_3_layer_lower, filtered_img_lower, binary_image_copy_lower,score_diff_from_1_lower,slope_of_lines_lower, pointA_lower, pointB_lower= find_falxline(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_lower) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
    
    pointA_lower,pointB_lower=lowerslice_points(np.uint8(img_gray_bet.get_fdata()[:,:,13]),np.uint8(img_gray.get_fdata()[:,:,13]),slice_points[13][1][0],slice_points[13][1][1],anglethreshold=10)
    show_slice_withaline(np.uint8(img_gray_bet_data[:,:,slice_points[13][0]]*255),np.array([pointB_lower,pointB_lower]))

#        show_slice_withaline(np.uint8(img[:,:,matched_image_id_lower]*255),np.array([pointA_lower,pointB_lower]))

    return "XX"
def call_pipeline_frontbackpoint():
    # find the slice matching back and front with point of interest
    static_template_image="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii"
    new_image1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/Helsinki2000_1/Helsinki2000_179_09262012_1256_Head_2.0_ax_Tilt_1_levelset.nii.gz" #Krak_125_10222015_1021_MOZG_3.0_H31s_do_3D_levelset.nii.gz" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Krak_094_04262015_0044_0_625mm_stnd.nii" #WUSTL_296_08162016_1150_Axial_Head.nii" #Helsinki2000_1655_2_09092016_1427_Head_4.0_e1.nii"# "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/Helsinki2000_5/Helsinki2000_1786_03252017_1430_Ax_Head_4.0_MPR_ax_Tilt_1_levelset.nii.gz" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/WUSTL_194_06122015_0608_Head_Spiral_3.0_J40s_2.nii" #Helsinki2000_1797_2_04012017_1601_Ax_Head_4.0_MPR_ax_e1_Tilt_1.nii" # Helsinki2000_2_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1.nii"
    
    new_image_dir=os.path.dirname(new_image1)
    five_files=glob.glob(os.path.join(new_image_dir,"*_levelset.nii.gz"))
    for x in five_files:
        new_image=x
        static_file= new_image 
    #    NECT_HET_filename="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/BET/Helsinki2000_1/Helsinki2000_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1_levelset_bet.nii.gz" #static_file[:-7]+ "BET.nii.gz"
        NECT_HET_filename=static_file[:-7]+ "BET.nii.gz"
        if "levelset" in os.path.basename(static_file):
            NECT_HET_filename=static_file[:-7]+ "_bet.nii.gz"
    
        static_image_path_jpg_lower="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/template_lower.jpg"
        static_image_path_jpg_upper="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/template_upper.jpg"
    
        matched_lower_slice=find_matching_lower_slice(new_image,NECT_HET_filename ,static_image_path_jpg_lower)
        matched_upper_slice=find_matching_upper_slice(new_image,NECT_HET_filename ,static_image_path_jpg_upper)
    
        backslicenumber=22
        frontslicenumber=21
        backslicenum_new=matchbackpy(static_template_image,new_image,backslicenumber,matched_lower_slice)
        frontslicenum_new=matchfrontpy(static_template_image,new_image,frontslicenumber,matched_lower_slice)  
        # Find the midline with the registration method
    #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Helsinki2000_2_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1.nii"
        moving_file_intact="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midlineWUSTL_664.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_711/WUSTL_711_08142018_2322_part2_Head_Spiral_3.0_J40s_2_20180815003926_2.nii"
        moving_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
    #    files=glob.glob(os.path.join(os.path.dirname(static_file),"*.nii"))
        img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
        img_gray_bet=nib.load(NECT_HET_filename) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
        img_gray_data=img_gray.get_fdata()
        img_gray_bet_data=img_gray_bet.get_fdata()
        img_gray_data[img_gray_bet_data<1]=np.min(img_gray_data)
        niigzfilenametosave=static_file[:-7]+ "BETgray.nii.gz"
        if not os.path.exists(niigzfilenametosave):
            array_img = nib.Nifti1Image(img_gray_data, affine=img_gray.affine, header=img_gray.header)
            nib.save(array_img, niigzfilenametosave)
            
        min_img_gray=np.min(img_gray_data)
    #    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
        if min_img_gray>=0:
            img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
        else:
            img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
        img_gray_data=normalizeimage0to1(img_gray_data)*255
    #    for i in range(1):
    #        static_file1=files[i]
        slice_points=midline_regmethod_oneachslice0200(static_file,moving_file,moving_file_intact)
        
    #        slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_upper) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
    #        show_slice_withaline(np.uint8(img[:,:,matched_image_id_upper]*255),np.array([pointA_upper,pointB_upper]))
    #        slice_3_layer_lower, filtered_img_lower, binary_image_copy_lower,score_diff_from_1_lower,slope_of_lines_lower, pointA_lower, pointB_lower= find_falxline(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_lower) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
        
        pointA_lower1,pointB_lower1=lowerslice_points(np.uint8(img_gray_bet.get_fdata()[:,:,frontslicenum_new]),np.uint8(img_gray.get_fdata()[:,:,frontslicenum_new]),slice_points[frontslicenum_new][1][0],slice_points[frontslicenum_new][1][1],anglethreshold=10)
        show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points[frontslicenum_new][0]]),np.array([pointB_lower1,pointB_lower1]))
    
        pointA_lower2,pointB_lower2=lowerslice_points(np.uint8(img_gray_bet.get_fdata()[:,:,backslicenum_new]),np.uint8(img_gray.get_fdata()[:,:,backslicenum_new]),slice_points[backslicenum_new][1][0],slice_points[backslicenum_new][1][1],anglethreshold=10)
        show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points[backslicenum_new][0]]),np.array([pointA_lower2,pointA_lower2]))
        
        slicenumbertomatch=41
        matched_top_slice=topslice_matching_refine_py(moving_file,niigzfilenametosave,slicenumbertomatch,matched_upper_slice)
        RESULT_DIR="./"
        filter_type='_Gabor'
#        slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_upper(static_file,NECT_HET_filename,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_top_slice) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
        slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_withREGNGabor_v1(static_file,NECT_HET_filename,(0,slice_points[matched_top_slice][1][0][0],slice_points[matched_top_slice][1][0][1],slice_points[matched_top_slice][1][1][0],slice_points[matched_top_slice][1][1][1]),RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_top_slice)


    #    show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))
        if len(pointA_upper)==0 or len(pointB_upper)==0:
            matched_top_slice=matched_upper_slice
            slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_withREGNGabor_v1(static_file,NECT_HET_filename,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_top_slice) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
            show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))
    
    
            
        point1=np.array([pointB_lower1,frontslicenum_new])
        point2=np.array([pointA_lower2,backslicenum_new])
        point3=np.array([pointB_upper,matched_top_slice])
        point4=np.array([pointA_upper,matched_top_slice])
    #    pair1=np.array([0,pointA_lower2,pointB_lower1])
    #    planeSource.SetPoint1(pair1[1][1], pair1[1][0], pair1[3])
        slice_points1=findcuttingplane(static_file,point1,point2,point3,point4)
        
        img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
        img_gray_data=img_gray.get_fdata()
        min_img_gray=np.min(img_gray_data)
    #    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
        if min_img_gray>=0:
            img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
        else:
            img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
        img_gray_data=normalizeimage0to1(img_gray_data)*255
        
        
        show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))
        
        lineThickness=2
        SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES"
        for x in range(len(slice_points1)) :
            show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points1[x][0]]),np.array([slice_points1[x][1][0],slice_points1[x][1][1]]))
            slice_3_layer= np.zeros([np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[0],np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[1],3])
            slice_3_layer[:,:,0]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
            slice_3_layer[:,:,1]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
            slice_3_layer[:,:,2]= np.uint8(img_gray_data[:,:,slice_points1[x][0]])# imgray1
            img_with_line1=cv2.line(slice_3_layer, (int(slice_points1[x][1][0][0]),int(slice_points1[x][1][0][1])), (int(slice_points1[x][1][1][0]),int(slice_points1[x][1][1][1])), (0,255,0), lineThickness)
            cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,os.path.basename(static_file).split(".nii")[0]+"_" +str(slice_points1[x][0]))+".png",img_with_line1)
 
        
def call_pipeline_frontbackpoint_sh():
    # find the slice matching back and front with point of interest
    static_template_image="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii"
    new_image=sys.argv[1] # 1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/Helsinki2000_1/Helsinki2000_179_09262012_1256_Head_2.0_ax_Tilt_1_levelset.nii.gz" #Krak_125_10222015_1021_MOZG_3.0_H31s_do_3D_levelset.nii.gz" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Krak_094_04262015_0044_0_625mm_stnd.nii" #WUSTL_296_08162016_1150_Axial_Head.nii" #Helsinki2000_1655_2_09092016_1427_Head_4.0_e1.nii"# "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/Helsinki2000_5/Helsinki2000_1786_03252017_1430_Ax_Head_4.0_MPR_ax_Tilt_1_levelset.nii.gz" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/WUSTL_194_06122015_0608_Head_Spiral_3.0_J40s_2.nii" #Helsinki2000_1797_2_04012017_1601_Ax_Head_4.0_MPR_ax_e1_Tilt_1.nii" # Helsinki2000_2_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1.nii"
    
#    new_image_dir=os.path.dirname(new_image1)
#    five_files=glob.glob(os.path.join(new_image_dir,"*_levelset.nii.gz"))
#    for x in five_files:
#        new_image=x
    static_file= new_image 
#    NECT_HET_filename="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/BET/Helsinki2000_1/Helsinki2000_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1_levelset_bet.nii.gz" #static_file[:-7]+ "BET.nii.gz"
    NECT_HET_filename=static_file[:-7]+ "BET.nii.gz"
    if "levelset" in os.path.basename(static_file):
        NECT_HET_filename=static_file[:-7]+ "_bet.nii.gz"

    static_image_path_jpg_lower="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/template_lower.jpg"
    static_image_path_jpg_upper="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/template_upper.jpg"

    matched_lower_slice=find_matching_lower_slice(new_image,NECT_HET_filename ,static_image_path_jpg_lower)
    matched_upper_slice=find_matching_upper_slice(new_image,NECT_HET_filename ,static_image_path_jpg_upper)

    backslicenumber=22
    frontslicenumber=21
    backslicenum_new=matchbackpy(static_template_image,new_image,backslicenumber,matched_lower_slice)
    frontslicenum_new=matchfrontpy(static_template_image,new_image,frontslicenumber,matched_lower_slice)  
    # Find the midline with the registration method
#"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Helsinki2000_2_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1.nii"
    moving_file_intact="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midlineWUSTL_664.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_711/WUSTL_711_08142018_2322_part2_Head_Spiral_3.0_J40s_2_20180815003926_2.nii"
    moving_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
#    files=glob.glob(os.path.join(os.path.dirname(static_file),"*.nii"))
    img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_bet=nib.load(NECT_HET_filename) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
    img_gray_data=img_gray.get_fdata()
    img_gray_bet_data=img_gray_bet.get_fdata()
    img_gray_data[img_gray_bet_data<1]=np.min(img_gray_data)
    niigzfilenametosave=static_file[:-7]+ "BETgray.nii.gz"
    if not os.path.exists(niigzfilenametosave):
        array_img = nib.Nifti1Image(img_gray_data, affine=img_gray.affine, header=img_gray.header)
        nib.save(array_img, niigzfilenametosave)
        
    min_img_gray=np.min(img_gray_data)
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
    img_gray_data=normalizeimage0to1(img_gray_data)*255
#    for i in range(1):
#        static_file1=files[i]
    slice_points=midline_regmethod_oneachslice0200(static_file,moving_file,moving_file_intact)
    
#        slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_upper) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
#        show_slice_withaline(np.uint8(img[:,:,matched_image_id_upper]*255),np.array([pointA_upper,pointB_upper]))
#        slice_3_layer_lower, filtered_img_lower, binary_image_copy_lower,score_diff_from_1_lower,slope_of_lines_lower, pointA_lower, pointB_lower= find_falxline(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_image_id_lower) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
    
    pointA_lower1,pointB_lower1=lowerslice_points(np.uint8(img_gray_bet.get_fdata()[:,:,frontslicenum_new]),np.uint8(img_gray.get_fdata()[:,:,frontslicenum_new]),slice_points[frontslicenum_new][1][0],slice_points[frontslicenum_new][1][1],anglethreshold=10)
    show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points[frontslicenum_new][0]]),np.array([pointB_lower1,pointB_lower1]))

    pointA_lower2,pointB_lower2=lowerslice_points(np.uint8(img_gray_bet.get_fdata()[:,:,backslicenum_new]),np.uint8(img_gray.get_fdata()[:,:,backslicenum_new]),slice_points[backslicenum_new][1][0],slice_points[backslicenum_new][1][1],anglethreshold=10)
    show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points[backslicenum_new][0]]),np.array([pointA_lower2,pointA_lower2]))
    
    slicenumbertomatch=41
    matched_top_slice=topslice_matching_refine_py(moving_file,niigzfilenametosave,slicenumbertomatch,matched_upper_slice)
    RESULT_DIR="./"
    filter_type='_Gabor'
#    slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_upper(static_file,NECT_HET_filename,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_top_slice) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
    slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_withREGNGabor_v1(static_file,NECT_HET_filename,(0,slice_points[matched_top_slice][1][0][0],slice_points[matched_top_slice][1][0][1],slice_points[matched_top_slice][1][1][0],slice_points[matched_top_slice][1][1][1]),RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_top_slice)

#    show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))
    if len(pointA_upper)==0 or len(pointB_upper)==0:
        matched_top_slice=matched_upper_slice
        slice_3_layer_upper, filtered_img_upper, binary_image_copy_upper,score_diff_from_1_upper,slope_of_lines_upper, pointA_upper, pointB_upper= find_falxline_withREGNGabor_v1(static_file,NECT_HET_filename,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber=matched_top_slice) #(file_gray,file_gray_bet,RESULT_DIR=RESULT_DIR,filter_type=filter_type,slicenumber)
        show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))


        
    point1=np.array([pointB_lower1,frontslicenum_new])
    point2=np.array([pointA_lower2,backslicenum_new])
    point3=np.array([pointB_upper,matched_top_slice])
    point4=np.array([pointA_upper,matched_top_slice])
#    pair1=np.array([0,pointA_lower2,pointB_lower1])
#    planeSource.SetPoint1(pair1[1][1], pair1[1][0], pair1[3])
    slice_points1=findcuttingplane(static_file,point1,point2,point3,point4)
    
    img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_data=img_gray.get_fdata()
    min_img_gray=np.min(img_gray_data)
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
    img_gray_data=normalizeimage0to1(img_gray_data)*255
    
    
    show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))
    
    lineThickness=2
    SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES"
    for x in range(len(slice_points1)) :
        show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points1[x][0]]),np.array([slice_points1[x][1][0],slice_points1[x][1][1]]))
        slice_3_layer= np.zeros([np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[0],np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[1],3])
        slice_3_layer[:,:,0]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
        slice_3_layer[:,:,1]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
        slice_3_layer[:,:,2]= np.uint8(img_gray_data[:,:,slice_points1[x][0]])# imgray1
        img_with_line1=cv2.line(slice_3_layer, (int(slice_points1[x][1][0][0]),int(slice_points1[x][1][0][1])), (int(slice_points1[x][1][1][0]),int(slice_points1[x][1][1][1])), (0,255,0), lineThickness)
        cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,os.path.basename(static_file).split(".nii")[0]+"_" +str(slice_points1[x][0]))+".png",img_with_line1)
   
    


            
def lowerslice_points(image_gray_bet,image_gray,pointA,pointB,anglethreshold=10,RESULT_DIR="",filter_type="",slicenumber=1):

    imgray=image_gray_bet
    imgray1=image_gray 
    show_slice(imgray)
#    ret, thresh = cv2.threshold(imgray, 127, 255,0)
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slice_3_layer= np.zeros([imgray1.shape[0],imgray.shape[1],3])
    slice_3_layer[:,:,0]=imgray1
    slice_3_layer[:,:,1]=imgray1
    slice_3_layer[:,:,2]=imgray1
    slice_3_layer=np.uint8(slice_3_layer)
#    cv2.drawContours(slice_3_layer, contours, -1, (0,255,0), 3)
#    pointA=(int(256.0),int(0.0))
#    pointB=(int(256.0),int(512.0))
    slice_3_layer=cv2.line(slice_3_layer, ( int(pointA[0]),int(pointA[1])),(int(pointB[0]),int(pointB[1])), (255,255,0), 2)

    # mid point of the line:
    pointAnp=np.array(pointA)
    pointBnp=np.array(pointB)
    mean_point_contour=[]
    contourNitslength=[]
    for each_contour in contours:
#        print(len(each_contour))
        each_contourNitslenght=[each_contour,len(each_contour)]
        contourNitslength.append(each_contourNitslenght)
        mean_point_contour.append(np.mean(each_contour,axis=0)[0])
    contourNitslength= np.array(contourNitslength)
    print(contourNitslength)
    largest_contour=contourNitslength[np.argmax(contourNitslength[:,1]),0]
    mean_point_contournp_mean=np.mean(np.array(largest_contour),axis=0)
    slope, perp_slope, per_bisec_A,per_bisec_B=perpendicular_throughapoint(pointAnp,pointBnp,mean_point_contournp_mean[0])
    ## separate the points into upper and lower half:
    for each_point in largest_contour:
        cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
            
    upperhalf=[]
    lowerhalf=[]
    for each_point in largest_contour:
        xx=whichsideofline(per_bisec_A,per_bisec_B,each_point[0])
        if xx >0:
            upperhalf.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,0),1)
        if xx <0:
            lowerhalf.append(each_point[0])
#            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
            

    upperhalf_20deg=[]
    lowerhalf_20deg=[]
    upperhalf_20degBP=[]
    lowerhalf_20degBP=[]
    upperhalf_20degPx=[] 
    lowerhalf_20degPx=[] 
#    show_slice_withaline(imgray1,np.array([pointAnp,pointBnp]))
#    show_slice_withaline(imgray1,np.array([per_bisec_A,per_bisec_B]))
    
    for each_point in upperhalf:
        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,255),1)
    for each_point in lowerhalf:
        cv2.circle(slice_3_layer,(int(each_point[0]),int(each_point[1])),2,(255,0,0),1)
    for each_point in largest_contour:
    #    print(each_point[0])
        v1= np.array(pointBnp) - mean_point_contournp_mean[0]
        v2= each_point[0] - mean_point_contournp_mean[0]
        angle1=np.abs(angle_bet_two_vector(v1,v2))
        print(angle1)
        if angle1<=anglethreshold:
            line1=np.array([per_bisec_A,per_bisec_B])
            line2=np.array([pointA,pointB])
            P= project_point_online(each_point[0],line1)
            Px= project_point_online(each_point[0],line2)
            upperhalf_20deg.append(P)
            upperhalf_20degPx.append(Px)
            upperhalf_20degBP.append(each_point[0])
            cv2.circle(slice_3_layer,(int(P[0]),int(P[1])),2,(255,0,255),1)
            cv2.circle(slice_3_layer,(int(Px[0]),int(Px[1])),2,(255,0,255),1)
            cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,255),1)
        if angle1>90:
            angle1=np.abs(180-angle1)
            if  angle1<=anglethreshold:
                line1=np.array([per_bisec_A,per_bisec_B])
                line2=np.array([pointA,pointB])
                P= project_point_online(each_point[0],line1)
                Px= project_point_online(each_point[0],line2)
                lowerhalf_20deg.append(P)
                lowerhalf_20degPx.append(Px)
                lowerhalf_20degBP.append(each_point[0])
                cv2.circle(slice_3_layer,(int(P[0]),int(P[1])),2,(255,0,0),1)
                cv2.circle(slice_3_layer,(int(Px[0]),int(Px[1])),2,(255,0,0),1)
                cv2.circle(slice_3_layer,(int(each_point[0][0]),int(each_point[0][1])),2,(255,0,0),1)




    upperhalf_20deg=np.array(upperhalf_20deg)
    upperhalf_20degPx=np.array(upperhalf_20degPx)
    upperhalf_20degBP=np.array(upperhalf_20degBP)
    
    lowerhalf_20deg=np.array(lowerhalf_20deg)
    lowerhalf_20degPx=np.array(lowerhalf_20degPx)
    lowerhalf_20degBP=np.array(lowerhalf_20degBP)
 
    upperhalf_20deg_dist=np.sqrt(np.sum(np.square(upperhalf_20degBP-mean_point_contournp_mean[0]),axis=1))
    lowerhalf_20deg_dist=np.sqrt(np.sum(np.square(lowerhalf_20degBP-mean_point_contournp_mean[0]),axis=1))
   
    upperhalf_20deg_dist_min_id= np.argmin(upperhalf_20deg_dist)
    lowerhalf_20deg_dist_min_id= np.argmin(lowerhalf_20deg_dist)
    midline_upper_point= upperhalf_20degBP[upperhalf_20deg_dist_min_id]
    midline_lower_point= lowerhalf_20degBP[lowerhalf_20deg_dist_min_id]
    
    slice_3_layer=cv2.line(slice_3_layer,( int(per_bisec_A[0]),int(per_bisec_A[1])),(int(per_bisec_B[0]),int(per_bisec_B[1])) , (255,255,0), 2)
    
    slice_3_layer=cv2.line(slice_3_layer,( int(midline_upper_point[0]),int(midline_upper_point[1])),(int(midline_lower_point[0]),int(midline_lower_point[1])) , (255,0,0), 5)

    PointA=( int(midline_upper_point[0]),int(midline_upper_point[1]))
    PointB=(int(midline_lower_point[0]),int(midline_lower_point[1]))
    PointAnp=np.array(PointA)
    PointBnp=np.array(PointB)
    if PointAnp.size==0 or PointBnp.size==0:
        PointA=( int(pointA[0]),int(pointA[1]))
        PointB=(int(pointB[0]),int(pointB[1]))

    return  PointA,PointB

def find_matching_lower_slice(moving_image_path_nifti,gray_bet_image_filepath ,static_image_path_jpg):
    img_gray_bet=nib.load(gray_bet_image_filepath) 
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    min_img_gray=np.min(img_gray.get_fdata())
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
        
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    show_slice(img[:,:,int(img.shape[2]/2)])
    features_template_lower = image_features([static_image_path_jpg]) 
    min_diff=9999999
    matched_image_id_lower=0
    diff_array_lower=[]          
    for x in range(img.shape[2]):
        if x< img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_lower-features_test)*(features_template_lower-features_test))/features_template_lower.shape[1])
            diff_array_lower.append(feature_diff)
#            print("feature_diff")
#            print(feature_diff)
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_lower=x
    return matched_image_id_lower

def find_matching_lower_slice_withstack(moving_image_path_nifti,gray_bet_image_filepath ,static_image_path_jpg):
    img_gray_bet=nib.load(gray_bet_image_filepath) 
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
#    min_img_gray=np.min(img_gray.get_fdata())
    img=contrast_stretch(img_gray.get_fdata(),1)
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
#    if min_img_gray>=0:
#        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
        
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
    show_slice(img[:,:,int(img.shape[2]/2)])
    features_template_lower = image_features(static_image_path_jpg) 
    min_diff=9999999
    matched_image_id_lower=0
    diff_array_lower=[]          
    for x in range(img.shape[2]):
        if x< img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_lower-features_test)*(features_template_lower-features_test))/features_template_lower.shape[1])
            diff_array_lower.append(feature_diff)
#            print("feature_diff")
#            print(feature_diff)
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_lower=x
    return matched_image_id_lower

def find_matching_upper_slice(moving_image_path_nifti,gray_bet_image_filepath ,static_image_path_jpg):
    img_gray_bet=nib.load(gray_bet_image_filepath) 
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
#    show_slice(img[:,:,int(img.shape[2]/2)])
    features_template_upper = image_features([static_image_path_jpg]) 
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]          
    for x in range(img.shape[2]):
        if x > img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
            diff_array.append(feature_diff)
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_upper=x
    return matched_image_id_upper

def find_matching_upper_slice_withstack(moving_image_path_nifti,gray_bet_image_filepath ,static_image_path_jpg):
    img_gray_bet=nib.load(gray_bet_image_filepath) 
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
#    min_img_gray=np.min(img_gray.get_fdata())
#    if min_img_gray>=0:
#        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    img=contrast_stretch( img_gray.get_fdata() ,1)
    img=img*normalizeimage0to1(img_gray_bet.get_fdata())
#    show_slice(img[:,:,int(img.shape[2]/2)])
    features_template_upper = image_features(static_image_path_jpg) 
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]          
    for x in range(img.shape[2]):
        if x > img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
            diff_array.append(feature_diff)
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_upper=x
    return matched_image_id_upper
def find_matching_eye_slice(moving_image_path_nifti ,static_image_path_jpg):

    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    
    features_template_upper = image_features([static_image_path_jpg]) 
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]          
    for x in range(img.shape[2]):
        if x < img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
            diff_array.append(feature_diff)
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_upper=x
    return matched_image_id_upper

def find_matching_lower_slice_v1(moving_image_path_nifti ,static_image_path_jpg):

    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    
    features_template_upper = image_features([static_image_path_jpg]) 
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]          
    for x in range(img.shape[2]):
        if x < img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
            diff_array.append(feature_diff)
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_upper=x
    return matched_image_id_upper

def find_matching_amongallslices(moving_image_path_nifti ,static_image_path_jpg):

    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    
    features_template_upper = image_features([static_image_path_jpg]) 
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]          
    for x in range(img.shape[2]):
#        if x < img.shape[2]*0.5:
        cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
        features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
        feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
        diff_array.append(feature_diff)
        if min_diff> feature_diff:
            min_diff=feature_diff
            matched_image_id_upper=x
    return matched_image_id_upper


def find_matching_upper_slice_with10average(moving_image_path_nifti,features_template_upper):
    file_slice=pd.read_csv("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CLASSIFYEACHSLICE/CTFileForSliceMatching.csv")
#    errors=[]
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]
    for x in range(img.shape[2]):
        if x > img.shape[2]*0.6:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])

            feature_diff_sum=0
            feature_diffs=[]
#            templatefiles=[]
#            for y in range(file_slice.shape[0]):
#                fixed_file=os.path.join(file_slice['DirectoryName'][y],file_slice['filename'][y]+file_slice['FileExtension'][y])
#                upper_slice=file_slice['upper_slice_num'][y]-1
#
#                static_image_path_jpg=os.path.join(os.path.dirname(fixed_file),os.path.basename(fixed_file).split(".nii")[0]+ str(upper_slice)+ ".jpg")
#                templatefiles.append(static_image_path_jpg)
#            features_template_upper = image_features(templatefiles,augment=True) 
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
#                feature_diff_sum=feature_diff_sum+feature_diff
            feature_diffs.append(feature_diff)
#        feature_diffs=np.array(feature_diffs)
            feature_diff_sum=np.mean(feature_diffs)
            diff_array.append([x,feature_diff_sum])
            if min_diff> feature_diff_sum:
                min_diff=feature_diff_sum
                matched_image_id_upper=x
                print(min_diff)
    #    errors.append(min_diff)
        
        
    return matched_image_id_upper,diff_array
def find_matching_upper_slice_with10averagewithaug(moving_image_path_nifti,features_template_upper):
    file_slice=pd.read_csv("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CLASSIFYEACHSLICE/CTFileForSliceMatching.csv")
#    errors=[]
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]
    for x in range(img.shape[2]):
        if x > img.shape[2]*0.6:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'],augment=True)

            feature_diff_sum=0
            feature_diffs=[]
#            templatefiles=[]
#            for y in range(file_slice.shape[0]):
#                fixed_file=os.path.join(file_slice['DirectoryName'][y],file_slice['filename'][y]+file_slice['FileExtension'][y])
#                upper_slice=file_slice['upper_slice_num'][y]-1
#
#                static_image_path_jpg=os.path.join(os.path.dirname(fixed_file),os.path.basename(fixed_file).split(".nii")[0]+ str(upper_slice)+ ".jpg")
#                templatefiles.append(static_image_path_jpg)
#            features_template_upper = image_features(templatefiles,augment=True) 
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
#                feature_diff_sum=feature_diff_sum+feature_diff
            feature_diffs.append(feature_diff)
#        feature_diffs=np.array(feature_diffs)
            feature_diff_sum=np.mean(feature_diffs)
            diff_array.append([x,feature_diff_sum])
            if min_diff> feature_diff_sum:
                min_diff=feature_diff_sum
                matched_image_id_upper=x
                print(min_diff)
    #    errors.append(min_diff)
        
        
    return matched_image_id_upper,diff_array

def find_matching_lower_slice_with10average(moving_image_path_nifti,features_template_lower):
    file_slice=pd.read_csv("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CLASSIFYEACHSLICE/CTFileForSliceMatching.csv")
#    errors=[]
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]
    for x in range(img.shape[2]):
        if x < img.shape[2]*0.45:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])

            feature_diff_sum=0
            feature_diffs=[]
#            templatefiles=[]
#            for y in range(file_slice.shape[0]):
#                fixed_file=os.path.join(file_slice['DirectoryName'][y],file_slice['filename'][y]+file_slice['FileExtension'][y])
#                upper_slice=file_slice['lower_slice_num'][y]-1
#
#                static_image_path_jpg=os.path.join(os.path.dirname(fixed_file),os.path.basename(fixed_file).split(".nii")[0]+ str(upper_slice)+ ".jpg")
#                templatefiles.append(static_image_path_jpg)
#            features_template_upper = image_features(templatefiles,augment=True) 
            feature_diff=np.sqrt(np.sum((features_template_lower-features_test)*(features_template_lower-features_test))/features_template_lower.shape[1])
#                feature_diff_sum=feature_diff_sum+feature_diff
            feature_diffs.append(feature_diff)
#        feature_diffs=np.array(feature_diffs)
            feature_diff_sum=np.mean(feature_diffs)
            diff_array.append([x,feature_diff_sum])
            if min_diff> feature_diff_sum:
                min_diff=feature_diff_sum
                matched_image_id_upper=x
                print(min_diff)
    #    errors.append(min_diff)
        
        
    return matched_image_id_upper,diff_array
def find_matching_lower_slice_with10averagewithaug(moving_image_path_nifti,features_template_lower):
    file_slice=pd.read_csv("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CLASSIFYEACHSLICE/CTFileForSliceMatching.csv")
#    errors=[]
    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]
    for x in range(img.shape[2]):
        if x < img.shape[2]*0.45:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'],augment=True)

            feature_diff_sum=0
            feature_diffs=[]
#            templatefiles=[]
#            for y in range(file_slice.shape[0]):
#                fixed_file=os.path.join(file_slice['DirectoryName'][y],file_slice['filename'][y]+file_slice['FileExtension'][y])
#                upper_slice=file_slice['lower_slice_num'][y]-1
#
#                static_image_path_jpg=os.path.join(os.path.dirname(fixed_file),os.path.basename(fixed_file).split(".nii")[0]+ str(upper_slice)+ ".jpg")
#                templatefiles.append(static_image_path_jpg)
#            features_template_upper = image_features(templatefiles,augment=True) 
            feature_diff=np.sqrt(np.sum((features_template_lower-features_test)*(features_template_lower-features_test))/features_template_lower.shape[1])
#                feature_diff_sum=feature_diff_sum+feature_diff
            feature_diffs.append(feature_diff)
#        feature_diffs=np.array(feature_diffs)
            feature_diff_sum=np.mean(feature_diffs)
            diff_array.append([x,feature_diff_sum])
            if min_diff> feature_diff_sum:
                min_diff=feature_diff_sum
                matched_image_id_upper=x
                print(min_diff)
    #    errors.append(min_diff)
        
        
    return matched_image_id_upper,diff_array

#def find_matching_lower_slice_with10average(moving_image_path_nifti):
#    file_slice=pd.read_csv("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CLASSIFYEACHSLICE/CTFileForSliceMatching.csv")
##    errors=[]
#    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
#    min_img_gray=np.min(img_gray.get_fdata())
#    if min_img_gray>=0:
#        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
#    min_diff=9999999
#    matched_image_id_upper=0
#    diff_array=[]
#    for x in range(img.shape[2]):
#        if x < img.shape[2]*0.45:
#            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
#            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
#
#            feature_diff_sum=0
#            feature_diffs=[]
#            for y in range(file_slice.shape[0]):
#                fixed_file=os.path.join(file_slice['DirectoryName'][y],file_slice['filename'][y]+file_slice['FileExtension'][y])
#                upper_slice=file_slice['lower_slice_num'][y]-1
#                static_image_path_jpg=os.path.join(os.path.dirname(fixed_file),os.path.basename(fixed_file).split(".nii")[0]+ str(upper_slice)+ ".jpg")
#                features_template_upper = image_features([static_image_path_jpg]) 
#                feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
##                feature_diff_sum=feature_diff_sum+feature_diff
#                feature_diffs.append(feature_diff)
##        feature_diffs=np.array(feature_diffs)
#            feature_diff_sum=np.mean(feature_diffs)
#            diff_array.append([x,feature_diff_sum])
#            if min_diff> feature_diff_sum:
#                min_diff=feature_diff_sum
#                matched_image_id_upper=x
#                print(min_diff)
#    #    errors.append(min_diff)
#        
#        
#    return matched_image_id_upper,diff_array

def find_matching_upper_slice_v1(moving_image_path_nifti ,static_image_path_jpg):

    img_gray=nib.load(moving_image_path_nifti) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    min_img_gray=np.min(img_gray.get_fdata())
    if min_img_gray>=0:
        img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(1000, 1200))
    
    features_template_upper = image_features([static_image_path_jpg]) 
    min_diff=9999999
    matched_image_id_upper=0
    diff_array=[]          
    for x in range(img.shape[2]):
        if x > img.shape[2]*0.5:
            cv2.imwrite(moving_image_path_nifti.split(".nii")[0]+'test.jpg',img[:,:,x]*255)
            features_test = image_features([moving_image_path_nifti.split(".nii")[0]+'test.jpg'])
            feature_diff=np.sqrt(np.sum((features_template_upper-features_test)*(features_template_upper-features_test))/features_template_upper.shape[1])
            diff_array.append([x,feature_diff])
            if min_diff> feature_diff:
                min_diff=feature_diff
                matched_image_id_upper=x
    return matched_image_id_upper,diff_array

def findcuttingplane(file_gray,point1,point2,point3,point4,SAVE_IMAGE=False,SAVE_DIRECTORY="./"):
#    NECT_HET_filename=static_file[:-7]+ "BET.nii.gz"
#    img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
#    img_gray_bet=nib.load(NECT_HET_filename) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
#    img_gray_data=img_gray.get_fdata()
#    img_gray_bet_data=img_gray_bet.get_fdata()
#    img_gray_data[img_gray_bet_data<1]=np.min(img_gray_data)
#    array_img = nib.Nifti1Image(img_gray_data, affine=img_gray.affine, header=img_gray.header)
#    BET_GRAY_FILE_SAVE=static_file.split(".nii")[0]+"BETgray.nii.gz"
#    nib.save(array_img,BET_GRAY_FILE_SAVE )
#    pairs1=[]
#    pairs2=[]
#    file_gray=BET_GRAY_FILE_SAVE#static_file#moving_file_intact
#    slope_of_lines= get_midline_pointswithregto664_2(BET_GRAY_FILE_SAVE,moving_file,moving_file_intact)
#    pairs1.append(slope_of_lines[1])
#    pairs2.append(slope_of_lines[0])             
#    pair1=np.mean(pairs1,axis=0) #slope_of_lines[0] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
#    pair2=np.mean(pairs2,axis=0) #slope_of_lines[1] #(pair2_1[1],pair2_1[4],pair2_1[5],pair2_1[0])
    filename_gray_data_np=nib.load(file_gray).get_fdata()
    planeSource = vtk.vtkPlaneSource()
    planeSource.SetPoint1(point1[0][1], point1[0][0], point1[1])
    planeSource.SetPoint2(point2[0][1], point2[0][0], point2[1])
    planeSource.SetOrigin((point3[0][1]+point4[0][1])/2, (point3[0][0]+point4[0][0])/2, point3[1])
    planeSource.Update()
    normal_p=planeSource.GetNormal()
    center_p=planeSource.GetCenter() #[(pair1[1][1]+pair1[2][1])/2, (pair1[1][0]+pair1[2][0])/2, pair1[3]] #planeSource.GetCenter(
#    renderer = vtk.vtkRenderer()
#    transformPD0, actor0= draw_plane_2((pair1[1][1], pair1[1][0], pair1[3]),(pair2[1][1], pair2[1][0], pair2[3]),(pair1[2][1], pair1[2][0], pair1[3]),renderer,scale_factor=5, N="Z")
    filename_gray_data_np=exposure.rescale_intensity( filename_gray_data_np , in_range=(0, 200))
    numpy_image=normalizeimage0to1(filename_gray_data_np)*255
    slice_points=[]
    for img_idx in range(numpy_image.shape[2]):
        transformPD1,actor1= image_plane_vtk_getplane(numpy_image[:,:,img_idx],img_idx, rgbFrame=None)
        act,intersection_line= cutter_polydata_v1(center_p,normal_p,transformPD1)
        points=np.array([[intersection_line.GetOutput().GetPoint(0)[1],intersection_line.GetOutput().GetPoint(0)[0]],[intersection_line.GetOutput().GetPoint(1)[1],intersection_line.GetOutput().GetPoint(1)[0]]])
        points_copy=np.copy(points)
        slice_points.append([img_idx,points_copy])
        
#        print([img_idx,points_copy])
    
    if SAVE_IMAGE==True:
        img_gray_data= numpy_image #normalizeimage0to1(img_gray_data)*255
    # make cutting plane and save images:
        slice_points1=slice_points
        lineThickness=2
        static_file=file_gray
        SLICE_OUTPUT_DIRECTORY= SAVE_DIRECTORY #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES"
        for x in range(len(slice_points1)) :
            show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points1[x][0]]),np.array([slice_points1[x][1][0],slice_points1[x][1][1]]))
            slice_3_layer= np.zeros([np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[0],np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[1],3])
            slice_3_layer[:,:,0]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
            slice_3_layer[:,:,1]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
            slice_3_layer[:,:,2]= np.uint8(img_gray_data[:,:,slice_points1[x][0]])# imgray1
            img_with_line1=cv2.line(slice_3_layer, (int(slice_points1[x][1][0][0]),int(slice_points1[x][1][0][1])), (int(slice_points1[x][1][1][0]),int(slice_points1[x][1][1][1])), (0,255,0), lineThickness)
            imagefilename=re.sub('[^A-Za-z0-9]+', '',os.path.basename(static_file).split(".nii")[0]+"_" +str(slice_points1[x][0]))
            cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,imagefilename +".png"),img_with_line1)
    return slice_points

def create_BETgray(static_file):
    NECT_HET_filename=static_file[:-7]+ "BET.nii.gz"
    if "levelset" in os.path.basename(static_file):
        NECT_HET_filename=static_file[:-7]+ "_bet.nii.gz"
    
    filexists="NOT CREATED. FILE EXIST"
    BET_GRAY_FILE_SAVE=static_file.split(".nii")[0]+"BETgray.nii.gz"
    if not os.path.exists(BET_GRAY_FILE_SAVE):
        img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
        img_gray_bet=nib.load(NECT_HET_filename) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+"_bet.nii.gz"))
        img_gray_data=img_gray.get_fdata()
        img_gray_bet_data=img_gray_bet.get_fdata()
        img_gray_data[img_gray_bet_data<1]=np.min(img_gray_data)
        array_img = nib.Nifti1Image(img_gray_data, affine=img_gray.affine, header=img_gray.header)
        nib.save(array_img,BET_GRAY_FILE_SAVE )
        filexists="FILE CREATED"
    return filexists,BET_GRAY_FILE_SAVE

def midline_2D_with_reg(template_graybet,template_mildine,target_graybet,slicenumber_target,slicenumber_template):
    template_mildine="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/IDEALML/midlineWUSTL_664.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_711/WUSTL_711_08142018_2322_part2_Head_Spiral_3.0_J40s_2_20180815003926_2.nii"
    template_graybet="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/NECT/BASELINE/WUSTL_664/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz"
#    target_graybet="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_2018050709413BETgray.nii.gz" #3_2.nii"
    target_graybet1="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Helsinki2000_1655_2_09092016_1427_Head_4.0_e1.nii"#"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/Helsinki2000_1/Helsinki2000_3_04282012_1438_Paa_natiivi_5.0_levelset.nii.gz" #Krak_125_10222015_1021_MOZG_3.0_H31s_do_3D_levelset.nii.gz" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/Krak_094_04262015_0044_0_625mm_stnd.nii" #WUSTL_296_08162016_1150_Axial_Head.nii" #Helsinki2000_1655_2_09092016_1427_Head_4.0_e1.nii"# "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/NECT/Helsinki2000_5/Helsinki2000_1786_03252017_1430_Ax_Head_4.0_MPR_ax_Tilt_1_levelset.nii.gz" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/NetWaterUptake/DATA/Hel_Krak_Wustl/WUSTL_194_06122015_0608_Head_Spiral_3.0_J40s_2.nii" #Helsinki2000_1797_2_04012017_1601_Ax_Head_4.0_MPR_ax_e1_Tilt_1.nii" # Helsinki2000_2_2_04292012_1039_Head_4.0_J45_Safir2_Tilt_1.nii"
    filexists,BET_GRAY_FILE_SAVE = create_BETgray(target_graybet1)
    target_graybet=BET_GRAY_FILE_SAVE
    slicenumber_target=24
    slicenumber_target_1=24
    slicenumber_template=25 #41
    niifilenametosave_moving=template_graybet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
    saveoneslicein2Dpy(template_graybet,slicenumber_template,niifilenametosave_moving)
    niifilenametosave_movingmidline=template_mildine.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
    saveoneslicein2Dpy(template_mildine,slicenumber_template,niifilenametosave_movingmidline)
    niifilenametosave_fixed=target_graybet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
    saveoneslicein2Dpy(target_graybet,slicenumber_target,niifilenametosave_fixed)
#    fixed=niifilenametosave_fixed
#    moving=niifilenametosave_moving
#    mytx,fixed=register_twoimages_test(fixed,moving,type_of_transform= 'Affine')
#    mywarpedimage=apply_transformation_1(fixed, moving,mytx)    
    slope_of_lines=     get_midline_pointswithregto664_2(niifilenametosave_fixed,niifilenametosave_moving,niifilenametosave_movingmidline)
#get_midline_pointswithregto664_2(BET_GRAY_FILE_SAVE,moving_file,moving_file_intact)
    pairs1=[]
    pairs2=[]
    pairs1.append(slope_of_lines[1])
    pairs2.append(slope_of_lines[0])             
    pair1=np.mean(pairs1,axis=0) #slope_of_lines[0] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
    pair2=np.mean(pairs2,axis=0) #slope_of_lines[1] #(pair2_1[1],pair2_1[4],pair2_1[5],pair2_1[0])
    
    img_gray=nib.load(niifilenametosave_fixed) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_data=img_gray.get_fdata()
    min_img_gray=np.min(img_gray_data)
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
    img_gray_data=normalizeimage0to1(img_gray_data)*255
    show_slice_withaline(np.uint8(img_gray_data),np.array([pair1[1],pair1[2]]))
    ##############################################################################################
    slicenumber_target=36
    slicenumber_target_2=36
    slicenumber_template=41
    niifilenametosave_moving=template_graybet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
    saveoneslicein2Dpy(template_graybet,slicenumber_template,niifilenametosave_moving)
    niifilenametosave_movingmidline=template_mildine.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
    saveoneslicein2Dpy(template_mildine,slicenumber_template,niifilenametosave_movingmidline)
    niifilenametosave_fixed=target_graybet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
    saveoneslicein2Dpy(target_graybet,slicenumber_target,niifilenametosave_fixed)
#    fixed=niifilenametosave_fixed
#    moving=niifilenametosave_moving
#    mytx,fixed=register_twoimages_test(fixed,moving,type_of_transform= 'Affine')
#    mywarpedimage=apply_transformation_1(fixed, moving,mytx)    
    slope_of_lines=     get_midline_pointswithregto664_2(niifilenametosave_fixed,niifilenametosave_moving,niifilenametosave_movingmidline)
#get_midline_pointswithregto664_2(BET_GRAY_FILE_SAVE,moving_file,moving_file_intact)
    pairs1_1=[]
    pairs2_1=[]
    pairs1_1.append(slope_of_lines[1])
    pairs2_1.append(slope_of_lines[0])             
    pair1_1=np.mean(pairs1,axis=0) #slope_of_lines[0] #(pointA_upper[0],pointA_upper[1],pointB_upper[5],pair1_1[0])
    pair2_1=np.mean(pairs2,axis=0) #slope_of_lines[1] #(pair2_1[1],pair2_1[4],pair2_1[5],pair2_1[0])
    
    img_gray=nib.load(niifilenametosave_fixed) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_data=img_gray.get_fdata()
    min_img_gray=np.min(img_gray_data)
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
    img_gray_data=normalizeimage0to1(img_gray_data)*255
    show_slice_withaline(np.uint8(img_gray_data),np.array([pair1_1[1],pair1_1[2]]))
    
    point1=np.array([pair1[1],slicenumber_target_1])
    point2=np.array([pair1[2],slicenumber_target_1])
    point3=np.array([pair1_1[1],slicenumber_target_2])
    point4=np.array([pair1_1[2],slicenumber_target_2])
#    pair1=np.array([0,pointA_lower2,pointB_lower1])
#    planeSource.SetPoint1(pair1[1][1], pair1[1][0], pair1[3])
    slice_points1=findcuttingplane(target_graybet,point1,point2,point3,point4)
    static_file=target_graybet
    img_gray=nib.load(static_file) #os.path.join(DATA_DIRECTORY,file_base_name[:-7]+".nii.gz"))
    img_gray_data=img_gray.get_fdata()
    min_img_gray=np.min(img_gray_data)
#    img=exposure.rescale_intensity( img_gray.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(1000, 1200)) 
    else:
        img_gray_data=exposure.rescale_intensity( img_gray_data , in_range=(0, 200))
    img_gray_data=normalizeimage0to1(img_gray_data)*255
    
    
#    show_slice_withaline(np.uint8(img_gray_data[:,:,matched_top_slice]),np.array([pointA_upper,pointB_upper]))
    
    lineThickness=2
    SLICE_OUTPUT_DIRECTORY="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES"
    for x in range(len(slice_points1)) :
        show_slice_withaline(np.uint8(img_gray_data[:,:,slice_points1[x][0]]),np.array([slice_points1[x][1][0],slice_points1[x][1][1]]))
        slice_3_layer= np.zeros([np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[0],np.uint8(img_gray_data[:,:,slice_points1[x][0]]).shape[1],3])
        slice_3_layer[:,:,0]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
        slice_3_layer[:,:,1]= np.uint8(img_gray_data[:,:,slice_points1[x][0]]) #imgray1
        slice_3_layer[:,:,2]= np.uint8(img_gray_data[:,:,slice_points1[x][0]])# imgray1
        img_with_line1=cv2.line(slice_3_layer, (int(slice_points1[x][1][0][0]),int(slice_points1[x][1][0][1])), (int(slice_points1[x][1][1][0]),int(slice_points1[x][1][1][1])), (0,255,0), lineThickness)
        cv2.imwrite(os.path.join(SLICE_OUTPUT_DIRECTORY,os.path.basename(static_file).split(".nii")[0]+"_" +str(slice_points1[x][0]))+".png",img_with_line1)
 
    
## midline with only Gabor method:
        
def midline_withOnly_gabor():
    
    
    return "XX"
    


    
    
## midline with registration method:
        
## midline with regsitration + Gabor method:

def register_twoimages_test_savefile_v1_sh():
#    filenam_suffix="CTnolesion.nii.gz"
    type_of_transform= 'Rigid'
    static_file= "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #CTnolesion.nii" #WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii"
#sys.argv[1]
    moving_file=sys.argv[1]
    static_file_basename_split=os.path.basename(static_file).split(".nii")
    filenam_suffix="REG"+ static_file_basename_split[0] +".nii.gz"  # + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    fixed=fixed.resample_image_to_target(moving)
#    moving_numpy=moving.numpy()
    #show_slices(moving_numpy)
    #fixed = ants.resample_image(fixed, (64,64), 1, 0)
    #moving = ants.resample_image(moving, (64,64), 1, 0)
#    nameebeforenii=moving.split(".nii")
    create_BETgray(moving_file)
    static_file_bet=static_file.split(".nii")[0]+ "BETgray.nii.gz" 
    
    
    moving_file_bet=moving_file.split(".nii")[0]+ "BETgray.nii.gz"
    
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # transform the BET gray image and save:
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    
    # Transform the complete image and save: 
    mywarpedimage=apply_transformation_1(fixed_bet, moving,mytx)
    moving_file_split=moving_file.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
#    hf = h5py.File(moving_file_split[0]+'TX.h5', 'w')
#    hf.create_dataset('mytx', data=mytx)
#    hf.close()
#    ants.write_transform(mytx, moving_file_split[0]+'TX.mat')
#    print(mytx)
#    command="cp " + mytx['fwdtransforms'][0] + "  " + moving_file_split[0]+'TX0.mat'
#    subprocess.call(command,shell=True)
#    command="cp " + mytx['fwdtransforms'][1] + "  " + moving_file_split[0]+'TX1.mat'
#    subprocess.call(command,shell=True)
#    mytx['fwdtransforms'][0]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX0.mat'
#    mytx['fwdtransforms'][1]=moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX1.mat'
#    np.save(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2' +'.npy', {mytx['fwdtransforms']}) 
#    hf = h5py.File(moving_file_split[0]+filenam_suffix.split(".nii")[0] + 'TX2.h5', 'w')
#    hf.create_dataset('mytx', data=mytx['fwdtransforms'])
#    hf.close()
    return mytx,fixed_bet

def registertotemplateHEADNSave_sh():
    #    filenam_suffix="CTnolesion.nii.gz"
    type_of_transform= 'Rigid'
    static_file=" /media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2WHOLEHEAD.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #CTnolesion.nii" #
#sys.argv[1]
    moving_file=sys.argv[1]
    eht_image_itk,seg_explicit_thresholds=seg_brain_with_skull(moving_file)
    filenametosave=moving_file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
    sitk.WriteImage(eht_image_itk,filenametosave)  
    
    
#    seg_skullbone(moving_file)
    static_file_basename_split=os.path.basename(static_file).split(".nii")
    filenam_suffix="REG"+ static_file_basename_split[0] +".nii.gz"  # + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    create_BETgray(moving_file)
    static_file_bet=static_file#static_file.split(".nii")[0]+ "HEADgray.nii.gz" 
    moving_file_bet=filenametosave#moving_file.split(".nii")[0]+ "HEADgray.nii.gz"
    
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
#    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # transform the BET gray image and save:
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ants.image_write(mywarpedimage,reg_output_filename1) 
    
    return "XX"
    
    
def registertotemplateHEADNSave_py(moving_file):
    #    filenam_suffix="CTnolesion.nii.gz"
    type_of_transform= 'Rigid'
    static_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2WHOLEHEAD.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #CTnolesion.nii" #
#sys.argv[1]
#    moving_file=sys.argv[1]
    eht_image_itk,seg_explicit_thresholds=seg_brain_with_skull(moving_file)
    filenametosave=moving_file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
    sitk.WriteImage(eht_image_itk,filenametosave) 
    betfile=moving_file[:-7] + "BET.nii.gz"
    if not os.path.exists(betfile):
        betfile=moving_file.split(".nii.gz")[0] + "_bet.nii.gz"
        
#    seg_skullbone(moving_file)
    static_file_basename_split=os.path.basename(static_file).split(".nii")
    filenam_suffix="REG"+ static_file_basename_split[0].split('.')[0] +".nii.gz"  # + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    create_BETgray(moving_file)
    static_file_bet=static_file#static_file.split(".nii")[0]+ "HEADgray.nii.gz" 
    moving_file_bet=filenametosave#moving_file.split(".nii")[0]+ "HEADgray.nii.gz"
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving_realbet=ants.image_read(betfile)
#    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # transform the BET gray image and save:
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ## save the registered HEAD area gray scale
    ants.image_write(mywarpedimage,reg_output_filename1)
    image=sitk.ReadImage(reg_output_filename1)
    image_np=sitk.GetArrayFromImage(image)
    image_np[image_np>np.min(image_np)]=np.max(image_np)
    image_np_itk=sitk.GetImageFromArray(image_np)
    image_np_itk.CopyInformation(image)
    BETFILENAME=reg_output_filename1.split(".nii.gz")[0]+"BET.nii.gz"
    ## save the registered HEAD area BET
    sitk.WriteImage(image_np_itk,BETFILENAME)
    mywarpedimage=apply_transformation_1(fixed_bet, moving_realbet,mytx)
    BETFILENAME_T=betfile.split(".nii.gz")[0]+"REALBETBIN_T.nii.gz"
    ## save the registered original BET area binary
    ants.image_write(mywarpedimage,BETFILENAME_T)
#    print(reg_output_filename1)
    GRAYSCAL_ORIG_R_FILENAME_T=moving_file.split(".nii")[0]+"REALGRAY_T.nii.gz"
    ## save the registered original BET area binary
    mywarpedimage=apply_transformation_1(fixed_bet, ants.image_read(moving_file),mytx)
    ants.image_write(mywarpedimage,GRAYSCAL_ORIG_R_FILENAME_T)
    print("GRAYSCAL_ORIG_R_FILENAME_T")
    print(GRAYSCAL_ORIG_R_FILENAME_T)
    
    return reg_output_filename1,BETFILENAME,BETFILENAME_T,GRAYSCAL_ORIG_R_FILENAME_T
def registertotemplateHEADNSave_py_1(moving_file):
    #    filenam_suffix="CTnolesion.nii.gz"
    type_of_transform= 'Rigid'
    static_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2WHOLEHEAD.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #CTnolesion.nii" #
#sys.argv[1]
#    moving_file=sys.argv[1]
    eht_image_itk,seg_explicit_thresholds=seg_brain_with_skull(moving_file)
    filenametosave=moving_file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
    sitk.WriteImage(eht_image_itk,filenametosave) 
    betfile=moving_file.split(".nii")[0] + "BET.nii.gz"
    print("betfile")
    print(betfile)
    if not os.path.exists(betfile):
        betfile=moving_file.split(".nii.gz")[0] + "_bet.nii.gz"
        
#    seg_skullbone(moving_file)
    static_file_basename_split=os.path.basename(static_file).split(".nii")
    filenam_suffix="REG"+ static_file_basename_split[0].split('.')[0] +".nii.gz"  # + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    create_BETgray(moving_file)
    static_file_bet=static_file#static_file.split(".nii")[0]+ "HEADgray.nii.gz" 
    moving_file_bet=filenametosave#moving_file.split(".nii")[0]+ "HEADgray.nii.gz"
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving_realbet=ants.image_read(betfile)
#    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # transform the BET gray image and save:
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ## save the registered HEAD area gray scale
    ants.image_write(mywarpedimage,reg_output_filename1)
    image=sitk.ReadImage(reg_output_filename1)
    image_np=sitk.GetArrayFromImage(image)
    image_np[image_np>np.min(image_np)]=np.max(image_np)
    image_np_itk=sitk.GetImageFromArray(image_np)
    image_np_itk.CopyInformation(image)
    BETFILENAME=reg_output_filename1.split(".nii.gz")[0]+"BET.nii.gz"
    ## save the registered HEAD area BET
    sitk.WriteImage(image_np_itk,BETFILENAME)
    mywarpedimage=apply_transformation_1(fixed_bet, moving_realbet,mytx)
    BETFILENAME_T=betfile.split(".nii.gz")[0]+"REALBETBIN_T.nii.gz"
    ## save the registered original BET area binary
    ants.image_write(mywarpedimage,BETFILENAME_T)
#    print(reg_output_filename1)
    GRAYSCAL_ORIG_R_FILENAME_T=moving_file.split(".nii")[0]+"REALGRAY_T.nii.gz"
    ## save the registered original BET area binary
    mywarpedimage=apply_transformation_1(fixed_bet, ants.image_read(moving_file),mytx)
    ants.image_write(mywarpedimage,GRAYSCAL_ORIG_R_FILENAME_T)
    print("GRAYSCAL_ORIG_R_FILENAME_T")
    print(GRAYSCAL_ORIG_R_FILENAME_T)
    
    return reg_output_filename1,BETFILENAME,BETFILENAME_T,GRAYSCAL_ORIG_R_FILENAME_T



def registeramasktotemplateHEADNSave_py(moving_file,betfile,mask_file):
    #    filenam_suffix="CTnolesion.nii.gz"
    type_of_transform= 'Rigid'
    static_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2WHOLEHEAD.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #CTnolesion.nii" #
#sys.argv[1]
#    moving_file=sys.argv[1]
    eht_image_itk,seg_explicit_thresholds=seg_brain_with_skull(moving_file)
    filenametosave=moving_file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
    sitk.WriteImage(eht_image_itk,filenametosave) 
#    betfile=moving_file[:-7] + "BET.nii.gz"
#    if not os.path.exists(betfile):
#        betfile=moving_file.split(".nii.gz")[0] + "_bet.nii.gz"
        
#    seg_skullbone(moving_file)
    static_file_basename_split=os.path.basename(static_file).split(".nii")
    filenam_suffix="REG"+ static_file_basename_split[0].split('.')[0] +".nii.gz"  # + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    create_BETgray(moving_file)
    static_file_bet=static_file#static_file.split(".nii")[0]+ "HEADgray.nii.gz" 
    moving_file_bet=filenametosave#moving_file.split(".nii")[0]+ "HEADgray.nii.gz"
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving_realbet=ants.image_read(betfile)
#    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # transform the BET gray image and save:
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ## save the registered HEAD area gray scale
    ants.image_write(mywarpedimage,reg_output_filename1)
    image=sitk.ReadImage(reg_output_filename1)
    image_np=sitk.GetArrayFromImage(image)
    image_np[image_np>np.min(image_np)]=np.max(image_np)
    image_np_itk=sitk.GetImageFromArray(image_np)
    image_np_itk.CopyInformation(image)
    BETFILENAME=reg_output_filename1.split(".nii.gz")[0]+"BET.nii.gz"
    ## save the registered HEAD area BET
    sitk.WriteImage(image_np_itk,BETFILENAME)
    mywarpedimage=apply_transformation_1(fixed_bet, moving_realbet,mytx)

    
    BETFILENAME_T=betfile.split(".nii.gz")[0]+"REALBETBIN_T.nii.gz"
    ## save the registered original BET area binary
    ants.image_write(mywarpedimage,BETFILENAME_T)
    image=sitk.ReadImage(BETFILENAME_T)
    image_np=sitk.GetArrayFromImage(image)
    image_np[image_np>np.min(image_np)]=np.max(image_np)
    image_np_itk=sitk.GetImageFromArray(image_np)
    image_np_itk.CopyInformation(image)
    sitk.WriteImage(image_np_itk,BETFILENAME_T)    
#    print(reg_output_filename1)
    GRAYSCAL_ORIG_R_FILENAME_T=moving_file.split(".nii")[0]+"REALGRAY_T.nii.gz"
    ## save the registered original BET area binary
    mywarpedimage=apply_transformation_1(fixed_bet, ants.image_read(moving_file),mytx)
    ants.image_write(mywarpedimage,GRAYSCAL_ORIG_R_FILENAME_T)
### save mask T
    MASK_FILENAME_T=mask_file.split(".nii")[0]+"_T.nii.gz"
    mywarpedimage=apply_transformation_1(fixed_bet, ants.image_read(mask_file),mytx)
    ants.image_write(mywarpedimage,MASK_FILENAME_T)
    MASK_FILENAME_T_nib=nib.load(MASK_FILENAME_T)
    MASK_FILENAME_T_nib_np=MASK_FILENAME_T_nib.get_fdata()
    MASK_FILENAME_T_nib_np[MASK_FILENAME_T_nib_np>0]=1.0
#    MASK_FILENAME_T_nib_np[MASK_FILENAME_T_nib_np<=0.5]=0.0
    array_img = nib.Nifti1Image(MASK_FILENAME_T_nib_np, affine=MASK_FILENAME_T_nib.affine, header=MASK_FILENAME_T_nib.header)
    nib.save(array_img, MASK_FILENAME_T)

    
    print("GRAYSCAL_ORIG_R_FILENAME_T")
    print(GRAYSCAL_ORIG_R_FILENAME_T)
    
    return reg_output_filename1,BETFILENAME,BETFILENAME_T,GRAYSCAL_ORIG_R_FILENAME_T,MASK_FILENAME_T,mytx,fixed_bet

def registeramasktotemplateBETSave_py(moving_file,betfile,mask_file):
    #    filenam_suffix="CTnolesion.nii.gz"
    type_of_transform= 'Rigid'
    static_file="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2BETgray.nii.gz" #"WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2WHOLEHEAD.nii.gz" # "/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/REGISTRATION2TEMPLATE/DATA/TEMPLATE/WUSTL_664_05072018_0942_Head_Spiral_3.0_J40s_2_20180507094133_2.nii" #CTnolesion.nii" #
#sys.argv[1]
#    moving_file=sys.argv[1]
#    eht_image_itk,seg_explicit_thresholds=seg_brain_with_skull(moving_file)
#    filenametosave=moving_file.split(".nii")[0] + "WHOLEHEAD.nii.gz"
#    sitk.WriteImage(eht_image_itk,filenametosave) 
    filenametosave=moving_file
#    betfile=moving_file[:-7] + "BET.nii.gz"
#    if not os.path.exists(betfile):
#        betfile=moving_file.split(".nii.gz")[0] + "_bet.nii.gz"
        
#    seg_skullbone(moving_file)
    static_file_basename_split=os.path.basename(static_file).split(".nii")
    filenam_suffix="REG"+ static_file_basename_split[0].split('.')[0] +".nii.gz"  # + static_file_basename_split[1] + static_file_basename_split[1] + static_file_basename_split[3] + ".nii.gz"
#    create_BETgray(moving_file)
    static_file_bet=static_file#static_file.split(".nii")[0]+ "HEADgray.nii.gz" 
    moving_file_bet=filenametosave#moving_file.split(".nii")[0]+ "HEADgray.nii.gz"
    print(static_file_bet)
    print(moving_file_bet)
    print(os.path.isfile(static_file_bet))
    print(os.path.isfile(moving_file_bet))
    fixed_bet = ants.image_read(static_file_bet)#os.path.join(result_directory,"fixed_slice.nii.gz"))  # ants.get_ants_data('r16') )
    moving_bet = ants.image_read(moving_file_bet)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    moving_realbet=ants.image_read(betfile)
#    moving = ants.image_read(moving_file)#(os.path.join(result_directory,"moving_slice.nii.gz"))    #.resample_image(fixed.GetSize()) # ants.get_ants_data('r64') )
    mytx = ants.registration(fixed=fixed_bet , moving=moving_bet , type_of_transform = type_of_transform,reg_iterations=(100,100,200))
    # transform the BET gray image and save: 
    mywarpedimage=apply_transformation_1(fixed_bet, moving_bet,mytx)
    moving_file_split=moving_file_bet.split(".nii")
    reg_output_filename1=moving_file_split[0]+ filenam_suffix
    ## save the registered HEAD area gray scale
    ants.image_write(mywarpedimage,reg_output_filename1)
    image=sitk.ReadImage(reg_output_filename1)
    image_np=sitk.GetArrayFromImage(image)
    image_np[image_np>np.min(image_np)]=np.max(image_np)
    image_np_itk=sitk.GetImageFromArray(image_np)
    image_np_itk.CopyInformation(image)
    BETFILENAME=reg_output_filename1.split(".nii.gz")[0]+"BET.nii.gz"
    ## save the registered HEAD area BET
    sitk.WriteImage(image_np_itk,BETFILENAME)
    mywarpedimage=apply_transformation_1(fixed_bet, moving_realbet,mytx)

    
    BETFILENAME_T=betfile.split(".nii.gz")[0]+"REALBETBIN_T.nii.gz"
    ## save the registered original BET area binary
    ants.image_write(mywarpedimage,BETFILENAME_T)
    image=sitk.ReadImage(BETFILENAME_T)
    image_np=sitk.GetArrayFromImage(image)
    image_np[image_np>np.min(image_np)]=np.max(image_np)
    image_np_itk=sitk.GetImageFromArray(image_np)
    image_np_itk.CopyInformation(image)
    sitk.WriteImage(image_np_itk,BETFILENAME_T)    
#    print(reg_output_filename1)
    GRAYSCAL_ORIG_R_FILENAME_T=moving_file.split(".nii")[0]+"REALGRAY_T.nii.gz"
    ## save the registered original BET area binary
    mywarpedimage=apply_transformation_1(fixed_bet, ants.image_read(moving_file),mytx)
    ants.image_write(mywarpedimage,GRAYSCAL_ORIG_R_FILENAME_T)
### save mask T
    MASK_FILENAME_T=mask_file.split(".nii")[0]+"_T.nii.gz"
    mywarpedimage=apply_transformation_1(fixed_bet, ants.image_read(mask_file),mytx)
    ants.image_write(mywarpedimage,MASK_FILENAME_T)
    MASK_FILENAME_T_nib=nib.load(MASK_FILENAME_T)
    MASK_FILENAME_T_nib_np=MASK_FILENAME_T_nib.get_fdata()
    MASK_FILENAME_T_nib_np[MASK_FILENAME_T_nib_np>0]=1.0
#    MASK_FILENAME_T_nib_np[MASK_FILENAME_T_nib_np<=0.5]=0.0
    array_img = nib.Nifti1Image(MASK_FILENAME_T_nib_np, affine=MASK_FILENAME_T_nib.affine, header=MASK_FILENAME_T_nib.header)
    nib.save(array_img, MASK_FILENAME_T)

    
    print("GRAYSCAL_ORIG_R_FILENAME_T")
    print(GRAYSCAL_ORIG_R_FILENAME_T)
    
    return reg_output_filename1,BETFILENAME,BETFILENAME_T,GRAYSCAL_ORIG_R_FILENAME_T,MASK_FILENAME_T,mytx,fixed_bet

def topslice_matching_refine_withjpg_py(staticfile_gray_bet,movingfile_gray_bet,slicenumbertomatch=10,startingslice=0):

#    niifilenametosave=staticfile_gray_bet.split(".nii")[0]+ "SLICENUM" + str(slicenumbertomatch) + ".nii.gz"
#    saveslicesofnifti(staticfile_gray_bet,savetodir=os.path.dirname(staticfile_gray_bet))#os.path.dirname(filenametosave))
    niifilenametosave=savesingleslicesofnifti(staticfile_gray_bet,slicenumbertomatch,savetodir=os.path.dirname(staticfile_gray_bet))
    print(niifilenametosave)
##    saveoneslicein2Dpy(staticfile_gray_bet,slicenumbertomatch,niifilenametosave)
    gray_nifti_data=nib.load(movingfile_gray_bet).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
    all_errors=[]
    for x in range(int(gray_nifti_data.shape[2])):
#        if np.sum(gray_nifti_data[:,:,x]) > 0:
    #        x=10
            #        niifilenametosave_moving=staticfile_gray_bet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        niifilenametosave_moving=savesingleslicesofnifti(movingfile_gray_bet,x,savetodir=os.path.dirname(movingfile_gray_bet))
        #        saveoneslicein2Dpy(movingfile_gray_bet,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Rigid",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile_gray_bet.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".jpg"

        error=ants.image_mutual_information(fixed, mywarpedimage)
        errorslice=0
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        print("mywarpedimage")
        print(mywarpedimage)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
            errorslice=minimum_error
            all_errors.append([minimum_error,x])
#                ants.image_write(mywarpedimage,reg_output_filename1) 
    #    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
    #    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select,errorslice
#    return niifilenametosave

def topslice_matching_refine1_py(staticfile_gray_bet,movingfile_gray_bet,slicenumbertomatch=10,startingslice=0):

    niifilenametosave=staticfile_gray_bet.split(".nii")[0]+ "SLICENUM" + str(slicenumbertomatch) + ".nii.gz"
    saveoneslicein2Dpy(staticfile_gray_bet,slicenumbertomatch,niifilenametosave)
    gray_nifti_data=nib.load(movingfile_gray_bet).get_fdata()
    fixed=ants.image_read(niifilenametosave)
    minimum_error=9999999999
    slice_to_select=0
    for x in range(int(gray_nifti_data.shape[2])):
        niifilenametosave_moving=staticfile_gray_bet.split(".nii")[0]+ "SLICENUM" + "DUMMY" + ".nii.gz"
        saveoneslicein2Dpy(movingfile_gray_bet,x,niifilenametosave_moving)
        moving=ants.image_read(niifilenametosave_moving)
        mytx = ants.registration(fixed=fixed , moving=moving , 
                                 type_of_transform = "Affine",reg_iterations=(100,100,200),aff_sampling=512,syn_sampling=512)

        mywarpedimage=apply_transformation_1(fixed,moving,mytx)
        moving_file_split=movingfile_gray_bet.split(".nii")
        reg_output_filename1=moving_file_split[0]+ "SLICENUM" + "DUMMYREG" + ".nii.gz"
        ants.image_write(mywarpedimage,reg_output_filename1) 
        error=ants.image_mutual_information(fixed, mywarpedimage)
        print("mytx")
        print(mytx)
        print("x")
        print(x)    
        print("error")
        print(error)
        if error < minimum_error:
            minimum_error=error
            slice_to_select=x
#    niifilenametosave_moving=os.path.join(os.path.dirname(staticfile),os.path.basename(movingfile.split(".nii")[0]+ "SLICENUM" +str(slice_to_select) + "SELECTED" + ".nii.gz"))
#    saveoneslicein2Dpy(movingfile,slice_to_select,niifilenametosave_moving)
    return slice_to_select

def flipNSaveNIFTI(filename):
    filename_nib=nib.load(filename)
    filename_gray_data_np=filename_nib.get_fdata()
    for x in range(filename_gray_data_np.shape[2]):
        filename_gray_data_np[:,:,x] =   cv2.flip( filename_gray_data_np[:,:,x], 1 )
    
#    filename_gray_data_np_flipped=nib.orientations.flip_axis(filename_gray_data_np, axis=0)
    array_img = nib.Nifti1Image(filename_gray_data_np, affine=filename_nib.affine, header=filename_nib.header)
    
    nib.save(array_img, filename.split(".nii")[0]+"flippedY.nii.gz")
    return "XX"

def display_maskonCT(ct_slice,mask_slice):
    binary_image_non_zero_cord = np.transpose(np.nonzero(mask_slice))
    for  each_point in binary_image_non_zero_cord : #range(0,512): #
        cv2.circle(ct_slice,(each_point[1],each_point[0]),2,(0,200,0),3)
    return ct_slice,each_point
def ct_mask_fromfolder_display(each_file,file_pineal): # directory_ct,directory_pineal):
#    file_ct= glob.glob(os.path.join(directory_ct,"*.nii")) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CTs_SP_Pineal/WUSTL_166_05022015_1332.nii"
#    
#    for each_file in file_ct:
#    mask_filename=os.path.basename(each_file).split(".nii")[0]+"*.hdr"
#    file_pineal=glob.glob(os.path.join(directory_pineal,mask_filename))[0]
    print(file_pineal)
    #file_pineal="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CTs_SP_Pineal/Pineal/WUSTL_166_05022015_1332_P.hdr"
    ct_image=nib.load(each_file).get_fdata()
    ct_image=exposure.rescale_intensity(ct_image , in_range=(0, 200))
    mask_image=nib.AnalyzeImage.from_filename(file_pineal).get_fdata()
    for x in range(ct_image.shape[2]) :
        if np.sum(mask_image[:,:,x]) > 0 :
            slice_3_layer= np.zeros([ct_image[:,:,x].shape[0],ct_image[:,:,x].shape[1],3])
            slice_3_layer[:,:,0]=ct_image[:,:,x]
            slice_3_layer[:,:,1]=ct_image[:,:,x]
            slice_3_layer[:,:,2]=ct_image[:,:,x]
            ct_slice = display_maskonCT(slice_3_layer,mask_image[:,:,x])
            show_slice(ct_slice) 
            
def niitojpg_sh():
    inputfilename=sys.argv[1]
    outputdirectory=sys.argv[2]
    img = nib.load(inputfilename)
#    ct_image=nib.load(each_file).get_fdata()

    img_data = img.get_fdata()
    ct_image=exposure.rescale_intensity(img_data , in_range=(0, 200))
#    arr = np.asarray(img_data)
#    sizearr=arr.shape
#    arr[arr > 80] = 0
#    arr[arr < 20] = 0
    #subp.call(["mkdir", imagename], shell=False)
    for i in range (ct_image.shape[2]):
        filename_base= os.path.basename(inputfilename).split(".nii")[0] + str(i) +'.jpg'
        filename= os.path.join(outputdirectory, filename_base)
        gray_image= ct_image[:,:,i] #cv.merge(arr[:,:,i],arr[:,:,i],arr[:,:,i])
        img2 = np.zeros(shape=(gray_image.shape[0],gray_image.shape[1],3))
        # img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image

        # gray_image = cv.cvtColor(arr[:,:,i], cv.COLOR_GRAY2RGB)
        cv.imwrite(filename, img2)
    #subp.call(["mv",'*.tiff' , imagename], shell=False)
    
def niitojpg_py(inputfilename,outputdirectory):
#    inputfilename=sys.argv[1]
#    outputdirectory=sys.argv[2]
    img = nib.load(inputfilename)
#    ct_image=nib.load(each_file).get_fdata()

#    img_data = img.get_fdata()
    
    min_img_gray=np.min(img.get_fdata())
    ct_image=exposure.rescale_intensity( img.get_fdata() , in_range=(0, 200))
    if min_img_gray>=0:
        ct_image=exposure.rescale_intensity( img.get_fdata() , in_range=(1000, 1200))
    
#    ct_image=exposure.rescale_intensity(img_data , in_range=(0, 200))
#    arr = np.asarray(img_data)
#    sizearr=arr.shape
#    arr[arr > 80] = 0
#    arr[arr < 20] = 0
    #subp.call(["mkdir", imagename], shell=False)
    for i in range (ct_image.shape[2]):
        filename_base= os.path.basename(inputfilename).split(".nii")[0] + str(i) +'.jpg'
        filename= os.path.join(outputdirectory, filename_base)
        gray_image= ct_image[:,:,i]*255 #cv.merge(arr[:,:,i],arr[:,:,i],arr[:,:,i])
        img2 = np.zeros(shape=(gray_image.shape[0],gray_image.shape[1],3))
        # img2 = np.zeros_like(img)
        img2[:, :, 0] = gray_image
        img2[:, :, 1] = gray_image
        img2[:, :, 2] = gray_image

        # gray_image = cv.cvtColor(arr[:,:,i], cv.COLOR_GRAY2RGB)
        cv.imwrite(filename, gray_image)
    #subp.call(["mv",'*.tiff' , imagename], shell=False)

def resample_image(stat_img,template_img): #stat_img: a string of the file name which will be resampled
    image=ants.image_read(stat_img)
    reference_image=ants.image_read(template_img)
    resample_image=ants.resample_image_to_target(image, reference_image)
    resampled_image_filename=stat_img.split(".nii")[0]+"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"

    ants.image_write(resample_image,resampled_image_filename)
#    template = nib.load(template_img) #load_mni152_template()
#    resampled_stat_img = resample_to_img(stat_img, template)
#    nib.save(resampled_stat_img,resampled_image_filename)
def resample_image_sh(): #stat_img: a string of the file name which will be resampled
    stat_img=sys.argv[1]
    template_img=sys.argv[2]
    print("nib.load(stat_img).get_fdata().shape")
    print(nib.load(stat_img).get_fdata().shape)
    affine=np.array([[1,  0, 0,
         0],
       [0, 1, 0,
         0],
       [ 0,  0,  1,
         0],
       [ 0.00,  0.000,  0.000,
         1.00]])
    resampled_stat_img = resample_img(nib.load(stat_img),target_affine=affine,target_shape=(512, 512, 256), interpolation='nearest')
#    nib.save(resampled_image, outputfile)
#
#    image=ants.image_read(stat_img)
#    reference_image=ants.image_read(template_img)
#    resample_image=ants.resample_image_to_target(image, reference_image)
#    resampled_image_filename=stat_img.split(".nii")[0]+"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"
#    ants.image_write(resample_image,resampled_image_filename)
#    template = nib.load(template_img) #load_mni152_template()
#    resampled_stat_img = resample_to_img(stat_img, template)
    resampled_image_filename=stat_img.split(".nii")[0]+ "Resampled.nii.gz" #"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"
    nib.save(resampled_stat_img,resampled_image_filename)
    
def resample_image_sh_preservename(): #stat_img: a string of the file name which will be resampled
    stat_img=sys.argv[1]
    template_img=sys.argv[2]
    print("nib.load(stat_img).get_fdata().shape")
    print(nib.load(stat_img).get_fdata().shape)
    resampled_stat_img = resample_img(nib.load(stat_img),target_affine=nib.load(stat_img).affine,target_shape=(512, 512, 256), interpolation='nearest')
#    nib.save(resampled_image, outputfile)
#
#    image=ants.image_read(stat_img)
#    reference_image=ants.image_read(template_img)
#    resample_image=ants.resample_image_to_target(image, reference_image)
#    resampled_image_filename=stat_img.split(".nii")[0]+"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"
#    ants.image_write(resample_image,resampled_image_filename)
#    template = nib.load(template_img) #load_mni152_template()
#    resampled_stat_img = resample_to_img(stat_img, template)
#    resampled_image_filename=stat_img.split(".nii")[0]+ "Resampled.nii.gz" #"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"
    nib.save(resampled_stat_img,stat_img)
def fillholes_bet_py(stat_img): #stat_img: a string of the file name which will be resampled

    file = stat_img #os.path.join(directory_name,filename)
    img_T1 =  sitk.ReadImage(file) # reader.Execute();
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(0,0,0)], lower=0, upper=0.5 )
    invertimagefilter = sitk.InvertIntensityImageFilter()
    seg_explicit_thresholds_inv=invertimagefilter.Execute(seg_explicit_thresholds)
    rescaleintimagefilter = sitk.RescaleIntensityImageFilter()
    rescaleintimagefilter.SetOutputMinimum(0)
    rescaleintimagefilter.SetOutputMaximum(1)
    seg_explicit_thresholds_inv_rescaled=rescaleintimagefilter.Execute(seg_explicit_thresholds_inv)
    filled_image_filename=stat_img.split(".nii")[0] + "F.nii.gz"
    sitk.WriteImage(seg_explicit_thresholds_inv_rescaled,filled_image_filename) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES/test.nii.gz")
    return seg_explicit_thresholds_inv_rescaled # seg_explicit_thresholds,img_T1
def fillholes_bet_sh(): #stat_img: a string of the file name which will be resampled

    file = sys.argv[1] #stat_img #os.path.join(directory_name,filename)
    img_T1 =  sitk.Cast(sitk.ReadImage(file) , sitk.sitkInt8)# reader.Execute();
    img_T1=sitk.BinaryMorphologicalClosing(img_T1,(2,2,2))
    print([(int(img_T1.GetSize()[0]/2),int(img_T1.GetSize()[1]/2),int(img_T1.GetSize()[2]/2))])
    seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=[(0,0,0)], lower=0, upper=0.5 )
    invertimagefilter = sitk.InvertIntensityImageFilter()
    seg_explicit_thresholds_inv=invertimagefilter.Execute(seg_explicit_thresholds)
    rescaleintimagefilter = sitk.RescaleIntensityImageFilter()
    rescaleintimagefilter.SetOutputMinimum(0)
    rescaleintimagefilter.SetOutputMaximum(1)
    seg_explicit_thresholds_inv_rescaled=rescaleintimagefilter.Execute(seg_explicit_thresholds_inv)
    filled_image_filename=file #file.split(".nii")[0] + "F.nii.gz"
    print("filled_image_filename")
    print(filled_image_filename)
    sitk.WriteImage(seg_explicit_thresholds_inv_rescaled,filled_image_filename) #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES/test.nii.gz")
    return seg_explicit_thresholds_inv_rescaled # seg_explicit_thresholds,img_T1    
    
#    stat_img_nib=nib.load(stat_img)
#    stat_img_nib_data=np.uint8(stat_img_nib.get_fdata())*255
#    stat_img_nib_data_copy=np.copy(stat_img_nib_data)
##    stat_img_nib_data=exposure.rescale_intensity(stat_img_nib_data , in_range=(0, 255))
#    for x in range(stat_img_nib_data.shape[2]):
#        if 1>0 : #np.sum(stat_img_nib_data[:,:,x]) > 0:
#            # Copy the thresholded image.
#            im_floodfill = stat_img_nib_data[:,:,x]
#            
#            # Mask used to flood filling.
#            # Notice the size needs to be 2 pixels than the image.
#            h, w = im_floodfill.shape[:2]
#            mask = np.zeros((h+2, w+2), np.uint8)
#            
#            # Floodfill from point (0, 0)
#            cv2.floodFill(im_floodfill, mask, (0,0), 255);
#            
#            # Invert floodfilled image
#            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#            testfile=os.path.basename(stat_img).split(".nii")[0]+ str(x)+".jpg"
#            filename=os.path.join("/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/TESTIMAGES",testfile)
#            cv2.imwrite(filename, im_floodfill)
#    #    stat_img=sys.argv[1]
#    #    template_img=sys.argv[2]
#    #    print("nib.load(stat_img).get_fdata().shape")
#    #    print(nib.load(stat_img).get_fdata().shape)
#    #    resampled_stat_img = resample_img(nib.load(stat_img),target_affine=nib.load(stat_img).affine,target_shape=nib.load(template_img).get_fdata().shape, interpolation='nearest')
#    ##    nib.save(resampled_image, outputfile)
#    ##
#    ##    image=ants.image_read(stat_img)
#    ##    reference_image=ants.image_read(template_img)
#    ##    resample_image=ants.resample_image_to_target(image, reference_image)
#    ##    resampled_image_filename=stat_img.split(".nii")[0]+"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"
#    ##    ants.image_write(resample_image,resampled_image_filename)
#    ##    template = nib.load(template_img) #load_mni152_template()
#    ##    resampled_stat_img = resample_to_img(stat_img, template)
#    #    resampled_image_filename=stat_img.split(".nii")[0]+ "Resampled.nii.gz" #"RS_WRT" + os.path.basename(template_img).split(".nii")[0] + ".nii.gz"
#    #    nib.save(resampled_stat_img,resampled_image_filename)
    
def resample_itk(imageF,reference_imageF):
    image=nib.load(imageF) #sitk.ReadImage(imageF)
    reference_image=nib.load(reference_imageF) #sitk.ReadImage(reference_imageF)
    affine=np.eye(3)
    affine[2][2]=0.41509433962264153
    resampled_image = resample_img(image, target_affine=image.affine,target_shape=reference_image.shape)
    nib.save(resampled_image,"resampledimage.nii.gz")#target_shape=(512, 512, 256)
#    dimension=len(image.GetSize())
#    identity = sitk.Transform(dimension, sitk.sitkIdentity)
#
#    interpolator = sitk.sitkCosineWindowedSinc
#    default_value = 0.0
#    resampled_image=sitk.Resample(image, reference_image, identity,interpolator, default_value)
#    resampled_image.CopyInformation(reference_image)
#    sitk.WriteImage(resampled_image,"resampledimage.nii.gz")
    return resampled_image

def savevariableinpckl(directoryname_jpgfiles,filenametosave):
    allfilesindirectory=glob.glob(os.path.join(directoryname_jpgfiles,"*.jpg"))
    features_template_upper = image_features(allfilesindirectory) 
    f = open(filenametosave.split(".")[0]+".pckl", 'wb')
    pickle.dump(features_template_upper, f)
    f.close()
    
def savevariableinpckl_1(variable_value,filenametosave):
#    allfilesindirectory=glob.glob(os.path.join(directoryname_jpgfiles,"*.jpg"))
#    features_template_upper = image_features(allfilesindirectory) 
    f = open(filenametosave.split(".")[0]+".pckl", 'wb')
    pickle.dump(variable_value, f)
    f.close()
def readvariableinpckl(filenamepckl):
    f = open(filenamepckl, 'rb')
    obj = pickle.load(f)
    f.close()  
    return obj

def contrast_stretch(img,threshold_id):
    if threshold_id==1:
        ct_image=exposure.rescale_intensity( img , in_range=(0, 200))
    if threshold_id==2:
        ct_image=exposure.rescale_intensity( img , in_range=(1000, 1200))
    return ct_image