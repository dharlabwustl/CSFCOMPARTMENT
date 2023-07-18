#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Segmentation: Region Growing</h1>
# 
# In this notebook we use one of the simplest segmentation approaches, region growing. We illustrate 
# the use of three variants of this family of algorithms. The common theme for all algorithms is that a voxel's neighbor is considered to be in the same class if its intensities are similar to the current voxel. The definition of similar is what varies:
# 
# * <b>ConnectedThreshold</b>: The neighboring voxel's intensity is within explicitly specified thresholds.
# * <b>ConfidenceConnected</b>: The neighboring voxel's intensity is within the implicitly specified bounds $\mu\pm c\sigma$, where $\mu$ is the mean intensity of the seed points, $\sigma$ their standard deviation and $c$ a user specified constant.
# * <b>VectorConfidenceConnected</b>: A generalization of the previous approach to vector valued images, for instance multi-spectral images or multi-parametric MRI. The neighboring voxel's intensity vector is within the implicitly specified bounds using the Mahalanobis distance $\sqrt{(\mathbf{x}-\mathbf{\mu})^T\Sigma^{-1}(\mathbf{x}-\mathbf{\mu})}<c$, where $\mathbf{\mu}$ is the mean of the vectors at the seed points, $\Sigma$ is the covariance matrix and $c$ is a user specified constant.
# 
# We will illustrate the usage of these three filters using a cranial MRI scan (T1 and T2) and attempt to segment one of the ventricles.

######## In[11]:


# To use interactive plots (mouse clicks, zooming, panning) we use the notebook back end. We want our graphs 
# to be embedded in the notebook, inline mode, this combination is defined by the magic "%matplotlib notebook".
import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from vtk import * 
import sys
import six
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import glob
from scipy import ndimage as ndi
import np_obb
from sympy import *
from skimage import exposure
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.util import invert
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts/imagedisplay')
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/SOFTWARE/pyscripts/csfbased')
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CommonPrograms/pyscripts')
sys.path.append('/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/HEMORRHAGE/SOFTWARE/pyScripts')
from utilities import *
from utilities_simple_trimmed import *
# from vtk_python_functions import *
# from csf_bounding_box import *
# from matplotlib.patches import Circle
# import subprocess
# import cv2
# from savenumpymatrix import *
# identify the slice which contains the
import vtk
import itk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import svd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import sys, subprocess
# from volume_rendering_vtk import *
# from vtk import *
# from vtk import (
#     vtkJPEGReader, vtkImageCanvasSource2D, vtkImageActor, vtkPolyDataMapper,
#     vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkSuperquadricSource,
#     vtkActor, VTK_MAJOR_VERSION
# )
# from myfunctions import *
import re
colors = vtk.vtkNamedColors()
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
#latexfilename="test.tex"
def slicenum_at_end(image):
    image_copy=np.zeros([image.shape[1],image.shape[2],image.shape[0]])
    for i in range(image.shape[0]):
        image_copy[:,:,i]=image[i,:,:]
    
    return image_copy
    
    

def combine_masks_as_color(several_masks, image=np.random.rand(10,10,3)):
    if len(several_masks)==0:
        several_masks=[]
        several_masks.append(image)
        image=np.random.rand(10,10,3)
        several_masks.append(image)
        image=np.random.rand(10,10,3)
        several_masks.append(image)
        image=np.random.rand(10,10,3)
        several_masks.append(image)
        image=np.random.rand(10,10,3)
        several_masks.append(image)
    np.random.seed(0)
    color_factor=np.random.rand(len(several_masks),3)
    
    for i in range(len(several_masks)):
        print("len(several_masks)")
        print(len(several_masks))
        
        this_image=several_masks[i]
        print("this_image")
        print(this_image.shape)
    for k in range(this_image.shape[2]):
        one_rgb_image=np.zeros([this_image.shape[0],this_image.shape[1],3])
        one_rgb_image[this_image[:,:,k]>0,0]=color_factor[i,0] #color_factor[i]
        one_rgb_image[this_image[:,:,k]>0,1]=color_factor[i,1] #color_factor[i]
        one_rgb_image[this_image[:,:,k]>0,2]=color_factor[i,2] #color_factor[i]
    filename=str(k)  + ".jpg" 
    save_matplot(one_rgb_image,filesaveas=filename)
   
def combine_masks_as_color_1(latexfilename,gray_image,several_masks,directoryname="../RESULTS/IMAGES/", niifilename="niifilename", image=np.random.rand(10,10,3)):
#    if len(several_masks)==0:
#        several_masks=[]
#        several_masks.append(image)
#        image=np.random.rand(10,10,3)
#        several_masks.append(image)
#        image=np.random.rand(10,10,3)
#        several_masks.append(image)
#        image=np.random.rand(10,10,3)
#        several_masks.append(image)
#        image=np.random.rand(10,10,3)
#        several_masks.append(image)
    thisfile=niifilename
    thisfile=thisfile.split('.')
    thisfile=thisfile[0].split('_')
    thisfileuniqueid=thisfile[0] +  thisfile[1] +  thisfile[2] + thisfile[3]  
    np.random.seed(0)
    num_class_thresh= np.unique(several_masks)
    color_factor=np.random.rand(len(num_class_thresh),3)

#    for i in range(len(several_masks)):
#        print("len(several_masks)")
#        print(len(several_masks))
#        
#        this_image=several_masks[i]
#        print("this_image")
#        print(this_image.shape)
#frame_start(filename,frametitle="ATUL")
#call_for_all_files(RAW_DATA_FOLDER,grayscale_suffix,masksuffix,betsuffix)
#latex_insert_text(filename,text="ATUL KUMAR",texttype="item")
#latex_insert_image(filename,image="testplot.jpg",caption="ATUL",imagescale=0.3)
#frame_end(filename) latex_insertimage_table2c(filename,image_dir="/home/atul/Pictures/",image1="lion.jpg", image2="lion.jpg",caption="ATUL",imagescale=0.5)
#    frame_start(latexfilename,frametitle=thisfileuniqueid)
#    latex_begin_image(latexfilename)
    counter=0

    for k in range(several_masks.shape[2]):

#            frame_start(latexfilename,frametitle="ATUL")
        one_rgb_image=np.zeros([several_masks.shape[0],several_masks.shape[1],3])
        gray_rgb_image=np.zeros([several_masks.shape[0],several_masks.shape[1],3])
        
        for i in range(len(num_class_thresh)):
            if num_class_thresh[i]==0:
                print("color_factor")
                print(num_class_thresh)
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],0]=0 #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],1]=0 #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],2]=0 #color_factor[i]
#                gray_rgb_image[:,:,0]=gray_image[:,:,0]
#                gray_rgb_image[:,:,1]=gray_image[:,:,1]
#                gray_rgb_image[:,:,2]=gray_image[:,:,2]
            else:
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],0]=color_factor[i,0] #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],1]=color_factor[i,1] #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],2]=color_factor[i,2] #color_factor[i]
        gray_rgb_image[:,:,0]=gray_image[:,:,k]
        gray_rgb_image[:,:,1]=gray_image[:,:,k]
        gray_rgb_image[:,:,2]=gray_image[:,:,k]
        filename=thisfileuniqueid+"slice" + str(k)  + ".jpg" 
        filename_gray=thisfileuniqueid+"grayslice" + str(k)  + ".jpg" 
        plot_im(one_rgb_image,filesaveas=os.path.join(directoryname,filename))
        plot_im(gray_rgb_image,filesaveas=os.path.join(directoryname,filename_gray))
        latex_start_table2c(latexfilename)
        latex_insertimage_table2c(latexfilename,image1=os.path.join(directoryname,filename_gray), image2=os.path.join(directoryname,filename),caption="ATUL",imagescale=0.4)
        latex_end_table2c(latexfilename)
#        latex_include_image(latexfilename,image_dir="../RESULTS/IMAGES/", image=filename,caption=filename,imagescale=0.1)
#        latex_insert_image(latexfilename,image=filename,caption=filename,imagescale=0.35)
#    latex_end_image(latexfilename)
#    frame_end(latexfilename)
#        if k%10==0 and k>0:
#            frame_end(latexfilename)


def combine_masks_as_color_v2(latexfilename,gray_image,several_masks,directoryname="../RESULTS/IMAGES/", niifilename="niifilename", image=np.random.rand(10,10,3)):

    thisfile=os.path.basename(niifilename)
    thisfile=thisfile.split('.')
    thisfile=thisfile[0].split('_')
    thisfileuniqueid=thisfile[0] +  thisfile[1] +  thisfile[2] + thisfile[3]  
    np.random.seed(0)
    num_class_thresh= np.unique(several_masks)
    color_factor=np.random.rand(len(num_class_thresh),3)
    counter=0
    for k in range(several_masks.shape[2]):
        one_rgb_image=np.zeros([several_masks.shape[0],several_masks.shape[1],3])
        gray_rgb_image=np.zeros([several_masks.shape[0],several_masks.shape[1],3])
        
        for i in range(len(num_class_thresh)):
            if num_class_thresh[i]==0:
#                print("color_factor")
#                print(num_class_thresh)
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],0]=0 #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],1]=0 #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],2]=0 #color_factor[i]
            else:
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],0]=color_factor[i,0] #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],1]=color_factor[i,1] #color_factor[i]
                one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],2]=color_factor[i,2] #color_factor[i]
        gray_rgb_image[:,:,0]=gray_image[:,:,k]
        gray_rgb_image[:,:,1]=gray_image[:,:,k]
        gray_rgb_image[:,:,2]=gray_image[:,:,k]
        filename=thisfileuniqueid+"slice" + str(k)  + ".jpg" 
        filename_gray=thisfileuniqueid+"grayslice" + str(k)  + ".jpg" 
#        print("filename")
#        print(filename)
        plot_im(one_rgb_image,filesaveas=os.path.join(directoryname,filename))
        plot_im(gray_rgb_image,filesaveas=os.path.join(directoryname,filename_gray))
        latex_insert_line(latexfilename,"\\texttt{\\detokenize{" +filename_gray[:-4] + "}}")
        latex_start_table2c(latexfilename)
        latex_insertimage_table2c(latexfilename,image1=os.path.join(directoryname,filename_gray), image2=os.path.join(directoryname,filename),caption="ATUL",imagescale=0.4)
        latex_end_table2c(latexfilename)

def save_matplot(image=np.ones([10,10,3]),filesaveas="testplot.jpg"):
    fig=plot_im(image, dpi=80)
#    plt.figure(figsize=(25,25))
#    plt.axis("off")
#
#    plt.imshow(image)
##    plt.show()
##    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
##    fig.subplots_adjust(hspace = .5, wspace=.001)
##    
##    axs = axs.ravel()
##    
##    for i in range(10):
##    
##        axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
##        axs[i].set_title(str(250+i))
    fig.savefig(filesaveas, format='jpg', dpi=400)
#    plt.close()
    return "AA"
#def save_allslices_inone(image=np.ones([512,512,30]),filesaveas="testplot.jpg"):
#
#    fig, axs = plt.subplots(figsize=(150, 60), facecolor='w', edgecolor='k') #int(image.shape[2]/5),5, 
##    plt.figure(figsize = (150, 60))
##    gs1 = gridspec.GridSpec(int(image.shape[2]), int(image.shape[2]/(image.shape[2]/5)))
##    gs1.update(wspace=0, hspace=0)
#    fig.subplots_adjust(hspace = 0, wspace=0)
#    
#    axs = axs.ravel()
#
#    for i in range(image.shape[2]):
##        ax1 = plt.subplot(gs1[i])
##        ax1.imshow(image[:,:,i])
##        axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
#        axs[i].imshow(image[:,:,i]) # (str(250+i))
#        axs[i].set_title(str(i))
#        axs[i].axis("off")
#    plt.savefig(filesaveas)
#    return "AA"
def save_allslices_inone(image=np.ones([512,512,30]),filesaveas="testplot.jpg"):
    
    fig, axs = plt.subplots(int(image.shape[2]/5),5, figsize=(150, 60), facecolor='w', edgecolor='k')
    
#    fig.subplots_adjust(hspace = 0, wspace=0)
    
    axs = axs.ravel()

    for i in range(image.shape[2]):
    
#        axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
        axs[i].imshow(image[:,:,i]) # (str(250+i))
        axs[i].set_title(str(i))
        axs[i].axis("off")
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
    plt.tight_layout()
    plt.savefig(filesaveas)
    return "AA"    


def subtract_binary(binary_imageBig,binary_imageSmall):
    print("I am here")
    binary_image_non_zero_cord_t=np.nonzero(binary_imageSmall>0)
#    print(binary_image_non_zero_cord_t)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    print("binary_image_non_zero_cord")
    print(binary_image_non_zero_cord)
#    binary_imageBig[binary_image_non_zero_cord]=0
#    print(binary_image_non_zero_cord_t)
    for each_point in binary_image_non_zero_cord:
        binary_imageBig[each_point[0],each_point[1],each_point[2]]=0
#        print("right")
#        if multiply_with_plane<0:
#            left_cords.append(each_point)    
#            print("left")
#    print(right_cords)
#    binary_image[right_cords]=0
#    plot_image_slice(binary_image)
    return binary_imageBig

def subtract_binary_1(binary_imageBig,binary_imageSmall):
    binary_imageBigCopy=np.copy(binary_imageBig)
    binary_imageBigCopy[binary_imageSmall>0]=0
    return binary_imageBigCopy


def planefit_lsq(data):
#    mean = np.array([0.0,0.0,0.0])
#    cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
#    data = np.random.multivariate_normal(mean, cov, 50)
#    data=data[0:6]
    #for p in data:
    #    draw_point_call(p)
    # regular grid covering the domain of the data
    X,Y = np.meshgrid(range(512), range(512))
    XX = X.flatten()
    YY = Y.flatten()
    
    order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
        Z1=Z.flatten()
        plane_cords=np.zeros((len(Z1),3))
        plane_cords[:,0]=XX
        plane_cords[:,1]=YY
        plane_cords[:,2]=Z1
    #    for p in plane_cords:
    #        draw_point_call(p)
#        print(plane_cords[0])
        pointA=plane_cords[0]
        pointB=plane_cords[int(len(plane_cords)/2)]
        pointC=plane_cords[len(plane_cords)-1]
        AB=pointB-pointA
        AC=pointC-pointA
        A_normal=np.cross([AC[0],AC[1],AC[2]],[AB[0],AB[1],AB[2]])
        normal_plane=np.cross(AB,AC)
    
        normal_plane_unit = normal_plane/np.sqrt(np.sum(normal_plane*normal_plane))
        print(AB)
        print(AC)
        print(normal_plane_unit)
#        print(normal_plane)
        actor=draw_plane(pointA,normal_plane_unit)
#        actor=draw_plane([0,0,0],[0,0,1])
        renderer.AddActor(actor)
    #    draw_perpendicular_plane_withpoints(plane_cords[0:4],renderer)
    
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
        
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    
def draw_point_call(points):
    actor1= draw_points(points)
    renderer.AddActor(actor1)

    
    
def draw_perpendicular_plane_eachslice(center,normal,renderer):
    for img_idx in range(15,20): #numpy_image.shape[2]):
#        center=[0,0,img_idx]
#        normal=[0,1,0]
        actor1= draw_plane(center,normal)
        renderer.AddActor(actor1)
def draw_perpendicular_plane_oneslice(center,normal,renderer):
    actor1= draw_plane(center,normal)
    renderer.AddActor(actor1)
    return "YY"
def draw_perpendicular_plane_oneslice1(center,normal,N,renderer):
    actor1= draw_plane(center,normal,N)
    renderer.AddActor(actor1)
    return "YY"
def draw_perpendicular_plane_withpoints(points,renderer):
    actor1= draw_plane_with_points(points)
    renderer.AddActor(actor1)
    return "YY"
   
def draw_imageplanes_vtk(numpy_image,renderer,slice_range1=0,slice_range2=0):
    for img_idx in range(0,numpy_image.shape[2]): #range(15,20): #
        if img_idx >=slice_range1 and img_idx<=slice_range2:
            actor=image_plane_vtk(numpy_image[:,:,img_idx],img_idx, rgbFrame=None)
            renderer.AddActor(actor)
    return "ZZ"
def draw_imageplanes_vtk_1(numpy_image,renderer,slice_range=1):
    for img_idx in range(0,numpy_image.shape[0]): #numpy_image.shape[2]):
        actor=image_plane_vtk(numpy_image[img_idx,:,:],img_idx, rgbFrame=None)
        renderer.AddActor(actor)
    return "ZZ"
def draw_imageplanes_vtk_2(numpy_image,img_idx,renderer,slice_range=1):
#    for img_idx in range(0,numpy_image.shape[0]): #numpy_image.shape[2]):
    actor=image_plane_vtk(numpy_image,img_idx, rgbFrame=None)
    renderer.AddActor(actor)
    return "ZZ"
def plot_image_slice(image_data):  ## slice number in the back
    fig = plt.figure()
    # Add an axes
    ax = fig.add_subplot(111,projection='3d')
    xx, yy = np.meshgrid(range(512), range(512))
    for slice_z in range(0,image_data.shape[2]):
        ax.contourf(xx, yy, image_data[:,:,slice_z], zdir='z', offset=slice_z, cmap=cm.gray,alpha=0.7)
    plt.show()
    return "XX"

def convert_imageslice_3Dpoints(image_data):
    
    return "XX"
def left_right_csf_itk(binary_image,plane_normal=np.array([1,0,0]),plane_point=np.array([256,256,10])):
    print("I am here")
    binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
    print(binary_image_non_zero_cord_t)
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    print(binary_image_non_zero_cord_t)
    left_cords=[]
    right_cords=[]
    for each_point in binary_image_non_zero_cord:
        multiply_with_plane=np.dot(np.subtract(np.array(each_point),plane_point),plane_normal) #np.dot(np.array(each_point),plane_normal) + plane_point
        if multiply_with_plane>0:
            right_cords.append(each_point)
            binary_image[each_point[0],each_point[1],each_point[2]]=0
            print("right")
#        if multiply_with_plane<0:
#            left_cords.append(each_point)    
#            print("left")
#    print(right_cords)
#    binary_image[right_cords]=0
#    plot_image_slice(binary_image)
    return binary_image


def left_right_csf(binary_image,plane_normal=np.array([1,0,0]),plane_point=np.array([256,256,10])):
#    print("I am here")
#    plane_normal=np.array([1,0,0])
#    plane_point=np.array([256,256,10])
    binary_image_copy=np.copy(binary_image)
    planeA = Plane(Point3D(plane_point[0],plane_point[1],plane_point[2]), normal_vector=(plane_normal[0],plane_normal[1],plane_normal[2]))
    
    binary_image_non_zero_cord_t=np.nonzero(binary_image_copy>0)
    
    binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
    binary_image_non_zero_cord_copy=np.copy(binary_image_non_zero_cord)
    binary_image_non_zero_cord_copy[:,0]=binary_image_non_zero_cord[:,1]
    binary_image_non_zero_cord_copy[:,1]=binary_image_non_zero_cord[:,0]
#    binary_image_non_zero_cord=np.copy(binary_image_non_zero_cord_copy)
##    print(binary_image_non_zero_cord.shape)
    left_cords=[]
    right_cords=[]
    count=0
    leftcount=0
    rightcount=0
    for each_point in binary_image_non_zero_cord:
        
#        multiply_with_plane=np.dot(np.subtract(np.array(each_point),plane_point),plane_normal) #np.dot(np.array(each_point),plane_normal) + plane_point
#        print(multiply_with_plane)
#        if multiply_with_plane>0:
#        print("planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) ")
#        print(planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) )
        dist_p=planeA.distance(Point3D(each_point[1],each_point[0],each_point[2]))
        x= np.float(str(N(dist_p))) #planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) )))
#        print(x)
        if not re.search('[a-zA-Z]', str(N(planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) ))):
#        if  x>0:
            yy=np.float(str(N(planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) )))
            if  yy > 0:
    #            right_cords.append(each_point)
    #            print("before")
    #            print(binary_image_copy[each_point[0],each_point[1],each_point[2]])
                binary_image_copy[each_point[0],each_point[1],each_point[2]]=100
    #            print("after")
    #            print(binary_image_copy[each_point[0],each_point[1],each_point[2]])
                leftcount=leftcount+1
    #            print("right")
    #        if multiply_with_plane<0:
            else:
    #            left_cords.append(each_point) 
                rightcount=rightcount+1
    #            binary_image_copy[each_point[0],each_point[1],each_point[2]]=0
    #            print("left")
    ##    print(right_cords)
    ##    binary_image[right_cords]=0
    ##    plot_image_slice(binary_image)
    #    print('count')
    #    print(count)
    return binary_image_copy,leftcount,rightcount
def plot3d_plane_image(image_data,point=np.array([256, 256, 5]),normal=np.array([0,0,512])):
#    point  = plane_point #np.array([1, 2, 3])
#    normal = plane_normal #np.array([1, 1, 2])
#    point2 = np.array([10, 50, 50])
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    print("image size")
    print(image_data.shape)
    d = -point.dot(normal)
#    d1 = -p1.dot(n1)
    # create x,y
    xx, yy = np.meshgrid(range(512), range(512))
    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

#    z1 = (-n1[0] * xx - n1[1] * yy - d1) * 1. /n1[2]
##    # plot the surface
##    plt3d = plt.figure().gca(projection='3d')
##    plt3d.plot_surface(xx, yy, z, alpha=0.2)
##    #and i would like to plot this point : 
##    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
#    # plot the surface
#    plt3d = plt.figure().gca(projection='3d')
#    plt3d.plot_surface(xx, yy, z, alpha=0.2)
#    # Ensure that the next plot doesn't overwrite the first plot
#    ax = plt.gca()
#    ax.scatter(point2[0], point2[1], point2[2], color='green')
#    plt.show()
    # Create the figure
    fig = plt.figure()
    # Add an axes
    ax = fig.add_subplot(111,projection='3d')
    # plot the surface
#    ax.plot_surface(xx, yy, z, alpha=0.5)
    for slice_z in range(int(image_data.shape[2]/2),image_data.shape[2]):
        ax.contourf(xx, yy, image_data[:,:,slice_z], 100, zdir='z', offset=slice_z, cmap=cm.gray,alpha=0.8)
#    ax.contourf(xx, yy, image_data[slice_z+1,:,:], 100, zdir='z', offset=slice_z+1, cmap=cm.gray,alpha=0.6)
    # and plot the point 
#    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
    plt.show()    
    
def plot3d_plane(image_data,slice_z,point=np.array([1, 2, 3]),normal=np.array([1, 1, 2]),p1=np.array([1, 2, 3]),n1=np.array([1, 1, 2]),point2 = np.array([10, 50, 50])):
#    point  = plane_point #np.array([1, 2, 3])
#    normal = plane_normal #np.array([1, 1, 2])
#    point2 = np.array([10, 50, 50])
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    print("image size")
    print(image_data.shape)
    d = -point.dot(normal)
    d1 = -p1.dot(n1)
    # create x,y
    xx, yy = np.meshgrid(range(512), range(512))
    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    
    z1 = (-n1[0] * xx - n1[1] * yy - d1) * 1. /n1[2]
##    # plot the surface
##    plt3d = plt.figure().gca(projection='3d')
##    plt3d.plot_surface(xx, yy, z, alpha=0.2)
##    #and i would like to plot this point : 
##    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
#    # plot the surface
#    plt3d = plt.figure().gca(projection='3d')
#    plt3d.plot_surface(xx, yy, z, alpha=0.2)
#    # Ensure that the next plot doesn't overwrite the first plot
#    ax = plt.gca()
#    ax.scatter(point2[0], point2[1], point2[2], color='green')
#    plt.show()
    # Create the figure
    fig = plt.figure()
    # Add an axes
    ax = fig.add_subplot(111,projection='3d')
    # plot the surface
    ax.plot_surface(xx, yy, z, alpha=0.5)
#    ax.plot_surface(xx, yy, z1, alpha=0.8)
#    for i in range(0,image_data.shape[0]):
    ax.contourf(xx, yy, image_data[slice_z,:,:], 100, zdir='z', offset=slice_z, cmap=cm.gray,alpha=0.7)
    ax.contourf(xx, yy, image_data[slice_z-1,:,:], 100, zdir='z', offset=slice_z-1, cmap=cm.gray,alpha=0.8)
    ax.contourf(xx, yy, image_data[slice_z+1,:,:], 100, zdir='z', offset=slice_z+1, cmap=cm.gray,alpha=0.6)
    # and plot the point 
#    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
    plt.show()
    
def plane_to_single_slice():
    
    return "XX"
    
    
def plot_imagein3d():
 #  Verify input arguments
#    if len(argv) > 1:
        # Read the image
#    jpeg_reader = vtkJPEGReader()
##        if not jpeg_reader.CanReadFile(argv[1]):
##            print("Error reading file %s" % argv[1])
##            return
#
#    jpeg_reader.SetFileName(argv[1])
#    jpeg_reader.Update()
#    image_data = jpeg_reader.GetOutput()
#    else:
    canvas_source = vtkImageCanvasSource2D()
    canvas_source.SetExtent(0, 100, 0, 100, 0, 0)
    canvas_source.SetScalarTypeToUnsignedChar()
    canvas_source.SetNumberOfScalarComponents(3)
    canvas_source.SetDrawColor(127, 127, 100)
    canvas_source.FillBox(0, 100, 0, 100)
    canvas_source.SetDrawColor(100, 255, 255)
    canvas_source.FillTriangle(10, 10, 25, 10, 25, 25)
    canvas_source.SetDrawColor(255, 100, 255)
    canvas_source.FillTube(75, 75, 0, 75, 5.0)
    canvas_source.Update()
    image_data = canvas_source.GetOutput()

    # Create an image actor to display the image
    image_actor = vtkImageActor()

    if VTK_MAJOR_VERSION <= 5:
        image_actor.SetInput(image_data)
    else:
        image_actor.SetInputData(image_data)

    # Create a renderer to display the image in the background
    background_renderer = vtkRenderer()

    # Create a superquadric
    superquadric_source = vtkSuperquadricSource()
    superquadric_source.SetPhiRoundness(1.1)
    superquadric_source.SetThetaRoundness(.2)

    # Create a mapper and actor
    superquadric_mapper = vtkPolyDataMapper()
    superquadric_mapper.SetInputConnection(superquadric_source.GetOutputPort())

    superquadric_actor = vtkActor()
    superquadric_actor.SetMapper(superquadric_mapper)

    scene_renderer = vtkRenderer()

    render_window = vtkRenderWindow()

    # Set up the render window and renderers such that there is
    # a background layer and a foreground layer
    background_renderer.SetLayer(0)
    background_renderer.InteractiveOff()
    scene_renderer.SetLayer(1)
    render_window.SetNumberOfLayers(2)
    render_window.AddRenderer(background_renderer)
    render_window.AddRenderer(scene_renderer)

    render_window_interactor = vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add actors to the renderers
    scene_renderer.AddActor(superquadric_actor)
    background_renderer.AddActor(image_actor)

    # Render once to figure out where the background camera will be
    render_window.Render()

    # Set up the background camera to fill the renderer with the image
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)

    # Render again to set the correct view
    render_window.Render()

    # Interact with the window
    render_window_interactor.Start()
    
def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """

    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]
def display_nii_vtk():
    
    return "vtk"

def calculate_plane(point1,point2,point3,point4):
    
    
    return "plane"
    

def draw_obb(labels,maskvalue):
    labels = np.nan_to_num(labels)
    mask = labels == maskvalue
    labels_sum=np.sum(labels)
#    print(labels_sum)
    if labels_sum>10:

#        print(np.isnan(labels).any())
#        print(np.isinf(labels).any())
        
        
        corners, centre = np_obb.get_obb_from_mask(mask)
        
        sideA=np.linalg.norm(np.array([corners[0,1],corners[0,0]]) - np.array([corners[1,1],corners[1,0]]))
        sideB=np.linalg.norm(np.array([corners[0,1],corners[0,0]]) - np.array([corners[3,1],corners[3,0]]))
        difference= np.abs(sideA-sideB)
#        print("difference")
#        print(difference)
#        print("sideA")
#        print(sideA)
#        print("sideB")
#        print(sideB)        
        if difference <  sideA and difference < sideB:
            obbs = np_obb.get_obb_from_labelim(labels)
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
#            ax.imshow(labels)
            for i in obbs.keys():
                corners, centre = obbs[i]
##                ax.scatter(centre[1],centre[0])    
#                ax.plot(corners[:,1],corners[:,0],'-')
#                ax.scatter(corners[:,1],corners[:,0]) 
##                ax.scatter((corners[0,1]+ corners[1,1])/2,(corners[0,0]+corners[1,0])/2,marker='^')
#                ax.scatter((corners[1,1]+ corners[2,1])/2,(corners[1,0]+corners[2,0])/2,marker='o')
##                ax.scatter((corners[2,1]+ corners[3,1])/2,(corners[2,0]+corners[3,0])/2,marker=r'$\clubsuit$')
#                ax.scatter((corners[3,1]+ corners[0,1])/2,(corners[3,0]+corners[0,0])/2,marker='*')            
##                plt.show()
#                ## plane point 1, plane point 2,
                plane_point1=np.array([(corners[1,1]+ corners[2,1])/2,(corners[1,0]+corners[2,0])/2])
                plane_point2=np.array([(corners[3,1]+ corners[0,1])/2,(corners[3,0]+corners[0,0])/2])
                normalofboundingbox=np.cross(np.subtract(corners[1],corners[0]),np.subtract(corners[3],corners[0]))
                plane_point3=plane_point1 + 10 * (normalofboundingbox / (normalofboundingbox**2).sum()**0.5)
                required_plane_normal= np.cross(np.subtract(plane_point3,plane_point1),np.subtract(plane_point2,plane_point1))
                # normal of the bounding box
                # Third point on the plane
                # normal of the required plane
#                print(type(corners))
                
                
        return ((corners[1,1]+ corners[2,1])/2,(corners[1,0]+corners[2,0])/2), ((corners[3,1]+ corners[0,1])/2,(corners[3,0]+corners[0,0])/2 ) # , required_plane_normal,plane_point1)
    
def loadniiwithsimpleitk(file):
#    file = os.path.join(RAW_DATA_FOLDER,filename_mask)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    image = reader.Execute();
    return image
def display_numpy_mat_bbox(image,bbox):
    fig, ax = plt.subplots(num="MRI_demo")
    circ = Circle((bbox[0],bbox[1]),10)
    ax.add_patch(circ)
    circ = Circle((bbox[2],bbox[3]),10)
    ax.add_patch(circ)
    circ = Circle((bbox[4],bbox[5]),10)
    ax.add_patch(circ)
    circ = Circle((bbox[6],bbox[7]),10)
    ax.add_patch(circ)
    ax.imshow(image, cmap="gray")
    ax.axis('off')
    plt.show()  
    
def display_numpy_mat_points(image,points):
    fig, ax = plt.subplots(num="MRI_demo")
    for point in points:
        circ = Circle((point[0],point[1]),10)
        ax.add_patch(circ)
    ax.imshow(image, cmap="gray")
    ax.axis('off')
    plt.show()
def display_numpy_mat(image):
    fig, ax = plt.subplots(num="MRI_demo")
    ax.imshow(image, cmap="gray")
    ax.axis('off')
    plt.show() 
def get_ventricles_range(numpy_array_3D_mask):
    zoneV_min_z=0
    zoneV_max_z=0
    counter=0
    for each_slice_num in range(0,numpy_array_3D_mask.shape[0]):
        pixel_gt_0 = np.sum(numpy_array_3D_mask[each_slice_num,:,:])
        if pixel_gt_0>0.0:
            if counter==0:
                zoneV_min_z=each_slice_num
                counter=counter+1
            zoneV_max_z=each_slice_num
#    print("zoneV_min_z")
#    print(zoneV_min_z)
#    print("zoneV_max_z")
#    print(zoneV_max_z)
    return zoneV_min_z,zoneV_max_z

def slice_with_ventricles(seg_explicit_thresholds,subt_image): ## itkmask
    zoneV_min_z,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
    seg_explicit_thresholds_numpy=sitk.GetArrayFromImage(seg_explicit_thresholds)
#    display_2dnumpystack(seg_explicit_thresholds_numpy)
    max_num_pixel_slice_ID=[]
    for i in range(0,seg_explicit_thresholds_numpy.shape[0]):
#        draw_perpendicular_plane_oneslice([0,0,i],[0,1,0],renderer)
        if i >=zoneV_min_z and i<=zoneV_max_z:
            this_slice_numpy=seg_explicit_thresholds_numpy[i,:,:]
            max_num_pixel_slice_ID.append([i,np.sum(this_slice_numpy)])
    max_num_pixel_slice_ID.sort(key = sortSecond, reverse = True)
    max_num_pixel_slice_ID_first2=max_num_pixel_slice_ID[0:3]
    print("max_num_pixel_slice_ID_first2.shape" )
    print(np.array(max_num_pixel_slice_ID_first2).shape)
    point_for_plane=[]
    point_x=[]
    point_y=[]
    point_z=[]
    plane_counter=0
    center1=[]
    normal1=[]
    centers=[]
    normals=[]
    for k in range(0,np.array(max_num_pixel_slice_ID_first2).shape[0]):
            this_slice_numpy=seg_explicit_thresholds_numpy[max_num_pixel_slice_ID[k][0],:,:]
#            img_T1 = sitk.BinaryDilate(img_T1!=0, 1)
            this_slice_itk=sitk.GetImageFromArray(this_slice_numpy)
#            this_slice_itk = sitk.BinaryErode(this_slice_itk!=0, 5)
#            this_slice_numpy=sitk.GetArrayFromImage(this_slice_itk)
            cc = sitk.ConnectedComponent(this_slice_itk>0)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(cc,this_slice_itk)
            maxsize_comp=0
            id_of_maxsize_comp=0
            threshold_num_point=0
#            print(len(stats.GetLabels()))
            if len(stats.GetLabels())>0:
                labels_ID_size=[]             
                for l in range(len(stats.GetLabels())):
                    labels_ID_size.append([l,stats.GetPhysicalSize(stats.GetLabels()[l])])
                    if stats.GetPhysicalSize(stats.GetLabels()[l])>maxsize_comp:
                        image = np.copy(this_slice_numpy)
                        image = skeletonize(image)
#                        display_numpy_mat(image)
                        maxsize_comp=stats.GetPhysicalSize(stats.GetLabels()[l])
                        id_of_maxsize_comp=l
                labels_ID_size.sort(key = sortSecond, reverse = True)  
#                labels_ID_size_np=np.array(labels_ID_size)
                print("labels_ID_size 2D")
                print(labels_ID_size)
                maskvalue=255
                l=id_of_maxsize_comp
                if maxsize_comp>threshold_num_point:
                    grayscale_image1 = np.zeros(shape=(this_slice_numpy.shape[0],this_slice_numpy.shape[1],3)).astype(np.float32)
                    grayscale_image1[:,:,0]=this_slice_numpy*255
                    grayscale_image1[:,:,1]=this_slice_numpy*255
                    grayscale_image1[:,:,2]=this_slice_numpy*255
                    pointa,pointb= draw_obb(grayscale_image1[:,:,0],maskvalue)
#                    point_for_plane.append([pointa[0],pointa[1],max_num_pixel_slice_ID[k][0]*1.0])
#                    point_for_plane.append([pointb[0],pointb[1],max_num_pixel_slice_ID[k][0]*1.0])
                    
                    draw_point_call([0,0,max_num_pixel_slice_ID[k][0]*1.0])
                    draw_point_call([0,512,max_num_pixel_slice_ID[k][0]*1.0])
                    draw_point_call([512,512,max_num_pixel_slice_ID[k][0]*1.0])
                    draw_point_call([512,0,max_num_pixel_slice_ID[k][0]*1.0])
                    point_for_plane.append([pointa[0],pointa[1],max_num_pixel_slice_ID[k][0]*1.0])
                    point_for_plane.append([pointb[0],pointb[1],max_num_pixel_slice_ID[k][0]*1.0])
                    center = (np.array([pointa[0],pointa[1],max_num_pixel_slice_ID[k][0]*1.0]) + np.array([pointb[0],pointb[1],max_num_pixel_slice_ID[k][0]*1.0]))/2
                    pointa1=np.array([pointa[0],pointa[1],max_num_pixel_slice_ID[k][0]*1.0])
                    pointonimageplane=np.array([0,0,max_num_pixel_slice_ID[k][0]*1.0])
                    point2onimageplane=np.array([0,512,max_num_pixel_slice_ID[k][0]*1.0])
                    point3onimageplane=np.array([512,512,max_num_pixel_slice_ID[k][0]*1.0])
                    normal_to_image_plane=np.cross((point2onimageplane-pointonimageplane),(point3onimageplane-pointonimageplane))
                    anotherpointon_req_plane = pointonimageplane + normal_to_image_plane
                    normal_req_plane=np.cross((anotherpointon_req_plane-center),(pointa1-center))
                    normal_req_plane=normal_req_plane/np.sqrt(np.sum(normal_req_plane*normal_req_plane))

                    if plane_counter==0:
                        center1=center
                        normal1=normal_req_plane
                        plane_counter=plane_counter + 1
                        
                    else:
                        center1[2]=center1[2] + 1
#                    draw_imageplanes_vtk_2(this_slice_numpy*255,max_num_pixel_slice_ID[k][0],renderer,slice_range=1)
                    
                    centers.append(center)
                    normals.append(normal_req_plane)
#                    point_x.append(pointa[0])
#                    point_x.append(pointb[0])
#                    point_y.append(pointa[1])
#                    point_y.append(pointb[1])
#                    point_z.append(max_num_pixel_slice_ID[k][0]*1.0)
#                    point_z.append(max_num_pixel_slice_ID[k][0]*1.0)                      
#                    centerlist=[]
#                    if np.array(labels_ID_size).shape[0]>1: # and stats.GetPhysicalSize(stats.GetLabels()[labels_ID_size[0][0]]) > 3000 and stats.GetPhysicalSize(stats.GetLabels()[labels_ID_size[1][0]]) >3000 :
#                        centerlist.append(stats.GetCenterOfGravity(stats.GetLabels()[labels_ID_size[0][0]]))
#                        centerlist.append(stats.GetCenterOfGravity(stats.GetLabels()[labels_ID_size[1][0]]))
#                        grayscale_image1 = np.zeros(shape=(this_slice_numpy.shape[0],this_slice_numpy.shape[1],3)).astype(np.float32)
#                        grayscale_image1[:,:,0]=this_slice_numpy*255
#                        grayscale_image1[:,:,1]=this_slice_numpy*255
#                        grayscale_image1[:,:,2]=this_slice_numpy*255
#                        pointa,pointb= draw_obb(grayscale_image1[:,:,0],maskvalue)
#                        point_x.append(pointa[0])
#                        point_x.append(pointb[0])
#                        point_y.append(pointa[1])
#                        point_y.append(pointb[1])
#                        point_z.append(max_num_pixel_slice_ID[k][0]*1.0)
#                        point_z.append(max_num_pixel_slice_ID[k][0]*1.0)                        
###                        point_for_plane.append([pointa[0],pointa[1],max_num_pixel_slice_ID[k][0]*1.0])
###                        point_for_plane.append([pointb[0],pointb[1],max_num_pixel_slice_ID[k][0]*1.0])
##                        print("Point on plane")
##                        print([pointa,pointb])
##                    else:
##                        centerlist.append(stats.GetCenterOfGravity(stats.GetLabels()[labels_ID_size[0][0]]))
##                        grayscale_image1 = np.zeros(shape=(this_slice_numpy.shape[0],this_slice_numpy.shape[1],3)).astype(np.float32)
##                        grayscale_image1[:,:,0]=this_slice_numpy*255
##                        grayscale_image1[:,:,1]=this_slice_numpy*255
##                        grayscale_image1[:,:,2]=this_slice_numpy*255
##                        pointa,pointb= draw_obb(grayscale_image1[:,:,0],maskvalue)
##                        point_x.append(pointa[0])
##                        point_x.append(pointb[0])
##                        point_y.append(pointa[1])
##                        point_y.append(pointb[1])
##                        point_z.append(max_num_pixel_slice_ID[k][0]*1.0)
##                        point_z.append(max_num_pixel_slice_ID[k][0]*1.0)  
##                        print("Point on plane")
##                        print([pointa,pointb])
###                    print("Label ID and size")
###                    print(labels_ID_size[0][0])
##                    print("Slice NUmber")
##                    print(max_num_pixel_slice_ID[k][0])
#    
##    print("Point for plane")
##    print(len(point_for_plane))
##    point_for_plane.append(np.array(point_x))
##    point_for_plane.append(np.array(point_y))
##    point_for_plane.append(np.array(point_z))
##    point_for_plane1=np.array(point_for_plane)
##    print("point_for_plane1.shape")
##    print(point_for_plane1.shape)
#    ## point perpendiculat to the image plane:
    binary_image=np.zeros([512,512,seg_explicit_thresholds_numpy.shape[0]])
    for i in range(0,seg_explicit_thresholds_numpy.shape[0]):
        binary_image[:,:,i]=seg_explicit_thresholds_numpy[i,:,:]*255

    print("subt_image.shape")
    binary_image_subt=np.zeros([512,512,subt_image.shape[0]])
    for i in range(0,seg_explicit_thresholds_numpy.shape[0]):
        binary_image_subt[:,:,i]=subt_image[i,:,:]*255
    print(subt_image.shape)
#    draw_imageplanes_vtk(binary_image_subt,renderer,zoneV_min_z,zoneV_max_z)
#    binary_image=  seg_explicit_thresholds_numpy*255 
#    draw_imageplanes_vtk_2(this_slice_numpy*255,max_num_pixel_slice_ID[k][0],renderer,slice_range=1)
    centers=np.array(centers)
    centers_mean=np.mean(centers,axis=0)
    centers_mean=[centers_mean[0],centers_mean[1],centers_mean[2]]
    normals=np.array(normals)
    normals_mean=np.mean(normals,axis=0)
    normals_mean=[normals_mean[0],normals_mean[1],normals_mean[2]]
    draw_perpendicular_plane_oneslice1(centers_mean,normals_mean,"X",renderer)
    binary_image1,leftcountven,rightcountven=left_right_csf(binary_image,normals_mean,centers_mean)
    binary_image_subt1,leftcountsul,rightcountsul=left_right_csf(binary_image_subt,normals_mean,centers_mean)
    draw_imageplanes_vtk(binary_image1,renderer,zoneV_min_z,zoneV_max_z)#,zoneV_min_z,zoneV_max_z)
    return leftcountven,rightcountven,leftcountsul,rightcountsul
##    print("binary_image.shape")
##    print(binary_image.shape)
##    print("centers")
##    print(max_num_pixel_slice_ID[2][0])
##    print("normals")
##    print(normals_mean)
##    image_x=[]
##    image_y=[]
##    image_z=[]
##    image_x.append(0)
##    image_y.append(0)
##    image_z.append(max_num_pixel_slice_ID[k][0]*1.0)
##    image_x.append(0)
##    image_y.append(511)
##    image_z.append(max_num_pixel_slice_ID[k][0]*1.0)
##    image_x.append(511)
##    image_y.append(0)
##    image_z.append(max_num_pixel_slice_ID[k][0]*1.0)
##    image_x.append(511)
##    image_y.append(511)
##    image_z.append(max_num_pixel_slice_ID[k][0]*1.0)
##    this_slice_numpy1=seg_explicit_thresholds_numpy
##    one_image_plane=[]
##    one_image_plane.append(image_x)
##    one_image_plane.append(image_y)
##    one_image_plane.append(image_z)
##    one_image_plane1=np.array(one_image_plane)
##    print("points")
##    print(point_for_plane1)
##    points=np.zeros([4,3])
##    if point_for_plane1.shape[0] > 3:
##        points[0,:]= (point_for_plane1[0]+point_for_plane1[2])/2
##        points[1,:]= (point_for_plane1[1]+point_for_plane1[3])/2
##        points[2,:]= (point_for_plane1[2]+point_for_plane1[4])/2
##        points[3,:]= (point_for_plane1[3]+point_for_plane1[5])/2
##    else:
##        points[0,:]= point_for_plane1[0]
##        points[1,:]= point_for_plane1[1]
##        points[2,:]= point_for_plane1[2]
##        points[3,:]= point_for_plane1[3]
##    p=(point_for_plane1[0]+point_for_plane1[1])/2
##    draw_point_call([p[0],p[1],p[2]])
##    draw_point_call([point_for_plane1[0][0],point_for_plane1[0][1],point_for_plane1[0][2]])
#
##    AB=point_for_plane1[1]-point_for_plane1[0]
##    AC=point_for_plane1[3]-point_for_plane1[0]
##    n=np.cross(AB,AC)*100
##    n=n/np.sqrt(np.sum(n*n))
##    print("points")
##    print(points)
##    print("p")
##    print(p)
##    print("AB")
##    print(AB)
##    print("AC")
##    print(AC)
###    draw_perpendicular_plane_withpoints(points,renderer)
##    actor=draw_plane(p,n)
##    #        actor=draw_plane([0,0,0],[0,0,1])
##    renderer.AddActor(actor)
##        p, n = planeFit(point_for_plane1)
###        p1, n1 = planeFit(one_image_plane1)
##        
###        print("point length")
###        print(point_for_plane1)
###        print("Point on plane")
##        print(p)
##        print("Normal on plane")
##        print(n)
##        draw_perpendicular_plane_oneslice(p,n,renderer)
##    print("point_for_plane")
##    point_for_plane=np.array(point_for_plane)
##    draw_perpendicular_plane_withpoints(point_for_plane,renderer)
##    print(point_for_plane.shape)
##    draw_imageplanes_vtk_1(this_slice_numpy1*255,renderer)
###        center=[0,0,10]
###        normal=[0,1,0]
###        plot3d_plane(this_slice_numpy1,max_num_pixel_slice_ID[k][0],p,n,p1,n1)
##                    im = sitk.GetImageFromArray(this_slice_numpy*255, isVector=False)
##                    sitk.WriteImage(im, "test.png", True) 
##                    display_numpy_mat_points(this_slice_numpy,centerlist)
##                    filter1=sitk.BinaryFillholeImageFilter()
##                    filter1.SetForegroundValue(1)
##                    filter1.SetFullyConnected(True)
##                    this_slice_itk=filter1.Execute(this_slice_itk)
##                    this_slice_itk = sitk.BinaryDilate(this_slice_itk!=0, 5)
##                    this_slice_itk = sitk.BinaryErode(this_slice_itk!=0, 6)
##                    this_slice_numpy=sitk.GetArrayFromImage(this_slice_itk)
##                    this_slice_numpy = ndi.binary_fill_holes(this_slice_numpy)
##                    this_slice_numpy = ndimage.binary_fill_holes(this_slice_numpy).astype(int)
##                    display_numpy_mat(this_slice_numpy)
##                    
##                    print("Label: {0} -> Mean: {1} Size: {2} Centroid: {3}".format(stats.GetLabels()[l], stats.GetMean(stats.GetLabels()[l]), stats.GetPhysicalSize(stats.GetLabels()[l]),stats.GetCenterOfGravity(stats.GetLabels()[l])))
##                    print("Bounding box")
##                    lsif = sitk.LabelShapeStatisticsImageFilter()
##                    lsif.ComputeOrientedBoundingBoxOn()
###                    lsif.GetOrientedBoundingBoxVertices(1)
##                    lsif.Execute(this_slice_itk)
##                    boundingBox = np.array(lsif.GetOrientedBoundingBoxVertices(1))
##                    print(boundingBox) #stats.GetWeightedPrincipalAxes(stats.GetLabels()[l]))
##                    display_numpy_mat_bbox(this_slice_numpy,boundingBox)
##                    src=np.copy(this_slice_numpy)
##                    src=(src- np.min(src)) /(np.max(src)-np.min(src)) * 255
##                    grayscale_image=np.copy(this_slice_numpy) #(grayscale_image- np.min(grayscale_image)) /(np.max(grayscale_image)-np.min(grayscale_image)) * 255
##                    #grayscale_image = cv2.convertScaleAbs(grayscale_image-np.min(grayscale_image), alpha=(255.0 / min(np.max(grayscale_image)-np.min(grayscale_image), 10000)))
##
##                    cv2.imwrite("test.tif",grayscale_image1[:,:,0])
##                    
##            #        src=(src- np.min(src)) /(np.max(src)-np.min(src)) * 255
##                    #img = src#*post_point_mask_reg #plt.imread("test.png")
##                   # print(np.max(src))
##            
##            #        fig, ax = plt.subplots(num="MRI_demo")
##            #        ax.imshow(src, cmap="gray")
##            #        ax.axis('off') 
##            #        plt.show()
###                    kernel = np.ones((3,3),np.uint8)
##            
##                    src=src.astype(np.uint8)
##                    grayscale_image=grayscale_image.astype(np.uint8)
##                    grayscale_image1=grayscale_image1.astype(np.uint8)
##        #            src = cv2.erode(src,kernel,iterations = 3)
##        #            src = cv2.dilate(src,kernel,iterations = 3)
##                    #cv.imwrite('src.jpg', src)
##                    #src=cv.imread()
##                    point_thresh=20
##                    plot_contours(grayscale_image1,src,i,point_thresh)
##        print(id_of_maxsize_comp) 
##        print(this_slice_itk.shape)
##        print(this_slice_numpy.shape)
##        display_numpy_mat(this_slice_numpy)
##    
##     find the connected components
##    
##     if any  connected components size is more than 200 then it is the ventricle:
##                    for l in range(len(stats.GetLabels())):  
##            if stats.GetPhysicalSize(stats.GetLabels()[l])>maxsize_comp:
##                maxsize_comp=stats.GetPhysicalSize(stats.GetLabels()[l])
##                id_of_maxsize_comp=l
##    #        print("Label: {0} -> Mean: {1} Size: {2} CentroiplaneFitd: {3}".format(stats.GetLabels()[l], stats.GetMean(stats.GetLabels()[l]), stats.GetPhysicalSize(stats.GetLabels()[l]),stats.GetCenterOfGravity(stats.GetLabels()[l])))
###        print(id_of_maxsize_comp)
##        initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])] #img_T1.TransformPhysicalPointToIndex(stats.GetCenterOfGravity(stats.GetLabels()[id_of_maxsize_comp]))]
##        initial_seed_point_indexes1=stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])
##        seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
###        display_2dnumpystack(sitk.GetArrayFromImage(seg_explicit_thresholds))
##        slice_with_ventricles(seg_explicit_thresholds)
#    return "XX"
#    
#    
#
#def get_slice_zone(numpy_array_3D,numpy_array_3D_mask,filename,outputdirectory,affine):
#    command = "mkdir  -p  " + os.path.join(outputdirectory,"V")
#    subprocess.call(command,shell=True)
#    command = "mkdir  -p  " + os.path.join(outputdirectory,"BV")
#    subprocess.call(command,shell=True)
#    command = "mkdir  -p  " + os.path.join(outputdirectory,"AV")
#    subprocess.call(command,shell=True)
#    zoneV_min_z=0
#    zoneV_max_z=0
#    counter=0
#    AV_stack=[]
#    V_stack=[]
#    BV_stack=[]
#    for each_slice_num in range(0,numpy_array_3D_mask.shape[0]):
#        pixel_gt_0 = np.sum(numpy_array_3D_mask[each_slice_num,:,:])
##        print(pixel_gt_0)
#        zone="V"
#        if pixel_gt_0>0.0:
#            if counter==0:
#                zoneV_min_z=each_slice_num
#                counter=counter+1
#            zoneV_max_z=each_slice_num
##            filenameniigz=filename[:-7] + "_" +str(each_slice_num)
##            outputdirectory_v=os.path.join(outputdirectory,zone)
#    V_3D_gray=np.copy(numpy_array_3D)
#    V_3D_gray[:,:,0:zoneV_min_z]=np.min(numpy_array_3D)
#    V_3D_gray[:,:,zoneV_max_z+1:len(numpy_array_3D_mask)]=np.min(numpy_array_3D)
#    AV_3D_gray=np.copy(numpy_array_3D)
#    AV_3D_gray[:,:,0:zoneV_max_z+1]=np.min(numpy_array_3D)
#    BV_3D_gray=np.copy(numpy_array_3D)
#    BV_3D_gray[:,:,zoneV_min_z:len(numpy_array_3D_mask)]=np.min(numpy_array_3D)
#    array_img_V = nib.Nifti1Image(V_3D_gray, affine)
#    array_img_AV = nib.Nifti1Image(AV_3D_gray, affine)
#    array_img_BV = nib.Nifti1Image(BV_3D_gray, affine)
##    print(zoneV_min_z)
##    print(zoneV_max_z)
##    print(numpy_array_3D_mask.shape[0])
#    outputdirectory_v=os.path.join(outputdirectory,"V")
#    outputdirectory_Av=os.path.join(outputdirectory,"AV")
#    outputdirectory_Bv=os.path.join(outputdirectory,"BV")
#    filenameniigz1=filename #+ ".nii.gz"
#    nib.save(array_img_V, os.path.join(outputdirectory_v,filenameniigz1))
#    nib.save(array_img_AV, os.path.join(outputdirectory_Av,filenameniigz1))
#    nib.save(array_img_BV, os.path.join(outputdirectory_Bv,filenameniigz1))   
#    
##            numpy2dmat2niigzNOAFFINE(numpy_array_3D[:,:,each_slice_num],outputdirectory_v,filenameniigz)
##            img_data = cv2.resize(numpy_array_3D[:,:,each_slice_num], (512, 512))
##            V_stack.append(img_data)
##    for each_slice_num in range(len(numpy_array_3D_mask)):call_for_all_files()
##        pixel_gt_0 = np.sum(numpy_array_3D_mask[each_slice_num,:,:])
###        zone="V"
##        counter=0
##        if pixel_gt_0==0.0:
##            print("YES")
##            if each_slice_num < zoneV_min_z:
##                zone="BV"
##                filenameniigz=filename[:-7]+ "_" +str(each_slice_num) 
##                outputdirectory_v=os.path.join(outputdirectory,zone)
###                numpy2dmat2niigzNOAFFINE(numpy_array_3D[:,:,each_slice_num],outputdirectory_v,filenameniigz)
###                img_data = cv2.resize(numpy_array_3D[:,:,each_slice_num], (512, 512))
###                BV_stack.append(img_data)
##            if each_slice_num > zoneV_max_z:
##                zone="AV"
##                filenameniigz=filename[:-7]+ "_"+str(each_slice_num) 
##
###                numpy2dmat2niigzNOAFFINE(numpy_array_3D[:,:,each_slice_num],outputdirectory_v,filenameniigz)
###                img_data = cv2.resize(numpy_array_3D[:,:,each_slice_num], (512, 512))
###                AV_stack.append(img_data)
#
##    V_stack=np.asarray(V_stack)
##    print(V_stack.shape)
##    print("\nGrayscale")
##    print(numpy_array_3D.shape)
#            
#            
#            
#            
#    
#    
##get_ipython().run_line_magic('matplotlib', 'notebook')
#
#import SimpleITK as sitk
#import subprocess
#import glob
#import os
##get_ipython().run_line_magic('run', 'update_path_to_download_script')
###command="python   update_path_to_download_script.py"
###subprocess.call(command,shell=True)
##from downloaddata import fetch_data as fdata
##import gui
#
## Using an external viewer (ITK-SNAP or 3D Slicer) we identified a visually appealing window-level setting
#T1_WINDOW_LEVEL = (0,200)#(1050,500)
#
#
######### In[12]:
#
#
RAW_DATA_FOLDER='/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CSF_SEGMENTATION/'
#OUTPUT_FOLDER='/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/Ideal/USING_CSF'
#
###        print(each_unique_names_file_split)
##        filename = each_unique_names_file_split + '_final_seg.nii.gz'
##        filename_gray = each_unique_names_file_split + '_levelset.nii.gz'
##        filename_BET = each_unique_names_file_split + '_levelset_bet.nii.gz'
##        if os.path.exists(os.path.join(RAW_DATA_FOLDER,filename)) and os.path.exists(os.path.join(RAW_DATA_FOLDER,filename_gray)) and os.path.exists(os.path.join(RAW_DATA_FOLDER,filename_BET)):
##            print("YES")
###                plot_contours_call(RAW_DATA_FOLDER,filename,filename_gray,filename_BET)
##            plot_contours_call(RAW_DATA_FOLDER,filename_BET,filename_gray,filename_BET)
#import numpy as np
#
##    print(imagenparray[25,:,:].shape)
##    print(np.max(imagenparray))
##    print(np.min(imagenparray))
#import matplotlib.pyplot as plt
def sortSecond(val): 
    return val[1]  
#def display_2dnumpystack(images):
#    for img in range(0,images.shape[0]):
#        fig, ax = plt.subplots(num="MRI_demo")
#        ax.imshow(images[img,:,:], cmap="gray")
#        print(img)
#        ax.axis('off')
#        plt.show()
#        
#def calc_central_vs_peripheral_csf(completeiamge,centralimage):
##    subtract_filter=sitk.SubtractImageFilter()
##    subtract_filter.SetInput1(completeiamge)
##    subtract_filter.SetInput2(centralimage)
##    subt_image=subtract_filter.Execute()
#    return completeiamge-centralimage
def calculate_volume(nii_img,mask_img):
#    mask_data=myfunctions.analyze_stack(mask_file)
#    img = nib.load(nii_file)
    resol= np.prod(np.array(nii_img.header["pixdim"][1:4]))
#    print("header[pixdim][1:4]")
#    print(nii_img.header["pixdim"][1:4])
    mask_data_flatten= mask_img.flatten()
    num_pixel_gt_0=mask_data_flatten[np.where(mask_data_flatten>0)]
#    print(num_pixel_gt_0)
    return (resol * num_pixel_gt_0.size)/1000
#
#    
#    
def divideintozones(latexfilename,inputdirectory,filename,filename_gray,filename_mask,filename_bet):
    sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image

    RAW_DATA_FOLDER=inputdirectory
#    filename_gray = filename+grayscale_suffix+ ".nii.gz" #'WUSTL_166_05022015_1332.nii.gz'
#    filename_mask= filename +masksuffix + ".nii.gz"  #+"_unet.nii.gz" #filename[:-9]+"_final_seg.nii.gz" # 'WUSTL_166_05022015_1332_final_seg.nii.gz'
#    filename_bet= filename +betsuffix + ".nii.gz"  #+ "_levelset_bet.nii.gz" # filename[:-9]+'WUSTL_166_05022015_1332_final_seg.nii.gz'
    file_gray = os.path.join(RAW_DATA_FOLDER,filename_gray)
    reader_gray = sitk.ImageFileReader()
    reader_gray.SetImageIO("NiftiImageIO")
    reader_gray.SetFileName(file_gray)
    filename_gray_data = reader_gray.Execute();
    
    gray_scale_file=os.path.join(RAW_DATA_FOLDER,filename_gray)
    gray_image=nib.load(gray_scale_file)
    resol= np.prod(np.array(gray_image.header["pixdim"][1:4]))
    affine_1=gray_image.affine
    gray_image_data=gray_image.get_fdata()

    file = os.path.join(RAW_DATA_FOLDER,filename_mask)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    img_T1_Copy=img_T1
#    img_T1_255 = sitk.BinaryDilate(img_T1_255!=0, 5)
    

#    img_T1 = sitk.BinaryErode(img_T1!=0, 1)
#
#    img_T1 = sitk.BinaryDilate(img_T1!=0, 1)
    imagenparray=sitk.GetArrayFromImage(img_T1)
#    display_2dnumpystack(imagenparray)
    initial_seed_point_indexes1=[]
    if np.sum(imagenparray)>200:
        img_T1=img_T1*255
        # ## Read Data and Select Seed Point(s)
        # 
        # We first load a T1 MRI brain scan and select our seed point(s). If you are unfamiliar with the anatomy you can use the preselected seed point specified below, just uncomment the line.
        
        ######## In[13]:
        # img_T1 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd"))
        # Rescale the intensities and map them to [0,255], these are the default values for the output
        # We will use this image to display the results of segmentation
        img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1) ,sitk.sitkUInt8)
#        img_T1_255_numpy=sitk.GetArrayFromImage(img_T1_255)
#        img_T1_255=sitk.GetImageFromArray(img_T1_255_numpy, isVector=True)
#        print(img_T1_255_numpy.shape)
        ######## In[16]:
    #    
    #    print(img_T1_255.GetDepth())
    #    print(img_T1_255.GetHeight())
    #    print(img_T1_255.GetWidth())
        file1 = os.path.join(RAW_DATA_FOLDER,filename_bet)
        reader1 = sitk.ImageFileReader()
        reader1.SetImageIO("NiftiImageIO")
        reader1.SetFileName(file1)
        img_T1_bet = reader1.Execute();
        cc1 = sitk.ConnectedComponent(img_T1_bet>0)
        stats1 = sitk.LabelIntensityStatisticsImageFilter()
        stats1.Execute(cc1,img_T1_bet)
#        
        cc = sitk.ConnectedComponent(img_T1_255>0)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(cc,img_T1)
        maxsize_comp=0
        id_of_maxsize_comp=0
        maxsize_comp_1=0
        id_of_maxsize_comp_1=0
        bet_ids=[]
        for l in range(len(stats1.GetLabels())):  
            if stats1.GetPhysicalSize(stats1.GetLabels()[l])>maxsize_comp_1:
                maxsize_comp_1=stats1.GetPhysicalSize(stats1.GetLabels()[l])
#                bet_ids.append([l,maxsize_comp_1])
                id_of_maxsize_comp_1=l
        csf_ids=[]
        for l in range(len(stats.GetLabels())):  
#            if stats.GetPhysicalSize(stats.GetLabels()[l])>maxsize_comp:
#            maxsize_comp=stats.GetPhysicalSize(stats.GetLabels()[l])
#                id_of_maxsize_comp=l
            csf_ids.append([l,stats.GetPhysicalSize(stats.GetLabels()[l])])
        csf_ids.sort(key = sortSecond, reverse = True)
        print("labels_ID_size 3D")
        print(csf_ids)
        ## 
#        centroid_image=np.array([256,256,int(gray_image.shape[2]/2)])
        first_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[0][0]]))
        second_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[1][0]]))
        bet_centroid=np.array(stats.GetCentroid(stats.GetLabels()[id_of_maxsize_comp_1]))
        first2bet_centroid=np.linalg.norm(first_seg_centroid - bet_centroid)
        second2bet_centroid=np.linalg.norm(second_seg_centroid - bet_centroid)
        if first2bet_centroid< second2bet_centroid:
            id_of_maxsize_comp=csf_ids[0][0]
            print("csf_ids[0][0]")
            print(csf_ids[0][0])
        else:
            if stats.GetPhysicalSize(stats.GetLabels()[csf_ids[1][0]]) > 10000: 
                id_of_maxsize_comp=csf_ids[1][0]
                print("csf_ids[1][0]")
                print(csf_ids[1][0])
            else:
                id_of_maxsize_comp=csf_ids[0][0]
                print("csf_ids[0][0]")
                print(csf_ids[0][0])
            
    #        print("Label: {0} -> Mean: {1} Size: {2} Centroid: {3}".format(stats.GetLabels()[l], stats.GetMean(stats.GetLabels()[l]), stats.GetPhysicalSize(stats.GetLabels()[l]),stats.GetCenterOfGravity(stats.GetLabels()[l])))
#        print(id_of_maxsize_comp)
#        id_of_maxsize_comp=csf_ids[1][0]
        initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])] #img_T1.TransformPhysicalPointToIndex(stats.GetCenterOfGravity(stats.GetLabels()[id_of_maxsize_comp]))]
#        initial_seed_point_indexes1=stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])
#        initial_seed_point_indexes=[img_T1_bet.TransformPhysicalPointToIndex(stats.GetCenterOfGravity(stats.GetLabels()[id_of_maxsize_comp_1]))] #[stats.GetCentroid(stats.GetLabels()[id_of_maxsize_comp])]
        seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)
#        seg_explicit_thresholds = sitk.BinaryDilate(seg_explicit_thresholds!=0, 2)
#        print("seg_explicit_thresholds")
#        print(sitk.GetArrayFromImage(seg_explicit_thresholds).shape)
#        print("img_T1")
#        print(sitk.GetArrayFromImage(img_T1).shape)
#        seg_explicit_thresholds_numpy1= sitk.GetArrayFromImage(seg_explicit_thresholds) * 255
#        print(np.max(seg_explicit_thresholds_numpy1))
#        complete_image_numpy= sitk.GetArrayFromImage(img_T1)
#        print(np.max(complete_image_numpy))
#        subtracted_image=complete_image_numpy-seg_explicit_thresholds_numpy1
##        subtracted_image[subtracted_image<0]=0
#        print(np.max(subtracted_image))
        zoneV_min_z,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
        subtracted_image=subtract_binary_1(sitk.GetArrayFromImage(img_T1_Copy),sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
        subtracted_image=sitk.GetImageFromArray(subtracted_image)
        above_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
        above_ventricle_image[0:zoneV_max_z+1,:,:]=0
        covering_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
        covering_ventricle_image[0:zoneV_min_z+1,:,:]=0
        covering_ventricle_image[zoneV_max_z+1:above_ventricle_image.shape[0],:,:]=0
        below_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
        below_ventricle_image[zoneV_min_z:above_ventricle_image.shape[0],:,:]=0


        
#        seg_explicit_thresholds_numpy=sitk.GetArrayFromImage()
#        binary_image=np.zeros([512,512,seg_explicit_thresholds_numpy.shape[0]])
#        for i in range(0,seg_explicit_thresholds_numpy.shape[0]):
#            binary_image[:,:,i]=seg_explicit_thresholds_numpy[i,:,:]*255
#        
#        print("subt_image.shape")
#        binary_image_subt=np.zeros([512,512,seg_explicit_thresholds_numpy.shape[0]])
#        for i in range(0,seg_explicit_thresholds_numpy.shape[0]):
#            binary_image_subt[:,:,i]=subtracted_image[i,:,:]*255
#        print(subtracted_image.shape)
#        draw_imageplanes_vtk(binary_image_subt,renderer,zoneV_min_z,zoneV_max_z)
        
#        subtractfilter= sitk.SubtractImageFilter()
#        subtracted_image=subtractfilter.Execute(img_T1,seg_explicit_thresholds)
#        
#        xx=sitk.GetArrayFromImage(seg_explicit_thresholds)
#        print("MASK SIZE")
#        print(xx.shape)
##        display_2dnumpystack(sitk.GetArrayFromImage(seg_explicit_thresholds))
#        subt_image = sitk.GetArrayFromImage(img_T1_255) - sitk.GetArrayFromImage(seg_explicit_thresholds) 
#        aa= sitk.GetArrayFromImage(img_T1_255) 
##        divideintozones(img_T1,seg_explicit_thresholds)
        sulci_vol=calculate_volume(gray_image,sitk.GetArrayFromImage(subtracted_image)) 
        ventricle_vol=calculate_volume(gray_image,sitk.GetArrayFromImage(seg_explicit_thresholds))
        sulci_vol_above_vent=calculate_volume(gray_image,above_ventricle_image)
        sulci_vol_below_vent=calculate_volume(gray_image,below_ventricle_image)
        sulci_vol_at_vent=calculate_volume(gray_image,covering_ventricle_image)
        allinone=np.zeros(below_ventricle_image.shape)
        allinone[below_ventricle_image>0]=100
        allinone[above_ventricle_image>0]=180
        allinone[sitk.GetArrayFromImage(seg_explicit_thresholds)>0]=240
        allinone[covering_ventricle_image>0]=255        
#        array_img = nib.Nifti1Image(allinone, affine_1)
#        array_img.header['pixdim']=gray_image.header['pixdim']
#        nib.save(array_img, os.path.join(RAW_DATA_FOLDER,gray_scale_file[:-7]+ "compartments.nii.gz"))
#        print("Peripheral CSF volume")
#        print(calculate_volume(gray_image,sitk.GetArrayFromImage(subtracted_image)))
###        draw_imageplanes_vtk(subt_image,renderer,14,17)
#        print("Ventricular volume")
#        print(calculate_volume(gray_image,sitk.GetArrayFromImage(seg_explicit_thresholds)))
##        display_2dnumpystack(subt_image)
        image_slice_jpg_dir="../RESULTS/IMAGES/" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/RESULTS/IMAGES"
        filename_gray_data_np= exposure.rescale_intensity(slicenum_at_end(sitk.GetArrayFromImage(filename_gray_data)), in_range=(1000, 1200))
        combine_masks_as_color_1(latexfilename,filename_gray_data_np,slicenum_at_end(allinone),image_slice_jpg_dir,filename_gray[:-7]) #(allmasks)
#        leftcountven,rightcountven,leftcountsul,rightcountsul  =   slice_with_ventricles(seg_explicit_thresholds,sitk.GetArrayFromImage(subtracted_image))

#        allmasks=[] #slicenum_at_end(image)
#        allmasks.append(slicenum_at_end(above_ventricle_image))
#        allmasks.append(slicenum_at_end(sitk.GetArrayFromImage(seg_explicit_thresholds)))
#        allmasks.append(slicenum_at_end(below_ventricle_image))

#        start_renderer_1(sitk.GetImageFromArray(above_ventricle_image),seg_explicit_thresholds,img_T1,sitk.GetImageFromArray(below_ventricle_image),filename)
#        get_slice_zone(gray_image_data,sitk.GetArrayFromImage(seg_explicit_thresholds),filename_gray,OUTPUT_FOLDER,affine_1)
    return sulci_vol, ventricle_vol,leftcountven*resol,rightcountven*resol,leftcountsul*resol,rightcountsul*resol,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent #seg_explicit_thresholds, subtracted_image
def start_renderer_1(sulcus_zone,ventricle_zone,img,above_ventricle_image,filename):
#    filename=os.path.join(RAW_DATA_FOLDER,"SAH_2_01062014_1205_Head_w_o_4.8_Brain.nii") #"Biogen_22960_09192014_1947_scan15652_CT1_levelset.nii.gz")
#    ventricle_zone, sulcus_zone = divideintozones(RAW_DATA_FOLDER,"SAH_2_01062014_1205_Head_w_o_4.8_Brain")
#    xx=nib.load(filename)
#    xx_data=xx.get_fdata()
###    plot3d_plane_image(xx_data[:,:,int(xx_data.shape[2]/2):xx_data.shape[2]]*255)
###    plot3d_plane_image(xx_data*255)
##    
##    binary_image=left_right_csf(xx_data*255)
#    binary_image=xx_data*255
#    binary_image=left_right_csf(binary_image)
#    draw_imageplanes_vtk(binary_image,renderer)
#    center=[0,0,10]
#    normal=[0,1,0]
#    draw_perpendicular_plane_eachslice(xx_data,center,normal,renderer)
#    img = sitk.ReadImage( filename ) # SimpleITK object
    data2 = sitk.GetArrayFromImage( above_ventricle_image ).astype('float') # sitk.GetArrayFromImage( img ).astype('float') # numpy array
    data2 *= 255 / data2.max()
    
    from scipy.stats.mstats import mquantiles
    q = mquantiles(data2.flatten(),[0.7,0.98])
    q[0]=max(q[0],1)
    q[1] = max(q[1],1)
    tf=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],1,1,1,0.5],[data2.max(),1,1,1,1]]

    actor_list2 = volumeRender(data2, tf=tf, spacing=img.GetSpacing(),color_factor=0.1)
    for a2 in actor_list2:
    # assign actor to the renderer
        renderer.AddActor(a2)
    
    
    
    data = sitk.GetArrayFromImage( sulcus_zone ).astype('float') # sitk.GetArrayFromImage( img ).astype('float') # numpy array
    data *= 255 / data.max()
    
    from scipy.stats.mstats import mquantiles
    q = mquantiles(data.flatten(),[0.7,0.98])
    q[0]=max(q[0],1)
    q[1] = max(q[1],1)
    tf=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],1,1,1,0.5],[data.max(),1,1,1,1]]

    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(),color_factor=1)
    for a in actor_list:
    # assign actor to the renderer
        renderer.AddActor(a)
        
        
    data1 = sitk.GetArrayFromImage(  ventricle_zone).astype('float') # sitk.GetArrayFromImage( img ).astype('float') # numpy array
    data1 *= 255 / data1.max()
    
    from scipy.stats.mstats import mquantiles
    q1 = mquantiles(data1.flatten(),[0.7,0.98])
    q1[0]=max(q1[0],1)
    q1[1] = max(q1[1],1)
    tf1=[[0,0,0,0,0],[q1[0],0,0,0,0],[q1[1],1,1,1,0.5],[data1.max(),1,1,1,1]]

    actor_list1 = volumeRender(data1, tf=tf1, spacing=img.GetSpacing(),color_factor=0.5)
    for a1 in actor_list1:
    # assign actor to the renderer
        renderer.AddActor(a1)
    renderWindow.SetWindowName(filename);
    renderer.ResetCamera()
    renderer.SetBackground(colors.GetColor3d("SlateGray"))
    
    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()
    renderer.RemoveAllViewProps()
    renderer.ResetCamera()
def call_for_all_files(RAW_DATA_FOLDER,grayscale_suffix,masksuffix,betsuffix):
    dict_for_csv=[]
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
    counter=0
    seg_explicit_thresholds=[]
    unique_names_files_pattern.sort()
    ventricle_zone=[]
    sulcus_zone=[]
    for each_unique_names_file_pattern in unique_names_files_pattern:
        if counter<1: #'SAH_15_02282014_1137'  in each_unique_names_file_pattern : #counter<1:
#            counter=counter+1
            each_unique_names_file = glob.glob(RAW_DATA_FOLDER+ "/" +each_unique_names_file_pattern+ "*.nii.gz")
            for each_unique_file in each_unique_names_file:
                each_unique_file=os.path.basename(each_unique_file)
                each_unique_names_file_split=each_unique_file[:-7]
                if betsuffix in each_unique_names_file_split: # and "Biogen_22960_09192014_1947" in each_unique_names_file_split:
#                if  "levelset_mask" not in each_unique_names_file_split  and "seg" not in each_unique_names_file_split  and "levelset_bet" not in each_unique_names_file_split and "unet" not in  each_unique_names_file_split and "final" not in each_unique_names_file_split and "Infarct" not in each_unique_names_file_split: 
    #                print(each_unique_names_file_split)
                    filename_gray=each_unique_names_file_split[:-len(betsuffix)]+ grayscale_suffix+  ".nii.gz"
                    filename_mask=each_unique_names_file_split[:-len(betsuffix)]+ masksuffix + ".nii.gz" # "_unet.nii.gz" #each_unique_names_file_split[:-9]+ "_final_seg.nii.gz"
                    filename_bet=each_unique_file
                    print(filename_gray)
                    print(filename_mask)
#                    input("Press the <ENTER> key to continue....")
                    if os.path.exists(os.path.join(RAW_DATA_FOLDER,filename_gray)) and os.path.exists(os.path.join(RAW_DATA_FOLDER,filename_mask)):
                        latexfilename=os.path.join("./",filename_gray[:-7]+".tex")
                        file1=latex_start(latexfilename)
                        latex_start(latexfilename)
                        sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent = divideintozones(latexfilename,RAW_DATA_FOLDER,each_unique_names_file_pattern,filename_gray,filename_mask,filename_bet)
                        
                        latex_start_table2c(latexfilename)
                        latex_inserttext_table2c(latexfilename,text1='SulciVol:', text2=str(sulci_vol))
                        latex_insert_line(latexfilename,text='\\\\')
                        latex_inserttext_table2c(latexfilename,text1='VentricleVol:', text2=str(ventricle_vol))
                        latex_insert_line(latexfilename,text="\\\\")
                        latex_inserttext_table2c(latexfilename,text1='SulciVolAboveVent:', text2=str(sulci_vol_above_vent))
                        latex_insert_line(latexfilename,text="\\\\")
                        latex_inserttext_table2c(latexfilename,text1='SulciVolBelowVent:', text2=str(sulci_vol_below_vent))
                        latex_insert_line(latexfilename,text="\\\\")
                        latex_inserttext_table2c(latexfilename,text1='SulciVolAtVent:', text2=str(sulci_vol_at_vent))                        
                        latex_end_table2c(latexfilename)
                        
                        
                        
                        latex_end(latexfilename)

                        latex_file_build(latexfilename)
                        this_dict={"Subject": each_unique_names_file_split[:-len(betsuffix)],"Sulci_VolTotal":sulci_vol,"Ventricles_Vol":ventricle_vol,"Sulci_VolL":leftcountsul,"Sulci_VolR":rightcountsul,"Ventricles_VolL":leftcountven,"Ventricles_VolR":rightcountven,"sulci_vol_above_vent": sulci_vol_above_vent,"sulci_vol_below_vent" :sulci_vol_below_vent,"sulci_vol_at_vent":sulci_vol_at_vent}
                        dict_for_csv.append(this_dict)
    csv_filename=os.path.basename(RAW_DATA_FOLDER)
    csvfile_with_vol=csv_filename+'.csv'
    csv_columns=['Subject','Sulci_VolTotal','Ventricles_Vol','Sulci_VolL','Sulci_VolR','Ventricles_VolL','Ventricles_VolR','sulci_vol_above_vent','sulci_vol_below_vent','sulci_vol_at_vent']
    write_csv(csvfile_with_vol,csv_columns,dict_for_csv)
    command = "rm  *.jpg"
    subprocess.call(command,shell=True)
    command = "rm  *.aux"
    subprocess.call(command,shell=True)
    command = "rm  *.log"
    subprocess.call(command,shell=True)
    command = "rm  *.nav"
    subprocess.call(command,shell=True)
    command = "rm  *.out"
    subprocess.call(command,shell=True)  
    command = "rm  *.snm"
    subprocess.call(command,shell=True)
    command = "rm  *.toc"
    subprocess.call(command,shell=True)
#
#
#
#
#
##plot_imagein3d()
##        print(each_unique_names_file_split)
##        if "257" not in each_unique_names_file_split:
#        #
#            
##WUSTL_257_02022016_0344_Head_3.0_H31s_final_seg_final_seg.nii.gz is not recognized as a NIFTI file
#
##WUSTL_257_02022016_0344_Head_3.0_H31s_final_seg.nii.gz
    

def start_renderer():
    filename=os.path.join(RAW_DATA_FOLDER,"SAH_2_01062014_1205_Head_w_o_4.8_Brain.nii") #"Biogen_22960_09192014_1947_scan15652_CT1_levelset.nii.gz")
    ventricle_zone, sulcus_zone = divideintozones(RAW_DATA_FOLDER,"SAH_2_01062014_1205_Head_w_o_4.8_Brain")
#    xx=nib.load(filename)
#    xx_data=xx.get_fdata()
###    plot3d_plane_image(xx_data[:,:,int(xx_data.shape[2]/2):xx_data.shape[2]]*255)
###    plot3d_plane_image(xx_data*255)
##    
##    binary_image=left_right_csf(xx_data*255)
#    binary_image=xx_data*255
#    binary_image=left_right_csf(binary_image)
#    draw_imageplanes_vtk(binary_image,renderer)
#    center=[0,0,10]
#    normal=[0,1,0]
#    draw_perpendicular_plane_eachslice(xx_data,center,normal,renderer)
    img = sitk.ReadImage( filename ) # SimpleITK object
    data = sitk.GetArrayFromImage( sulcus_zone ).astype('float') # sitk.GetArrayFromImage( img ).astype('float') # numpy array
    data *= 255 / data.max()
    
    from scipy.stats.mstats import mquantiles
    q = mquantiles(data.flatten(),[0.7,0.98])
    q[0]=max(q[0],1)
    q[1] = max(q[1],1)
    tf=[[0,0,0,0,0],[q[0],0,0,0,0],[q[1],1,1,1,0.5],[data.max(),1,1,1,1]]
    color_factor=1
    actor_list = volumeRender(data, tf=tf, spacing=img.GetSpacing(),color_factor=1)
    for a in actor_list:
    # assign actor to the renderer
        renderer.AddActor(a)
        
        
    data1 = sitk.GetArrayFromImage(  ventricle_zone).astype('float') # sitk.GetArrayFromImage( img ).astype('float') # numpy array
    data1 *= 255 / data1.max()
    
    from scipy.stats.mstats import mquantiles
    q1 = mquantiles(data1.flatten(),[0.7,0.98])
    q1[0]=max(q1[0],1)
    q1[1] = max(q1[1],1)
    tf1=[[0,0,0,0,0],[q1[0],0,0,0,0],[q1[1],1,1,1,0.5],[data1.max(),1,1,1,1]]
    color_factor=0.5
    actor_list1 = volumeRender(data1, tf=tf1, spacing=img.GetSpacing(),color_factor=0.5)
    for a1 in actor_list1:
    # assign actor to the renderer
        renderer.AddActor(a1)
    
    renderer.ResetCamera()
    renderer.SetBackground(colors.GetColor3d("SlateGray"))
    
    # Render and interact

    renderWindow.Render()
    renderWindowInteractor.Start()
#def draw_perp_planes():
#    center=[0,0,0]
#    normal=[0,0,1]
#    draw_perpendicular_plane_oneslice1(center,normal,"Z",renderer)
#    normal=[0,1,0]
#    draw_perpendicular_plane_oneslice1(center,normal,"Y",renderer)
#    normal=[1,0,0]
#    draw_perpendicular_plane_oneslice1(center,normal,"X",renderer)


def call_call_for_all_files(RAW_DATA_FOLDER,grayscale_suffix,masksuffix,betsuffix):
#    RAW_DATA_FOLDER="/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/DATA/SAH1_outputs" #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CSF_SEGMENTATION" # 
#    grayscale_suffix="_levelset"
#    masksuffix="_unet" #"_final_seg" #
#    betsuffix="_levelset_bet"
#
###draw_perp_planes()
###planefit_lsq()
##start_renderer()
##filename="test.tex"
##file1=latex_start(latexfilename)
##latex_start(latexfilename)
##frame_start(latexfilename,frametitle="ATUL")
    call_for_all_files(RAW_DATA_FOLDER,grayscale_suffix,masksuffix,betsuffix)
#latex_insert_text(latexfilename,text="ATUL KUMAR",texttype="item")
#latex_insert_image(latexfilename,image="testplot.jpg",caption="ATUL",imagescale=0.3)
#frame_end(latexfilename)
#latex_end(latexfilename)
#latex_file_build(latexfilename)
    
    
def divideintozones_v1(latexfilename,SLICE_OUTPUT_DIRECTORY,filename_gray,filename_mask,filename_bet):
    sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image

    file_gray = filename_gray #os.path.join(RAW_DATA_FOLDER,filename_gray)
    reader_gray = sitk.ImageFileReader()
    reader_gray.SetImageIO("NiftiImageIO")
    reader_gray.SetFileName(file_gray)
    filename_gray_data = reader_gray.Execute();
    
    gray_scale_file=filename_gray #os.path.join(RAW_DATA_FOLDER,filename_gray)
    gray_image=nib.load(gray_scale_file)
    resol= np.prod(np.array(gray_image.header["pixdim"][1:4]))
    affine_1=gray_image.affine
    gray_image_data=gray_image.get_fdata()

    file =filename_mask # os.path.join(RAW_DATA_FOLDER,filename_mask)
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    img_T1_Copy=img_T1
    imagenparray=sitk.GetArrayFromImage(img_T1)

    initial_seed_point_indexes1=[]
    if np.sum(imagenparray)>200:
        img_T1=img_T1*255
        # ## Read Data and Select Seed Point(s)
        # 
        # We first load a T1 MRI brain scan and select our seed point(s). If you are unfamiliar with the anatomy you can use the preselected seed point specified below, just uncomment the line.
        
        ######## In[13]:
        # img_T1 = sitk.ReadImage(fdata("nac-hncma-atlas2013-Slicer4Version/Data/A1_grayT1.nrrd"))
        # Rescale the intensities and map them to [0,255], these are the default values for the output
        # We will use this image to display the results of segmentation
        img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1) ,sitk.sitkUInt8)

        file1 = filename_bet # os.path.join(RAW_DATA_FOLDER,filename_bet)
        reader1 = sitk.ImageFileReader()
        reader1.SetImageIO("NiftiImageIO")
        reader1.SetFileName(file1)
        img_T1_bet = reader1.Execute();
        cc1 = sitk.ConnectedComponent(img_T1_bet>0)
        stats1 = sitk.LabelIntensityStatisticsImageFilter()
        stats1.Execute(cc1,img_T1_bet)
#        
        cc = sitk.ConnectedComponent(img_T1_255>0)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(cc,img_T1)
        maxsize_comp=0
        id_of_maxsize_comp=0
        maxsize_comp_1=0
        id_of_maxsize_comp_1=0
        bet_ids=[]
        for l in range(len(stats1.GetLabels())):  
            if stats1.GetPhysicalSize(stats1.GetLabels()[l])>maxsize_comp_1:
                maxsize_comp_1=stats1.GetPhysicalSize(stats1.GetLabels()[l])

                id_of_maxsize_comp_1=l
        csf_ids=[]
        for l in range(len(stats.GetLabels())):  

            csf_ids.append([l,stats.GetPhysicalSize(stats.GetLabels()[l])])
        csf_ids.sort(key = sortSecond, reverse = True)

        first_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[0][0]]))
        second_seg_centroid=np.array(stats.GetCentroid(stats.GetLabels()[csf_ids[1][0]]))
        bet_centroid=np.array(stats.GetCentroid(stats.GetLabels()[id_of_maxsize_comp_1]))
        first2bet_centroid=np.linalg.norm(first_seg_centroid - bet_centroid)
        second2bet_centroid=np.linalg.norm(second_seg_centroid - bet_centroid)
        if first2bet_centroid< second2bet_centroid:
            id_of_maxsize_comp=csf_ids[0][0]

        else:
            if stats.GetPhysicalSize(stats.GetLabels()[csf_ids[1][0]]) > 10000: 
                id_of_maxsize_comp=csf_ids[1][0]

            else:
                id_of_maxsize_comp=csf_ids[0][0]

        initial_seed_point_indexes=[stats.GetMinimumIndex(stats.GetLabels()[id_of_maxsize_comp])]
        seg_explicit_thresholds = sitk.ConnectedThreshold(img_T1, seedList=initial_seed_point_indexes, lower=100, upper=255)

        zoneV_min_z,zoneV_max_z=get_ventricles_range(sitk.GetArrayFromImage(seg_explicit_thresholds))
        subtracted_image=subtract_binary_1(sitk.GetArrayFromImage(img_T1_Copy),sitk.GetArrayFromImage(seg_explicit_thresholds)*255)
        subtracted_image=sitk.GetImageFromArray(subtracted_image)
        above_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
        above_ventricle_image[0:zoneV_max_z+1,:,:]=0
        covering_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
        covering_ventricle_image[0:zoneV_min_z+1,:,:]=0
        covering_ventricle_image[zoneV_max_z+1:above_ventricle_image.shape[0],:,:]=0
        below_ventricle_image= sitk.GetArrayFromImage(subtracted_image)
        below_ventricle_image[zoneV_min_z:above_ventricle_image.shape[0],:,:]=0

        sulci_vol=calculate_volume(gray_image,sitk.GetArrayFromImage(subtracted_image)) 
        ventricle_vol=calculate_volume(gray_image,sitk.GetArrayFromImage(seg_explicit_thresholds))
        sulci_vol_above_vent=calculate_volume(gray_image,above_ventricle_image)
        sulci_vol_below_vent=calculate_volume(gray_image,below_ventricle_image)
        sulci_vol_at_vent=calculate_volume(gray_image,covering_ventricle_image)
        allinone=np.zeros(below_ventricle_image.shape)
        allinone[below_ventricle_image>0]=100
        allinone[above_ventricle_image>0]=180
        allinone[sitk.GetArrayFromImage(seg_explicit_thresholds)>0]=240
        allinone[covering_ventricle_image>0]=255        

        image_slice_jpg_dir=SLICE_OUTPUT_DIRECTORY #"/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/CSF_Compartment/RESULTS/IMAGES"
        filename_gray_data_np= exposure.rescale_intensity(slicenum_at_end(sitk.GetArrayFromImage(filename_gray_data)), in_range=(1000, 1200))
        combine_masks_as_color_v2(latexfilename,filename_gray_data_np,slicenum_at_end(allinone),image_slice_jpg_dir,filename_gray[:-7]) #(allmasks)

    return sulci_vol, ventricle_vol,leftcountven*resol,rightcountven*resol,leftcountsul*resol,rightcountsul*resol,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent #seg_explicit_thresholds, subtracted_image
