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
#
#
#
# def combine_masks_as_color(several_masks, image=np.random.rand(10,10,3)):
#     if len(several_masks)==0:
#         several_masks=[]
#         several_masks.append(image)
#         image=np.random.rand(10,10,3)
#         several_masks.append(image)
#         image=np.random.rand(10,10,3)
#         several_masks.append(image)
#         image=np.random.rand(10,10,3)
#         several_masks.append(image)
#         image=np.random.rand(10,10,3)
#         several_masks.append(image)
#     np.random.seed(0)
#     color_factor=np.random.rand(len(several_masks),3)
#
#     for i in range(len(several_masks)):
#         print("len(several_masks)")
#         print(len(several_masks))
#
#         this_image=several_masks[i]
#         print("this_image")
#         print(this_image.shape)
#     for k in range(this_image.shape[2]):
#         one_rgb_image=np.zeros([this_image.shape[0],this_image.shape[1],3])
#         one_rgb_image[this_image[:,:,k]>0,0]=color_factor[i,0] #color_factor[i]
#         one_rgb_image[this_image[:,:,k]>0,1]=color_factor[i,1] #color_factor[i]
#         one_rgb_image[this_image[:,:,k]>0,2]=color_factor[i,2] #color_factor[i]
#     filename=str(k)  + ".jpg"
#     save_matplot(one_rgb_image,filesaveas=filename)
#
# def combine_masks_as_color_1(latexfilename,gray_image,several_masks,directoryname="../RESULTS/IMAGES/", niifilename="niifilename", image=np.random.rand(10,10,3)):
# #    if len(several_masks)==0:
# #        several_masks=[]
# #        several_masks.append(image)
# #        image=np.random.rand(10,10,3)
# #        several_masks.append(image)
# #        image=np.random.rand(10,10,3)
# #        several_masks.append(image)
# #        image=np.random.rand(10,10,3)
# #        several_masks.append(image)
# #        image=np.random.rand(10,10,3)
# #        several_masks.append(image)
#     thisfile=niifilename
#     thisfile=thisfile.split('.')
#     thisfile=thisfile[0].split('_')
#     thisfileuniqueid=thisfile[0] +  thisfile[1] +  thisfile[2] + thisfile[3]
#     np.random.seed(0)
#     num_class_thresh= np.unique(several_masks)
#     color_factor=np.random.rand(len(num_class_thresh),3)
#
#     counter=0
#
#     for k in range(several_masks.shape[2]):
#
# #            frame_start(latexfilename,frametitle="ATUL")
#         one_rgb_image=np.zeros([several_masks.shape[0],several_masks.shape[1],3])
#         gray_rgb_image=np.zeros([several_masks.shape[0],several_masks.shape[1],3])
#
#         for i in range(len(num_class_thresh)):
#             if num_class_thresh[i]==0:
#                 print("color_factor")
#                 print(num_class_thresh)
#                 one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],0]=0 #color_factor[i]
#                 one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],1]=0 #color_factor[i]
#                 one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],2]=0 #color_factor[i]
# #                gray_rgb_image[:,:,0]=gray_image[:,:,0]
# #                gray_rgb_image[:,:,1]=gray_image[:,:,1]
# #                gray_rgb_image[:,:,2]=gray_image[:,:,2]
#             else:
#                 one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],0]=color_factor[i,0] #color_factor[i]
#                 one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],1]=color_factor[i,1] #color_factor[i]
#                 one_rgb_image[several_masks[:,:,k]==num_class_thresh[i],2]=color_factor[i,2] #color_factor[i]
#         gray_rgb_image[:,:,0]=gray_image[:,:,k]
#         gray_rgb_image[:,:,1]=gray_image[:,:,k]
#         gray_rgb_image[:,:,2]=gray_image[:,:,k]
#         filename=thisfileuniqueid+"slice" + str(k)  + ".jpg"
#         filename_gray=thisfileuniqueid+"grayslice" + str(k)  + ".jpg"
#         plot_im(one_rgb_image,filesaveas=os.path.join(directoryname,filename))
#         plot_im(gray_rgb_image,filesaveas=os.path.join(directoryname,filename_gray))
#         latex_start_table2c(latexfilename)
#         latex_insertimage_table2c(latexfilename,image1=os.path.join(directoryname,filename_gray), image2=os.path.join(directoryname,filename),caption="ATUL",imagescale=0.4)
#         latex_end_table2c(latexfilename)
# #        latex_include_image(latexfilename,image_dir="../RESULTS/IMAGES/", image=filename,caption=filename,imagescale=0.1)
# #        latex_insert_image(latexfilename,image=filename,caption=filename,imagescale=0.35)
# #    latex_end_image(latexfilename)
# #    frame_end(latexfilename)
# #        if k%10==0 and k>0:
# #            frame_end(latexfilename)
#
#
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
#
# def save_matplot(image=np.ones([10,10,3]),filesaveas="testplot.jpg"):
#     fig=plot_im(image, dpi=80)
# #    plt.figure(figsize=(25,25))
# #    plt.axis("off")
# #
# #    plt.imshow(image)
# ##    plt.show()
# ##    fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
# ##    fig.subplots_adjust(hspace = .5, wspace=.001)
# ##
# ##    axs = axs.ravel()
# ##
# ##    for i in range(10):
# ##
# ##        axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
# ##        axs[i].set_title(str(250+i))
#     fig.savefig(filesaveas, format='jpg', dpi=400)
# #    plt.close()
#     return "AA"
#
# def save_allslices_inone(image=np.ones([512,512,30]),filesaveas="testplot.jpg"):
#
#     fig, axs = plt.subplots(int(image.shape[2]/5),5, figsize=(150, 60), facecolor='w', edgecolor='k')
#
# #    fig.subplots_adjust(hspace = 0, wspace=0)
#
#     axs = axs.ravel()
#
#     for i in range(image.shape[2]):
#
# #        axs[i].contourf(np.random.rand(10,10),5,cmap=plt.cm.Oranges)
#         axs[i].imshow(image[:,:,i]) # (str(250+i))
#         axs[i].set_title(str(i))
#         axs[i].axis("off")
#         axs[i].set_xticklabels([])
#         axs[i].set_yticklabels([])
#     plt.tight_layout()
#     plt.savefig(filesaveas)
#     return "AA"
#
#
# def subtract_binary(binary_imageBig,binary_imageSmall):
#     print("I am here")
#     binary_image_non_zero_cord_t=np.nonzero(binary_imageSmall>0)
# #    print(binary_image_non_zero_cord_t)
#     binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
#     print("binary_image_non_zero_cord")
#     print(binary_image_non_zero_cord)
# #    binary_imageBig[binary_image_non_zero_cord]=0
# #    print(binary_image_non_zero_cord_t)
#     for each_point in binary_image_non_zero_cord:
#         binary_imageBig[each_point[0],each_point[1],each_point[2]]=0
# #        print("right")
# #        if multiply_with_plane<0:
# #            left_cords.append(each_point)
# #            print("left")
# #    print(right_cords)
# #    binary_image[right_cords]=0
# #    plot_image_slice(binary_image)
#     return binary_imageBig
#
def subtract_binary_1(binary_imageBig,binary_imageSmall):
    binary_imageBigCopy=np.copy(binary_imageBig)
    binary_imageBigCopy[binary_imageSmall>0]=0
    return binary_imageBigCopy


# def planefit_lsq(data):
# #    mean = np.array([0.0,0.0,0.0])
# #    cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
# #    data = np.random.multivariate_normal(mean, cov, 50)
# #    data=data[0:6]
#     #for p in data:
#     #    draw_point_call(p)
#     # regular grid covering the domain of the data
#     X,Y = np.meshgrid(range(512), range(512))
#     XX = X.flatten()
#     YY = Y.flatten()
#
#     order = 1    # 1: linear, 2: quadratic
#     if order == 1:
#         # best-fit linear plane
#         A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
#         C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
#         # evaluate it on grid
#         Z = C[0]*X + C[1]*Y + C[2]
#         # or expressed using matrix/vector product
#         #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
#         Z1=Z.flatten()
#         plane_cords=np.zeros((len(Z1),3))
#         plane_cords[:,0]=XX
#         plane_cords[:,1]=YY
#         plane_cords[:,2]=Z1
#     #    for p in plane_cords:
#     #        draw_point_call(p)
# #        print(plane_cords[0])
#         pointA=plane_cords[0]
#         pointB=plane_cords[int(len(plane_cords)/2)]
#         pointC=plane_cords[len(plane_cords)-1]
#         AB=pointB-pointA
#         AC=pointC-pointA
#         A_normal=np.cross([AC[0],AC[1],AC[2]],[AB[0],AB[1],AB[2]])
#         normal_plane=np.cross(AB,AC)
#
#         normal_plane_unit = normal_plane/np.sqrt(np.sum(normal_plane*normal_plane))
#         print(AB)
#         print(AC)
#         print(normal_plane_unit)
# #        print(normal_plane)
#         actor=draw_plane(pointA,normal_plane_unit)
# #        actor=draw_plane([0,0,0],[0,0,1])
#         renderer.AddActor(actor)
#     #    draw_perpendicular_plane_withpoints(plane_cords[0:4],renderer)
#
#     elif order == 2:
#         # best-fit quadratic curve
#         A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
#         C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
#
#         # evaluate it on a grid
#         Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
#
# def draw_point_call(points):
#     actor1= draw_points(points)
#     renderer.AddActor(actor1)
#
#
#
# def draw_perpendicular_plane_eachslice(center,normal,renderer):
#     for img_idx in range(15,20): #numpy_image.shape[2]):
# #        center=[0,0,img_idx]
# #        normal=[0,1,0]
#         actor1= draw_plane(center,normal)
#         renderer.AddActor(actor1)
# def draw_perpendicular_plane_oneslice(center,normal,renderer):
#     actor1= draw_plane(center,normal)
#     renderer.AddActor(actor1)
#     return "YY"
# def draw_perpendicular_plane_oneslice1(center,normal,N,renderer):
#     actor1= draw_plane(center,normal,N)
#     renderer.AddActor(actor1)
#     return "YY"
# def draw_perpendicular_plane_withpoints(points,renderer):
#     actor1= draw_plane_with_points(points)
#     renderer.AddActor(actor1)
#     return "YY"
#
# def draw_imageplanes_vtk(numpy_image,renderer,slice_range1=0,slice_range2=0):
#     for img_idx in range(0,numpy_image.shape[2]): #range(15,20): #
#         if img_idx >=slice_range1 and img_idx<=slice_range2:
#             actor=image_plane_vtk(numpy_image[:,:,img_idx],img_idx, rgbFrame=None)
#             renderer.AddActor(actor)
#     return "ZZ"
# def draw_imageplanes_vtk_1(numpy_image,renderer,slice_range=1):
#     for img_idx in range(0,numpy_image.shape[0]): #numpy_image.shape[2]):
#         actor=image_plane_vtk(numpy_image[img_idx,:,:],img_idx, rgbFrame=None)
#         renderer.AddActor(actor)
#     return "ZZ"
# def draw_imageplanes_vtk_2(numpy_image,img_idx,renderer,slice_range=1):
# #    for img_idx in range(0,numpy_image.shape[0]): #numpy_image.shape[2]):
#     actor=image_plane_vtk(numpy_image,img_idx, rgbFrame=None)
#     renderer.AddActor(actor)
#     return "ZZ"
# def plot_image_slice(image_data):  ## slice number in the back
#     fig = plt.figure()
#     # Add an axes
#     ax = fig.add_subplot(111,projection='3d')
#     xx, yy = np.meshgrid(range(512), range(512))
#     for slice_z in range(0,image_data.shape[2]):
#         ax.contourf(xx, yy, image_data[:,:,slice_z], zdir='z', offset=slice_z, cmap=cm.gray,alpha=0.7)
#     plt.show()
#     return "XX"
#
# def convert_imageslice_3Dpoints(image_data):
#
#     return "XX"
# def left_right_csf_itk(binary_image,plane_normal=np.array([1,0,0]),plane_point=np.array([256,256,10])):
#     print("I am here")
#     binary_image_non_zero_cord_t=np.nonzero(binary_image>0)
#     print(binary_image_non_zero_cord_t)
#     binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
#     print(binary_image_non_zero_cord_t)
#     left_cords=[]
#     right_cords=[]
#     for each_point in binary_image_non_zero_cord:
#         multiply_with_plane=np.dot(np.subtract(np.array(each_point),plane_point),plane_normal) #np.dot(np.array(each_point),plane_normal) + plane_point
#         if multiply_with_plane>0:
#             right_cords.append(each_point)
#             binary_image[each_point[0],each_point[1],each_point[2]]=0
#             print("right")
# #        if multiply_with_plane<0:
# #            left_cords.append(each_point)
# #            print("left")
# #    print(right_cords)
# #    binary_image[right_cords]=0
# #    plot_image_slice(binary_image)
#     return binary_image
#
#
# def left_right_csf(binary_image,plane_normal=np.array([1,0,0]),plane_point=np.array([256,256,10])):
# #    print("I am here")
# #    plane_normal=np.array([1,0,0])
# #    plane_point=np.array([256,256,10])
#     binary_image_copy=np.copy(binary_image)
#     planeA = Plane(Point3D(plane_point[0],plane_point[1],plane_point[2]), normal_vector=(plane_normal[0],plane_normal[1],plane_normal[2]))
#
#     binary_image_non_zero_cord_t=np.nonzero(binary_image_copy>0)
#
#     binary_image_non_zero_cord=np.transpose(binary_image_non_zero_cord_t)
#     binary_image_non_zero_cord_copy=np.copy(binary_image_non_zero_cord)
#     binary_image_non_zero_cord_copy[:,0]=binary_image_non_zero_cord[:,1]
#     binary_image_non_zero_cord_copy[:,1]=binary_image_non_zero_cord[:,0]
# #    binary_image_non_zero_cord=np.copy(binary_image_non_zero_cord_copy)
# ##    print(binary_image_non_zero_cord.shape)
#     left_cords=[]
#     right_cords=[]
#     count=0
#     leftcount=0
#     rightcount=0
#     for each_point in binary_image_non_zero_cord:
#
# #        multiply_with_plane=np.dot(np.subtract(np.array(each_point),plane_point),plane_normal) #np.dot(np.array(each_point),plane_normal) + plane_point
# #        print(multiply_with_plane)
# #        if multiply_with_plane>0:
# #        print("planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) ")
# #        print(planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) )
#         dist_p=planeA.distance(Point3D(each_point[1],each_point[0],each_point[2]))
#         x= np.float(str(N(dist_p))) #planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) )))
# #        print(x)
#         if not re.search('[a-zA-Z]', str(N(planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) ))):
# #        if  x>0:
#             yy=np.float(str(N(planeA.equation(x=each_point[1], y=each_point[0],z=each_point[2]) )))
#             if  yy > 0:
#     #            right_cords.append(each_point)
#     #            print("before")
#     #            print(binary_image_copy[each_point[0],each_point[1],each_point[2]])
#                 binary_image_copy[each_point[0],each_point[1],each_point[2]]=100
#     #            print("after")
#     #            print(binary_image_copy[each_point[0],each_point[1],each_point[2]])
#                 leftcount=leftcount+1
#     #            print("right")
#     #        if multiply_with_plane<0:
#             else:
#     #            left_cords.append(each_point)
#                 rightcount=rightcount+1
#     #            binary_image_copy[each_point[0],each_point[1],each_point[2]]=0
#     #            print("left")
#     ##    print(right_cords)
#     ##    binary_image[right_cords]=0
#     ##    plot_image_slice(binary_image)
#     #    print('count')
#     #    print(count)
#     return binary_image_copy,leftcount,rightcount
# def plot3d_plane_image(image_data,point=np.array([256, 256, 5]),normal=np.array([0,0,512])):
# #    point  = plane_point #np.array([1, 2, 3])
# #    normal = plane_normal #np.array([1, 1, 2])
# #    point2 = np.array([10, 50, 50])
#     # a plane is a*x+b*y+c*z+d=0
#     # [a,b,c] is the normal. Thus, we have to calculate
#     # d and we're set
#     print("image size")
#     print(image_data.shape)
#     d = -point.dot(normal)
# #    d1 = -p1.dot(n1)
#     # create x,y
#     xx, yy = np.meshgrid(range(512), range(512))
#     # calculate corresponding z
#     z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
#
# #    z1 = (-n1[0] * xx - n1[1] * yy - d1) * 1. /n1[2]
# ##    # plot the surface
# ##    plt3d = plt.figure().gca(projection='3d')
# ##    plt3d.plot_surface(xx, yy, z, alpha=0.2)
# ##    #and i would like to plot this point :
# ##    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
# #    # plot the surface
# #    plt3d = plt.figure().gca(projection='3d')
# #    plt3d.plot_surface(xx, yy, z, alpha=0.2)
# #    # Ensure that the next plot doesn't overwrite the first plot
# #    ax = plt.gca()
# #    ax.scatter(point2[0], point2[1], point2[2], color='green')
# #    plt.show()
#     # Create the figure
#     fig = plt.figure()
#     # Add an axes
#     ax = fig.add_subplot(111,projection='3d')
#     # plot the surface
# #    ax.plot_surface(xx, yy, z, alpha=0.5)
#     for slice_z in range(int(image_data.shape[2]/2),image_data.shape[2]):
#         ax.contourf(xx, yy, image_data[:,:,slice_z], 100, zdir='z', offset=slice_z, cmap=cm.gray,alpha=0.8)
# #    ax.contourf(xx, yy, image_data[slice_z+1,:,:], 100, zdir='z', offset=slice_z+1, cmap=cm.gray,alpha=0.6)
#     # and plot the point
# #    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
#     plt.show()
#
# def plot3d_plane(image_data,slice_z,point=np.array([1, 2, 3]),normal=np.array([1, 1, 2]),p1=np.array([1, 2, 3]),n1=np.array([1, 1, 2]),point2 = np.array([10, 50, 50])):
# #    point  = plane_point #np.array([1, 2, 3])
# #    normal = plane_normal #np.array([1, 1, 2])
# #    point2 = np.array([10, 50, 50])
#     # a plane is a*x+b*y+c*z+d=0
#     # [a,b,c] is the normal. Thus, we have to calculate
#     # d and we're set
#     print("image size")
#     print(image_data.shape)
#     d = -point.dot(normal)
#     d1 = -p1.dot(n1)
#     # create x,y
#     xx, yy = np.meshgrid(range(512), range(512))
#     # calculate corresponding z
#     z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
#
#     z1 = (-n1[0] * xx - n1[1] * yy - d1) * 1. /n1[2]
# ##    # plot the surface
# ##    plt3d = plt.figure().gca(projection='3d')
# ##    plt3d.plot_surface(xx, yy, z, alpha=0.2)
# ##    #and i would like to plot this point :
# ##    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
# #    # plot the surface
# #    plt3d = plt.figure().gca(projection='3d')
# #    plt3d.plot_surface(xx, yy, z, alpha=0.2)
# #    # Ensure that the next plot doesn't overwrite the first plot
# #    ax = plt.gca()
# #    ax.scatter(point2[0], point2[1], point2[2], color='green')
# #    plt.show()
#     # Create the figure
#     fig = plt.figure()
#     # Add an axes
#     ax = fig.add_subplot(111,projection='3d')
#     # plot the surface
#     ax.plot_surface(xx, yy, z, alpha=0.5)
# #    ax.plot_surface(xx, yy, z1, alpha=0.8)
# #    for i in range(0,image_data.shape[0]):
#     ax.contourf(xx, yy, image_data[slice_z,:,:], 100, zdir='z', offset=slice_z, cmap=cm.gray,alpha=0.7)
#     ax.contourf(xx, yy, image_data[slice_z-1,:,:], 100, zdir='z', offset=slice_z-1, cmap=cm.gray,alpha=0.8)
#     ax.contourf(xx, yy, image_data[slice_z+1,:,:], 100, zdir='z', offset=slice_z+1, cmap=cm.gray,alpha=0.6)
#     # and plot the point
# #    ax.scatter(point2[0] , point2[1] , point2[2],  color='green')
#     plt.show()
#
# def plane_to_single_slice():
#
#     return "XX"
#
#
# def plot_imagein3d():
#  #  Verify input arguments
# #    if len(argv) > 1:
#         # Read the image
# #    jpeg_reader = vtkJPEGReader()
# ##        if not jpeg_reader.CanReadFile(argv[1]):
# ##            print("Error reading file %s" % argv[1])
# ##            return
# #
# #    jpeg_reader.SetFileName(argv[1])
# #    jpeg_reader.Update()
# #    image_data = jpeg_reader.GetOutput()
# #    else:
#     canvas_source = vtkImageCanvasSource2D()
#     canvas_source.SetExtent(0, 100, 0, 100, 0, 0)
#     canvas_source.SetScalarTypeToUnsignedChar()
#     canvas_source.SetNumberOfScalarComponents(3)
#     canvas_source.SetDrawColor(127, 127, 100)
#     canvas_source.FillBox(0, 100, 0, 100)
#     canvas_source.SetDrawColor(100, 255, 255)
#     canvas_source.FillTriangle(10, 10, 25, 10, 25, 25)
#     canvas_source.SetDrawColor(255, 100, 255)
#     canvas_source.FillTube(75, 75, 0, 75, 5.0)
#     canvas_source.Update()
#     image_data = canvas_source.GetOutput()
#
#     # Create an image actor to display the image
#     image_actor = vtkImageActor()
#
#     if VTK_MAJOR_VERSION <= 5:
#         image_actor.SetInput(image_data)
#     else:
#         image_actor.SetInputData(image_data)
#
#     # Create a renderer to display the image in the background
#     background_renderer = vtkRenderer()
#
#     # Create a superquadric
#     superquadric_source = vtkSuperquadricSource()
#     superquadric_source.SetPhiRoundness(1.1)
#     superquadric_source.SetThetaRoundness(.2)
#
#     # Create a mapper and actor
#     superquadric_mapper = vtkPolyDataMapper()
#     superquadric_mapper.SetInputConnection(superquadric_source.GetOutputPort())
#
#     superquadric_actor = vtkActor()
#     superquadric_actor.SetMapper(superquadric_mapper)
#
#     scene_renderer = vtkRenderer()
#
#     render_window = vtkRenderWindow()
#
#     # Set up the render window and renderers such that there is
#     # a background layer and a foreground layer
#     background_renderer.SetLayer(0)
#     background_renderer.InteractiveOff()
#     scene_renderer.SetLayer(1)
#     render_window.SetNumberOfLayers(2)
#     render_window.AddRenderer(background_renderer)
#     render_window.AddRenderer(scene_renderer)
#
#     render_window_interactor = vtkRenderWindowInteractor()
#     render_window_interactor.SetRenderWindow(render_window)
#
#     # Add actors to the renderers
#     scene_renderer.AddActor(superquadric_actor)
#     background_renderer.AddActor(image_actor)
#
#     # Render once to figure out where the background camera will be
#     render_window.Render()
#
#     # Set up the background camera to fill the renderer with the image
#     origin = image_data.GetOrigin()
#     spacing = image_data.GetSpacing()
#     extent = image_data.GetExtent()
#
#     camera = background_renderer.GetActiveCamera()
#     camera.ParallelProjectionOn()
#
#     xc = origin[0] + 0.5*(extent[0] + extent[1]) * spacing[0]
#     yc = origin[1] + 0.5*(extent[2] + extent[3]) * spacing[1]
#     # xd = (extent[1] - extent[0] + 1) * spacing[0]
#     yd = (extent[3] - extent[2] + 1) * spacing[1]
#     d = camera.GetDistance()
#     camera.SetParallelScale(0.5 * yd)
#     camera.SetFocalPoint(xc, yc, 0.0)
#     camera.SetPosition(xc, yc, d)
#
#     # Render again to set the correct view
#     render_window.Render()
#
#     # Interact with the window
#     render_window_interactor.Start()
#
# def planeFit(points):
#     """
#     p, n = planeFit(points)
#
#     Given an array, points, of shape (d,...)
#     representing points in d-dimensional space,
#     fit an d-dimensional plane to the points.
#     Return a point, p, on the plane (the point-cloud centroid),
#     and the normal, n.
#     """
#
#     points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
#     assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
#     ctr = points.mean(axis=1)
#     x = points - ctr[:,np.newaxis]
#     M = np.dot(x, x.T) # Could also use np.cov(x) here.
#     return ctr, svd(M)[0][:,-1]
# def display_nii_vtk():
#
#     return "vtk"
#
# def calculate_plane(point1,point2,point3,point4):
#
#
#     return "plane"
#
#
# def draw_obb(labels,maskvalue):
#     labels = np.nan_to_num(labels)
#     mask = labels == maskvalue
#     labels_sum=np.sum(labels)
# #    print(labels_sum)
#     if labels_sum>10:
#
# #        print(np.isnan(labels).any())
# #        print(np.isinf(labels).any())
#
#
#         corners, centre = np_obb.get_obb_from_mask(mask)
#
#         sideA=np.linalg.norm(np.array([corners[0,1],corners[0,0]]) - np.array([corners[1,1],corners[1,0]]))
#         sideB=np.linalg.norm(np.array([corners[0,1],corners[0,0]]) - np.array([corners[3,1],corners[3,0]]))
#         difference= np.abs(sideA-sideB)
# #        print("difference")
# #        print(difference)
# #        print("sideA")
# #        print(sideA)
# #        print("sideB")
# #        print(sideB)
#         if difference <  sideA and difference < sideB:
#             obbs = np_obb.get_obb_from_labelim(labels)
#             fig = plt.figure(figsize=(8,8))
#             ax = fig.add_subplot(111)
# #            ax.imshow(labels)
#             for i in obbs.keys():
#                 corners, centre = obbs[i]
# ##                ax.scatter(centre[1],centre[0])
# #                ax.plot(corners[:,1],corners[:,0],'-')
# #                ax.scatter(corners[:,1],corners[:,0])
# ##                ax.scatter((corners[0,1]+ corners[1,1])/2,(corners[0,0]+corners[1,0])/2,marker='^')
# #                ax.scatter((corners[1,1]+ corners[2,1])/2,(corners[1,0]+corners[2,0])/2,marker='o')
# ##                ax.scatter((corners[2,1]+ corners[3,1])/2,(corners[2,0]+corners[3,0])/2,marker=r'$\clubsuit$')
# #                ax.scatter((corners[3,1]+ corners[0,1])/2,(corners[3,0]+corners[0,0])/2,marker='*')
# ##                plt.show()
# #                ## plane point 1, plane point 2,
#                 plane_point1=np.array([(corners[1,1]+ corners[2,1])/2,(corners[1,0]+corners[2,0])/2])
#                 plane_point2=np.array([(corners[3,1]+ corners[0,1])/2,(corners[3,0]+corners[0,0])/2])
#                 normalofboundingbox=np.cross(np.subtract(corners[1],corners[0]),np.subtract(corners[3],corners[0]))
#                 plane_point3=plane_point1 + 10 * (normalofboundingbox / (normalofboundingbox**2).sum()**0.5)
#                 required_plane_normal= np.cross(np.subtract(plane_point3,plane_point1),np.subtract(plane_point2,plane_point1))
#                 # normal of the bounding box
#                 # Third point on the plane
#                 # normal of the required plane
# #                print(type(corners))
#
#
#         return ((corners[1,1]+ corners[2,1])/2,(corners[1,0]+corners[2,0])/2), ((corners[3,1]+ corners[0,1])/2,(corners[3,0]+corners[0,0])/2 ) # , required_plane_normal,plane_point1)
#
# def loadniiwithsimpleitk(file):
# #    file = os.path.join(RAW_DATA_FOLDER,filename_mask)
#     reader = sitk.ImageFileReader()
#     reader.SetImageIO("NiftiImageIO")
#     reader.SetFileName(file)
#     image = reader.Execute();
#     return image
# def display_numpy_mat_bbox(image,bbox):
#     fig, ax = plt.subplots(num="MRI_demo")
#     circ = Circle((bbox[0],bbox[1]),10)
#     ax.add_patch(circ)
#     circ = Circle((bbox[2],bbox[3]),10)
#     ax.add_patch(circ)
#     circ = Circle((bbox[4],bbox[5]),10)
#     ax.add_patch(circ)
#     circ = Circle((bbox[6],bbox[7]),10)
#     ax.add_patch(circ)
#     ax.imshow(image, cmap="gray")
#     ax.axis('off')
#     plt.show()
#
# def display_numpy_mat_points(image,points):
#     fig, ax = plt.subplots(num="MRI_demo")
#     for point in points:
#         circ = Circle((point[0],point[1]),10)
#         ax.add_patch(circ)
#     ax.imshow(image, cmap="gray")
#     ax.axis('off')
#     plt.show()
# def display_numpy_mat(image):
#     fig, ax = plt.subplots(num="MRI_demo")
#     ax.imshow(image, cmap="gray")
#     ax.axis('off')
#     plt.show()
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
#
#
# RAW_DATA_FOLDER='/media/atul/AC0095E80095BA32/WASHU_WORK/PROJECTS/MIDLINE/DATA/CSF_SEGMENTATION/'
#
# def sortSecond(val):
#     return val[1]
#
#     resol= np.prod(np.array(nii_img.header["pixdim"][1:4]))
# #    print("header[pixdim][1:4]")
# #    print(nii_img.header["pixdim"][1:4])
#     mask_data_flatten= mask_img.flatten()
#     num_pixel_gt_0=mask_data_flatten[np.where(mask_data_flatten>0)]
# #    print(num_pixel_gt_0)
#     return (resol * num_pixel_gt_0.size)/1000
# #
# #
# #
#
#
#


    
    
def divideintozones_v1(latexfilename,SLICE_OUTPUT_DIRECTORY,filename_gray,filename_mask,filename_bet):
    sulci_vol, ventricle_vol,leftcountven,rightcountven,leftcountsul,rightcountsul,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent=(0,0,0,0,0,0,0,0,0) #seg_explicit_thresholds, subtracted_image

    file_gray = filename_gray
    reader_gray = sitk.ImageFileReader()
    reader_gray.SetImageIO("NiftiImageIO")
    reader_gray.SetFileName(file_gray)
    filename_gray_data = reader_gray.Execute();
    
    gray_scale_file=filename_gray
    gray_image=nib.load(gray_scale_file)
    resol= np.prod(np.array(gray_image.header["pixdim"][1:4]))


    file =filename_mask
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(file)
    img_T1 = reader.Execute();
    img_T1_Copy=img_T1
    imagenparray=sitk.GetArrayFromImage(img_T1)

    if np.sum(imagenparray)>200:
        img_T1=img_T1*255

        img_T1_255 = sitk.Cast(sitk.IntensityWindowing(img_T1) ,sitk.sitkUInt8)

        file1 = filename_bet
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

        maxsize_comp_1=0
        id_of_maxsize_comp_1=0

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

        image_slice_jpg_dir=SLICE_OUTPUT_DIRECTORY
        filename_gray_data_np= exposure.rescale_intensity(slicenum_at_end(sitk.GetArrayFromImage(filename_gray_data)), in_range=(1000, 1200))
        combine_masks_as_color_v2(latexfilename,filename_gray_data_np,slicenum_at_end(allinone),image_slice_jpg_dir,filename_gray[:-7])

    return sulci_vol, ventricle_vol,leftcountven*resol,rightcountven*resol,leftcountsul*resol,rightcountsul*resol,sulci_vol_above_vent,sulci_vol_below_vent,sulci_vol_at_vent #seg_explicit_thresholds, subtracted_image
