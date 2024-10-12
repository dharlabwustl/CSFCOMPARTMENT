import sys,inspect,subprocess
# import six
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import glob
import nibabel as nib
from utilities_simple import *
import os,subprocess,sys,glob
import numpy as np
import argparse
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay

import numpy as np
import scipy.ndimage as ndi
import numpy as np
import scipy.ndimage as ndi
from skimage import filters

import numpy as np
import scipy.ndimage as ndi
##############################
import numpy as np
import cv2
import numpy as np
from sklearn.decomposition import PCA
#########################################################
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from itertools import product

import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import rotate

from matplotlib.path import Path

def load_nii_mask(file_path):
    """
    Load a binary mask from a NIfTI (.nii) file using nibabel.
    """
    nii_img = nib.load(file_path)
    binary_mask = nii_img.get_fdata()  # Get the data as a numpy array
    binary_mask = binary_mask > 0  # Convert to a binary mask (assuming non-zero values are part of the mask)

    return binary_mask, nii_img.affine, nii_img.header
def create_3d_model_from_mask(binary_mask, stl_filename):
    """
    Step 1: Create a 3D model (STL) from the binary mask using Marching Cubes
    """
    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0)

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Save the mesh as an STL file
    mesh.export(stl_filename)
    print(f"3D surface model saved as {stl_filename}")

    return mesh
def create_convex_hull_and_fix_normals(mesh, output_stl_filename):
    """
    Step 2: Compute the convex hull and ensure that all normals are pointing outwards.
    """
    # Create the convex hull of the mesh
    hull_mesh = mesh.convex_hull

    # Ensure normals are outward
    hull_mesh.fix_normals()

    # Save the convex hull mesh as an STL file
    hull_mesh.export(output_stl_filename)
    print(f"Convex hull with fixed normals saved as {output_stl_filename}")

    return hull_mesh
def scale_mesh_fixed_center(stl_filename, output_filename, scale_factor):
    """
    Scales a 3D mesh from an STL file, keeping the center fixed, and saves the scaled mesh as a new STL file.

    Parameters:
    - stl_filename: str, path to the input STL file.
    - output_filename: str, path to save the scaled STL file.
    - scale_factor: float or list of 3 floats (x, y, z scaling factors).
                    If a single float is provided, it scales uniformly in all directions.
    """
    # Load the STL file
    mesh = trimesh.load_mesh(stl_filename)

    # Find the centroid of the mesh
    centroid = mesh.centroid

    # Translate the mesh to the origin (centroid becomes [0, 0, 0])
    mesh.vertices -= centroid

    # Apply scaling to the mesh
    if isinstance(scale_factor, (float, int)):  # Uniform scaling
        mesh.apply_scale(scale_factor)
    elif isinstance(scale_factor, (list, tuple)) and len(scale_factor) == 3:  # Non-uniform scaling (x, y, z)
        mesh.vertices *= scale_factor
    else:
        raise ValueError("scale_factor must be a float (uniform scaling) or a list of 3 floats (x, y, z scaling).")

    # Translate the mesh back to its original position
    mesh.vertices += centroid

    # Save the scaled mesh as an STL file
    mesh.export(output_filename)
    print(f"Scaled mesh (with fixed center) saved as {output_filename}")
def stl_to_binary_mask(stl_filename,inputniftifilename, output_nifti_filename, volume_shape):
    """
    Converts an STL file to a binary mask and saves it as a NIfTI file (no affine required).

    Parameters:
    - stl_filename: str, path to the input STL file.
    - output_nifti_filename: str, path to save the binary mask as a NIfTI file.
    - volume_shape: tuple, shape of the output binary mask (e.g., (x, y, z)).
    """
    # Load the STL file
    inputniftifilename_nib=nib.load(inputniftifilename)
    mesh = trimesh.load_mesh(stl_filename)

    # Create an empty binary mask with the given volume shape
    binary_mask = np.zeros(volume_shape, dtype=bool)

    # Create a grid of voxel coordinates
    grid = np.indices(volume_shape).reshape(3, -1).T

    # Check which of the voxel grid coordinates are inside the mesh (assuming voxel size is 1 unit)
    inside = mesh.contains(grid)

    # Update the binary mask with the points that are inside the mesh
    binary_mask[tuple(grid[inside].T)] = 1

    # Save the binary mask as a NIfTI file
    nifti_img = nib.Nifti1Image(binary_mask.astype(np.uint8), affine=inputniftifilename_nib.affine,header=inputniftifilename_nib.header) #np.eye(4))  # Identity affine (default)
    nib.save(nifti_img, output_nifti_filename)

    print(f"Binary mask saved as {output_nifti_filename}")


def binarymask_to_convexhull_mask(input_nii_path,output_nii_path):
    # Step 1: Load the input binary mask from a NIfTI file
    # input_nii_path = 'workinginput/ventricle.nii'
    binary_mask, affine, header = load_nii_mask(input_nii_path)

    # Step 2: Create 3D model from binary mask and save the STL file
    surface_mesh_stl = 'surface_model.stl'
    surface_mesh = create_3d_model_from_mask(binary_mask, surface_mesh_stl)

    # Step 3: Create convex hull from the 3D model and fix the normals
    convex_hull_stl = 'convex_hull_fixed_normals.stl'
    convex_hull_mesh = create_convex_hull_and_fix_normals(surface_mesh, convex_hull_stl)
    scale_mesh_fixed_center(convex_hull_stl, 'convex_hull_stl_1.stl', 1.05)
    # Step 4: Create binary mask from convex hull aligned with the original mask
    # convex_hull_binary_mask = create_binary_mask_from_hull(convex_hull_mesh, binary_mask.shape, affine)

    # # Step 5: Save the binary mask as a new NIfTI file
    # output_nii_path = 'convex_hull_binary_mask.nii'
    # save_nii_mask(convex_hull_binary_mask, affine, header, output_nii_path)
    #
    # print(f"Convex hull binary mask saved to: {output_nii_path}")

    convex_hull_mesh = trimesh.load('convex_hull_stl_1.stl') #'convex_hull_fixed_normals.stl')
    stl_to_binary_mask('convex_hull_stl_1.stl',input_nii_path, output_nii_path, binary_mask.shape)
ventricle_mask=resizeinto_512by512_and_flip(nib.load(sys.argv[1]).get_fdata())
csf_mask_nib=nib.load(sys.argv[2])
array_img = nib.Nifti1Image(ventricle_mask, affine=csf_mask_nib.affine, header=csf_mask_nib.header)
nib.save(array_img, os.path.join(sys.argv[3],'ventricle.nii'))
input_nii_path=os.path.join(sys.argv[3],'ventricle.nii') #'/media/atul/WDJan20222/WASHU_WORKS/PROJECTS/DOCKERIZE/CSFSEPERATION/TESTING_CSF_SEPERATION/workinginput/ventricle.nii'
output_nii_path=os.path.join(sys.argv[3],'ventricle_convexhull_mask.nii') #'../TESTING_CSF_SEPERATION/workinginput/ventricle_convexhull_mask.nii'

binarymask_to_convexhull_mask(input_nii_path,output_nii_path)