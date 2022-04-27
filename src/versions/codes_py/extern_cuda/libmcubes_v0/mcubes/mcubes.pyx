# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is borrowed from https://github.com/autonomousvision/occupancy_networks

# distutils: language = c++
# cython: embedsignature = True

# from libcpp.vector cimport vector
import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()

cdef extern from "pywrapper.h":
    cdef object c_marching_cubes "marching_cubes"(np.ndarray, double) except +
    cdef object c_marching_cubes2 "marching_cubes2"(np.ndarray, double) except +
    cdef object c_marching_cubes3 "marching_cubes3"(np.ndarray, double) except +
    cdef object c_marching_cubes_func "marching_cubes_func"(tuple, tuple, int, int, int, object, double) except +

def marching_cubes(np.ndarray volume, float isovalue):
    
    verts, faces = c_marching_cubes(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes2(np.ndarray volume, float isovalue):

    verts, faces = c_marching_cubes2(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes3(np.ndarray volume, float isovalue):

    verts, faces = c_marching_cubes3(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes_func(tuple lower, tuple upper, int numx, int numy, int numz, object f, double isovalue):
    
    verts, faces = c_marching_cubes_func(lower, upper, numx, numy, numz, f, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces
