/*
# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is borrowed from https://github.com/autonomousvision/occupancy_networks
*/

#ifndef _PYWRAPPER_H
#define _PYWRAPPER_H

#include <Python.h>
#include "pyarraymodule.h"

#include <vector>

PyObject* marching_cubes(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes2(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes3(PyArrayObject* arr, double isovalue);
PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f, double isovalue);

#endif // _PYWRAPPER_H
