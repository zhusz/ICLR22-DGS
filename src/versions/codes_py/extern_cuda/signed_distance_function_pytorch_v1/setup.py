# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is borrowed from https://github.com/JiangWenPL/multiperson

from setuptools import setup, find_packages
import unittest
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def test_all():
    pass

ext_modules = [
    CUDAExtension('signed_distance_function_v1.cuda.mesh_preprocessing', [
        'signed_distance_function_v1/cuda/mesh_preprocessing.cpp',
        'signed_distance_function_v1/cuda/mesh_preprocessing_kernel.cu',
    ]),
    CUDAExtension('signed_distance_function_v1.cuda.sdf_core', [
        'signed_distance_function_v1/cuda/sdf_core_cuda.cpp',
        'signed_distance_function_v1/cuda/sdf_core_cuda_kernel.cu'
    ])
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='SDF package',
    author='Shizhan Zhu',
    license='MIT License',
    version='1.0.0',
    name='signed_distance_function_v1',
    test_suite='setup.test_all',
    packages=['signed_distance_function_v1', 'signed_distance_function_v1.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
