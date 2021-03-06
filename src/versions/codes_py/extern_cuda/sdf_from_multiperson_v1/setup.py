# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import unittest
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def test_all():
    pass

ext_modules = [
    CUDAExtension('sdf_from_multiperson_v1.cuda.sdf_csrc', [
        'sdf_from_multiperson_v1/cuda/sdf_cuda.cpp',
        'sdf_from_multiperson_v1/cuda/sdf_cuda_kernel.cu'
    ])
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision']

setup(
    description='SDF from MultiPerson',
    author='Authors of the multiperson paper cvpr 2020 + Shizhan modification',
    license='MIT License',
    version='1.0.0',
    name='sdf_from_multiperson_v1',
    test_suite='setup.test_all',
    packages=['sdf_from_multiperson_v1', 'sdf_from_multiperson_v1.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
