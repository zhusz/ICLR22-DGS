# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import unittest
import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA_FLAGS = ['-I/scratch/zhusz/local_data/Installations/anaconda2/envs/t10/include']


def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


ext_modules = [
    CUDAExtension(
        'dgs_v1.cuda.dgs', [
            'dgs_v1/cuda/dgs.cpp',
            'dgs_v1/cuda/dgs_kernel.cu',
        ]
    )
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='A package for partial derivative enabled grid_2d_sampler',
    author='Shizhan Zhu',
    author_email='Wu Ke Feng Gao',
    license='MIT License',
    version='1.0.0',
    name='dgs_v1',
    test_suite='setup.test_all',
    packages=['dgs_v1.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
