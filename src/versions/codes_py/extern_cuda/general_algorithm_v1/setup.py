# Resulted cuda binaries are supposed only runnable for pytorch >= 1.0.0
from setuptools import setup, find_packages
import unittest
import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


ext_modules = [
    CUDAExtension(
        'general_algorithm_v1.cuda.general_algorithm', [
            'general_algorithm_v1/cuda/general_algorithm.cpp',
            'general_algorithm_v1/cuda/general_algorithm_kernel.cu',
        ]
    )
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='A package for general algorithms',
    author='Shizhan Zhu',
    author_email='Wu Ke Feng Gao',
    license='MIT License',
    version='1.0.0',
    name='general_algorithm_v1',
    test_suite='setup.test_all',
    packages=['general_algorithm_v1.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
