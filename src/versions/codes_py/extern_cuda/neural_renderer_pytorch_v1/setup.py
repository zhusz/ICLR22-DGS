# Resulted cuda binaries are supposed only runnable for pytorch >= 1.0.0
from setuptools import setup, find_packages
import unittest
import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA_FLAGS = ['-I/scratch/zhusz/local_data/Installations/anaconda2/envs/t10/include']

def test_all():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

installationsDir = os.environ['INSTALLATIONS_DIR']
ext_modules=[
    CUDAExtension('neural_renderer_v1.cuda.load_textures', [
        'neural_renderer_v1/cuda/load_textures_cuda.cpp',
        'neural_renderer_v1/cuda/load_textures_cuda_kernel.cu',
        ]),
    CUDAExtension('neural_renderer_v1.cuda.rasterize', [
        'neural_renderer_v1/cuda/rasterize_cuda.cpp',
        'neural_renderer_v1/cuda/rasterize_cuda_kernel.cu',
        ]),
    CUDAExtension('neural_renderer_v1.cuda.create_texture_image', [
        'neural_renderer_v1/cuda/create_texture_image_cuda.cpp',
        'neural_renderer_v1/cuda/create_texture_image_cuda_kernel.cu',
        ]),
    CUDAExtension('neural_renderer_v1.cuda.uvmesh', [
        'neural_renderer_v1/cuda/uvmesh_cuda.cpp',
        'neural_renderer_v1/cuda/uvmesh_cuda_kernel.cu',
        ]),
    CUDAExtension('neural_renderer_v1.cuda.derasterize', [
        'neural_renderer_v1/cuda/derasterize_cuda.cpp',
        ], include_dirs=[installationsDir + '/' + 'eigen']),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "A 3D mesh renderer for neural networks"',
    author='Nikolaos Kolotouros',
    author_email='nkolot@seas.upenn.edu',
    license='MIT License',
    version='1.1.3',
    name='neural_renderer_v1',
    test_suite='setup.test_all',
    packages=['neural_renderer_v1', 'neural_renderer_v1.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
