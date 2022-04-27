# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This file is borrowed from https://github.com/autonomousvision/occupancy_networks

from setuptools import setup

from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'libmcubes_v0.mcubes',
    sources=[
        'mcubes/mcubes.pyx',
        'mcubes/pywrapper.cpp',
        'mcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir],
)

ext_modules = [
    mcubes_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    name='libmcubes_v0',
    version='0.0.0',
    cmdclass={
        'build_ext': BuildExtension
    }
)
