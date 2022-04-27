# Code Release for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
# Released on 04/25/2022.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# triangle hash (efficient mesh intersection)
ext_modules = [
    Extension(
        'libmesh_v0.triangle_hash',
        sources=[
            'libmesh_v0/triangle_hash.pyx'
        ],
        libraries=['m'],  # Unix-like specific
        include_dirs=[numpy_include_dir]
    ),
]

setup(
    ext_modules=cythonize(ext_modules),
    name='libmesh_v0',
    version='0.0.0',
    packages=['libmesh_v0'],
    cmdclass={
        'build_ext': BuildExtension
    }
)
