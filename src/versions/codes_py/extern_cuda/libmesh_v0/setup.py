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
