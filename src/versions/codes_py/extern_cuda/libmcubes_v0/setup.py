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
