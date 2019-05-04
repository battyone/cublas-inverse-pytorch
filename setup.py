from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os


conda = "/usr/local/cuda-10.0"
inc = [conda + "/include"]

libname = "cuda_batch_inverse"
setup(name=libname,
      ext_modules=[CppExtension(
          libname,
          ['cuda_inverse_pytorch.cpp'],
          include_dirs=inc,
          libraries=["cusolver", "cublas"],
          extra_compile_args={'cxx': ['-g', '-DDEBUG'],
                              'nvcc': ['-O2']}
      )],
      cmdclass={'build_ext': BuildExtension})


