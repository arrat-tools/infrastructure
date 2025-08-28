import glob
import os
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup, find_packages

extensions = []
op_files = glob.glob('./clrnet/ops/csrc/*.c*')
extension = CUDAExtension
ext_name = 'clrnet.ops.nms_impl'
ext_ops = extension(name=ext_name, sources=op_files)
extensions.append(ext_ops)


setup(ext_modules=extensions,cmdclass={'build_ext': BuildExtension}, zip_safe=False)
