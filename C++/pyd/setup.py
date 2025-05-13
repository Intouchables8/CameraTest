# python setup.py build_ext --inplace

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np
import sys

# file_name = r"D:\Code\CameraTest\C++\C_support_pyd\C_support_pyd\module.cpp"
# module_name = "edge_median_filter"

file_name = r"D:\Code\CameraTest\C++\C_support_pyd\ciede2000\ciede2000.cpp"
module_name = "ciede2000"

# OpenMP 支持设置
extra_compile_args = []
extra_link_args = []
if sys.platform == "win32":
    extra_compile_args.append("/openmp")  # Windows MSVC OpenMP
else:
    extra_compile_args.append("-fopenmp")  # Linux/macOS GCC/Clang OpenMP
    extra_link_args.append("-fopenmp")

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        module_name,  # 生成的 .pyd 模块名
        sources=[file_name],  # C++ 源文件
        include_dirs=[np.get_include()],  # NumPy 头文件路径
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]


# 运行 setup
setup(
    name=module_name,
    version="1.0",
    description="Median filter module implemented in C++ with pybind11 and OpenMP",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
