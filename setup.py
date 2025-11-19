import sys
from setuptools import setup, find_packages, Extension
import pybind11

if sys.platform.startswith("win"):
    cpp_std_flag = "/std:c++17"
else:
    cpp_std_flag = "-std=c++17"

ext_modules = [
    Extension(
        "MixTokenizer.core.decode",
        sources=["MixTokenizer/core/decode.cpp"],
        include_dirs=[pybind11.get_include(), "MixTokenizer/core"],
        language="c++",
        extra_compile_args=[cpp_std_flag],
    ),
    Extension(
        "MixTokenizer.core.utils",
        sources=["MixTokenizer/core/utils.cpp"],
        include_dirs=[pybind11.get_include(), "MixTokenizer/core"],
        language="c++",
        extra_compile_args=[cpp_std_flag],
    ),
]

setup(
    name="MixTokenizer",
    version="0.1.0",
    description="MixTokenizer",
    packages=find_packages(include=["MixTokenizer", "MixTokenizer.*"]),
    zip_safe=False,
    install_requires=[
        "transformers",
        "datasets",
        "numpy",
        "bitarray",
    ],
    ext_modules=ext_modules,
    python_requires=">=3.8",
)