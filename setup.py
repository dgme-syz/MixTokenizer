from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys
import pybind11

# OpenMP flags
if sys.platform == "win32":
    openmp_flag = ["/openmp", "/std:c++17"]
else:
    openmp_flag = ["-fopenmp", "-std=c++17"]

extensions = [
    # # === Cython  ===
    # Extension(
    #     "MixTokenizer.core.segment_core",
    #     ["MixTokenizer/core/segment_core.pyx"],
    #     include_dirs=[np.get_include()],
    #     language="c++",
    #     extra_compile_args=["-O3"] + openmp_flag,
    #     extra_link_args=openmp_flag,
    # ),

    # === C++ pybind11  ===
    Extension(
        "MixTokenizer.core.cpp_core",             
        sources=["MixTokenizer/core/cpp_core.cpp"],  
        include_dirs=[
            np.get_include(),
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["-O3"] + openmp_flag,
        extra_link_args=openmp_flag,
    ),
]

setup(
    name="mix_tokenizer",
    version="0.1.0",
    description="MixTokenizer with Cython and pybind11 extensions",
    packages=find_packages(include=["MixTokenizer", "MixTokenizer.*"]),
    ext_modules=cythonize(
        extensions,
        language_level="3",
        annotate=False,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False
        },
    ),
    zip_safe=False,
    install_requires=[
        "transformers>=4.57.1",
        "numpy",
        "scipy",
        "cython",
        "tokenizers",
        "pybind11",
    ],
    python_requires=">=3.8",
)
