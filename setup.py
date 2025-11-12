from setuptools import setup, find_packages, Extension
import pybind11

ext_modules = [
    Extension(
        "Mixtokenizer.core.decode",
        sources=["Mixtokenizer/core/decode.cpp"],
        include_dirs=[pybind11.get_include(), "Mixtokenizer/core"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
    Extension(
        "Mixtokenizer.core.utils",
        sources=["Mixtokenizer/core/utils.cpp"],
        include_dirs=[pybind11.get_include(), "Mixtokenizer/core"],
        language="c++",
        extra_compile_args=["-std=c++17"],  
    )
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
