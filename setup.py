from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

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
    ext_modules=cythonize(
        Extension(
            "MixTokenizer.core.str",
            ["MixTokenizer/core/str.pyx"],
            language="c++"
        ),
        compiler_directives={"language_level": "3"},
    ),
    python_requires=">=3.8",
)
