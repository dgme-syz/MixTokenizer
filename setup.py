from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "MixTokenizer.core.segment_core",  
        ["MixTokenizer/core/segment_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  
    )
]

setup(
    name="mix_tokenizer",
    version="0.1.0",
    description="MixTokenizer with custom Cython Trie and new language support",
    packages=["MixTokenizer", "MixTokenizer.core"],
    ext_modules=cythonize(
        extensions,
        language_level="3",  
        annotate=False,       
    ),
    zip_safe=False,
    install_requires=[
        "transformers>=4.57.1",
        "numpy",
        "scipy",
        "cython",
        "tokenizers",
    ],
    python_requires=">=3.8",
)
