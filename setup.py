from setuptools import setup, find_packages

setup(
    name="MixTokenizer",
    version="0.1.0",
    description="MixTokenizer",
    packages=find_packages(include=["MixTokenizer", "MixTokenizer.*"]),
    zip_safe=False,
    install_requires=[
        "transformers",
        "numpy",
    ],
    python_requires=">=3.8",
)
