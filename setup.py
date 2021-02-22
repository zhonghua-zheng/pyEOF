from setuptools import setup, find_packages
import os

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Atmospheric Science"
    ]

with open("README.rst", "r") as fp:
    long_description = fp.read()

setup(
    name="pyEOF",
    version="0.0.0",
    author="Zhonghua Zheng",
    author_email="zhonghua.zheng@outlook.com",
    url="https://github.com/zzheng93/pyEOF",
    description="Empirical Orthogonal Function (EOF) analysis and Rotated EOF analysis in Python",
    long_description=long_description,
    license="MIT",
    classifiers=classifiers,
    install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn', 'rpy2', 'tzlocal'],
    packages=find_packages()
    )