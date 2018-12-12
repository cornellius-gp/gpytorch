#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

this_file = os.path.dirname(__file__)

setup(
    name="gpytorch",
    version="0.1.0",
    description="An implementation of Gaussian Processes in Pytorch",
    url="https://gpytorch.ai",
    author="Jake Gardner, Geoff Pleiss",
    author_email="jrg365@cornell.edu, gpleiss@gmail.com",
    install_requires=[],
    setup_requires=[],
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=["Development Status :: 4 - Beta", "Programming Language :: Python :: 3"],
    project_urls={
        "Documentation": "https://gpytorch.readthedocs.io",
        "Source": "https://github.com/cornellius-gp/gpytorch/",
    },
    ext_package="",
)
