#!/usr/bin/env python
import os
from setuptools import setup, find_packages

import build

__all__ = [build]

this_file = os.path.dirname(__file__)

setup(
    name='gpytorch',
    version='0.1',
    description='An implementation of Gaussian Processes in Pytorch',
    url='https://github.com/jrg365/gpytorch',
    author='Jake Gardner, Geoff Pleiss',
    author_email='jrg365@cornell.edu',
    install_requires=['cffi>=1.4.0'],
    setup_requires=['cffi>=1.4.0'],
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[
        os.path.join(this_file, 'build.py:ffi')
    ],
)
