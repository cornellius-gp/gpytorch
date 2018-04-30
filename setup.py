#!/usr/bin/env python
import os
from setuptools import setup, find_packages

this_file = os.path.dirname(__file__)

setup(
    name='gpytorch',
    version='0.1',
    description='An implementation of Gaussian Processes in Pytorch',
    url='https://github.com/cornellius-gp/gpytorch',
    author='Jake Gardner, Geoff Pleiss',
    author_email='jrg365@cornell.edu',
    install_requires=[],
    setup_requires=[],
    packages=find_packages(),
    ext_package='',
)
