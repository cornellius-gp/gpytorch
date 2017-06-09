#!/usr/bin/env python
import os
import sys

from setuptools import setup

setup(
    name='gpytorch',
    version='0.1',
    description='An implementation of Gaussian Processes in Pytorch',
    url='https://github.com/jrg365/gpytorch',
    author='Jake Gardner, Geoff Pleiss',
    author_email='jrg365@cornell.edu',
    install_requires=[],
    setup_requires=[],
    packages=['gpytorch'],
    # packages=find_packages(exclude=['build']),
)
