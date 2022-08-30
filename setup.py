#!/usr/bin/env python3

import io
import os
import re
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 8

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


# Get version
def find_version(*file_paths):
    try:
        with io.open(os.path.join(os.path.dirname(__file__), *file_paths), encoding="utf8") as fp:
            version_file = fp.read()
        version_match = re.search(r"^__version__ = version = ['\"]([^'\"]*)['\"]", version_file, re.M)
        return version_match.group(1)
    except Exception:
        return None


readme = open("README.md").read()


torch_min = "1.11"
install_requires = [
    "scikit-learn",
    "linear_operator>=0.1.1",
]
# if recent dev version of PyTorch is installed, no need to install stable
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = [">=".join(["torch", torch_min])] + install_requires
except ImportError:
    pass


# Run the setup
setup(
    name="gpytorch",
    version=find_version("gpytorch", "version.py"),
    description="An implementation of Gaussian Processes in Pytorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Jake Gardner, Geoff Pleiss",
    url="https://gpytorch.ai",
    author_email="jrg365@cornell.edu, gpleiss@gmail.com",
    project_urls={
        "Documentation": "https://gpytorch.readthedocs.io",
        "Source": "https://github.com/cornellius-gp/gpytorch/",
    },
    license="MIT",
    classifiers=["Development Status :: 4 - Beta", "Programming Language :: Python :: 3"],
    packages=find_packages(exclude=["test", "test.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "twine", "pre-commit"],
        "examples": ["ipython", "jupyter", "matplotlib", "scipy", "torchvision", "tqdm"],
        "pyro": ["pyro-ppl>=1.8"],
        "keops": ["pykeops>=1.1.1"],
        "test": ["flake8==4.0.1", "flake8-print==4.0.0", "pytest", "nbval"],
    },
    test_suite="test",
)
