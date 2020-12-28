#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


readme = open("README.md").read()
version = find_version("gpytorch", "__init__.py")


torch_min = "1.7"
install_requires = [">=".join(["torch", torch_min]), "scikit-learn", "scipy"]
# if recent dev version of PyTorch is installed, no need to install stable
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = []
except ImportError:
    pass


# Run the setup
setup(
    name="gpytorch",
    version=version,
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
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "twine", "pre-commit"],
        "docs": ["ipython", "ipykernel", "sphinx<3.0.0", "sphinx_rtd_theme", "nbsphinx", "m2r"],
        "examples": ["ipython", "jupyter", "matplotlib", "scipy", "torchvision", "tqdm"],
        "pyro": ["pyro-ppl>=1.0.0"],
        "keops": ["pykeops>=1.1.1"],
        "test": ["flake8", "flake8-print", "pytest", "nbval"],
    },
)
