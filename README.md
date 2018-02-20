# GPyTorch (Pre-release, under development)
![Build status](https://travis-ci.org/cornellius-gp/gpytorch.svg?branch=master)

GPyTorch is a Gaussian Process library, implemented using PyTorch.
It is designed for creating flexible and modular Gaussian Process models with ease,
so that you don't have to be an expert to use GPs.

This package is currently under development, and is likely to change.
Some things you can do right now:

- Simple GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/simple_gp_regression.ipynb))
- Simple GP classification ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/simple_gp_classification.ipynb))
- Multitask GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/multitask_gp_regression.ipynb))
- Extrapolation using the spectral mixture kernel ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/spectral_mixture_gp_regression.ipynb))
- Scalable GP regression using kernel interpolation ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/kissgp_gp_regression.ipynb))
- Scalable GP classification using kernel interpolation ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/kissgp_gp_classification.ipynb))
- Scalable GP regression in multiple dimensions ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/kissgp_kronecker_product_regression.ipynb))
- Scalable GP classification in multiple dimensions ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/kissgp_kronecker_product_classification.ipynb))

## Installation

### Global installation

The easiest way to install GPyTorch is by installing the dependencies we require, `PyTorch >= 0.3.0` and `libfftw3 > 3.3.6` ([source](http://www.fftw.org/download.html)) using conda, and then installing 
GPyTorch using pip. This can be accomplished globally using one of the two sets of commands below depending on whether you want CUDA support.

For CUDA/GPU support, run:
```bash
conda install fftw cffi pytorch torchvision cuda80 -c conda-forge -c soumith
pip install git+https://github.com/jrg365/gpytorch.git
```

If you do not have or do not wish to use CUDA, instead run:
```bash
conda install fftw cffi pytorch torchvision -c conda-forge -c soumith
pip install git+https://github.com/jrg365/gpytorch.git
```

If you install libfftw3 from source, be sure to run `configure` with `--enable-shared`. To use packages globally but install GPyTorch as a user-only package, use `pip install --user` above.

### Installation in a conda environment

We also provide two conda environment files, `environment.yml` and `environment_cuda.yml`. As an example, to install GPyTorch in a conda environment with cuda support, run:

```bash
git clone git+https://github.com/jrg365/gpytorch.git
conda create -f gpytorch/environment_cuda.yml
source activate gpytorch
pip install gpytorch/
```

## Documentation

Still a work in progress. For now, please refer to the following [example Jupyter notebooks](https://nbviewer.jupyter.org/github/jrg365/gpytorch/tree/master/examples/).


## Development

To run the unit tests:
```bash
python -m pytest
```

## Acknowledgements
Development of GPyTorch is supported by funding from the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/).
