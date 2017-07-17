# GPyTorch
![Build status](https://travis-ci.org/jrg365/gpytorch.svg?branch=master)

GPyTorch is a Gaussian Process library, implemented using PyTorch.
It is designed for creating flexible and modular Gaussian Process models with ease,
so that you don't have to be an expert to use GPs.

This package is currently under development, and is likely to change.
Some things you can do right now:

- Simple GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/simple_gp_regression.ipynb))
- Simple GP classification ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/variational_inference/examples/simple_gp_classification.ipynb))
- Simple GP regression with hyperparameter priors using MAP ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/simple_map_gp_regression.ipynb))
- Multitask GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/multitask_gp_regression.ipynb))
- Multitask GP regression, but the model learns to group some tasks together ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/latent_multitask_gp_regression.ipynb))
- Extrapolation using the spectral mixture kernel ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/spectral_mixture_gp_regression.ipynb))
## Installation

Make sure you have PyTorch (>= 0.1.10) installed.

```bash
git clone https://github.com/jrg365/gpytorch.git
cd gpytorch
python setup.py install
```

## Documentation

Still a work in progress. For now, please refer to the following [example Jupyter notebooks](https://nbviewer.jupyter.org/github/jrg365/gpytorch/tree/master/examples/).


## Development

To run the unit tests:
```bash
python -m pytest
```
