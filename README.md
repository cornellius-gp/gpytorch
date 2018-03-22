# GPyTorch (Alpha Relase)
[![Build status](https://travis-ci.org/cornellius-gp/gpytorch.svg?branch=master)](https://travis-ci.org/cornellius-gp/gpytorch)

GPyTorch is a Gaussian Process library, implemented using PyTorch.
It is designed for creating flexible and modular Gaussian Process models with ease,
so that you don't have to be an expert to use GPs.

This package is currently under development, and is likely to change.
Some things you can do right now:

- Simple GP regression ([example here](https://nbviewer.jupyter.org/github/cornellius-gp/gpytorch/blob/master/examples/simple_gp_regression.ipynb))
- Simple GP classification ([example here](https://nbviewer.jupyter.org/github/cornellius-gp/gpytorch/blob/master/examples/simple_gp_classification.ipynb))
- Multitask GP regression ([example here](https://nbviewer.jupyter.org/github/cornellius-gp/gpytorch/blob/master/examples/multitask_gp_regression.ipynb))
- Scalable GP regression using kernel interpolation ([example here](https://nbviewer.jupyter.org/github/cornellius-gp/gpytorch/blob/master/examples/kissgp_gp_regression.ipynb))
- Scalable GP classification using kernel interpolation ([example here](https://nbviewer.jupyter.org/github/cornellius-gp/gpytorch/blob/master/examples/kissgp_gp_classification.ipynb))
- Deep kernel learning ([example here](https://nbviewer.jupyter.org/github/cornellius-gp/gpytorch/blob/master/examples/dkl_mnist.ipynb))
- And ([more!](http://github.com/cornellius-gp/gpytorch/blob/master/examples))

If you use GPyTorch, please cite the following papers:
> [Gardner, Jacob R., Geoff Pleiss, Ruihan Wu, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Product Kernel Interpolation for Scalable Gaussian Processes." In *AISTATS* (2018).](https://arxiv.org/abs/1802.08903)
```
@inproceedings{gardner2018product,
  title={Product Kernel Interpolation for Scalable Gaussian Processes},
  author={Gardner, Jacob R and Pleiss, Geoff and Wu, Ruihan and Weinberger, Kilian Q and Wilson, Andrew Gordon},
  booktitle={AISTATS},
  year={2018}

```
> [Pleiss, Geoff, Jacob R. Gardner, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Constant-Time Predictive Distributions for Gaussian Processes." arXiv preprint arXiv:1803.06058 (2018).](https://arxiv.org/abs/1803.06058)
```
@article{pleiss2018constant,
  title={Constant-Time Predictive Distributions for Gaussian Processes},
  author={Pleiss, Geoff and Gardner, Jacob R and Weinberger, Kilian Q and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:1803.06058},
  year={2018}
}
```

## Installation

### Global installation

The easiest way to install GPyTorch is by installing the dependencies we require, `PyTorch >= 0.3.0` and `libfftw3 > 3.3.6` ([source](http://www.fftw.org/download.html)) using conda, and then installing 
GPyTorch using pip. This can be accomplished globally using one of the two sets of commands below depending on whether you want CUDA support.

For CUDA/GPU support, run:
```bash
conda install fftw cffi pytorch torchvision cuda80 -c conda-forge -c pytorch
pip install git+https://github.com/cornellius-gp/gpytorch.git
```

If you do not have or do not wish to use CUDA, instead run:
```bash
conda install fftw cffi pytorch torchvision -c conda-forge -c pytorch
pip install git+https://github.com/cornellius-gp/gpytorch.git
```

If you install libfftw3 from source, be sure to run `configure` with `--enable-shared`. To use packages globally but install GPyTorch as a user-only package, use `pip install --user` above.

### Installation in a conda environment

We also provide two conda environment files, `environment.yml` and `environment_cuda.yml`. As an example, to install GPyTorch in a conda environment with cuda support, run:

```bash
git clone git+https://github.com/cornellius-gp/gpytorch.git
conda create -f gpytorch/environment_cuda.yml
source activate gpytorch
pip install gpytorch/
```

## Documentation

Still a work in progress. For now, please refer to the following [example Jupyter notebooks](https://github.com/cornellius-gp/gpytorch/tree/master/examples/).


## Development

To run the unit tests:
```bash
python -m unittest
```

By default, the random seeds are locked down for some of the tests.
If you want to run the tests without locking down the seed, run
```bash
UNLOCK_SEED=true python -m unittest
```


Please lint the code with `flake8`.
```bash
pip install flake8  # if not already installed
flake8
```

## Acknowledgements
Development of GPyTorch is supported by funding from the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/).
