# GPyTorch
GPyTorch is a Gaussian Process library, implemented using PyTorch.
It is designed for creating flexible an modular Gaussian Process models,
so that you don't have to be an expert to use GPs.

Soem things you can do right now:

- Simple GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/simple_gp_regression.ipynb))

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
