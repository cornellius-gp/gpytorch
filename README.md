# GPyTorch (Pre-release, under development)
![Build status](https://travis-ci.org/jrg365/gpytorch.svg?branch=master)

GPyTorch is a Gaussian Process library, implemented using PyTorch.
It is designed for creating flexible and modular Gaussian Process models with ease,
so that you don't have to be an expert to use GPs.

This package is currently under development, and is likely to change.
Some things you can do right now:

- Simple GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/simple_gp_regression.ipynb))
- Simple GP classification (Currently slow. [example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/variational_inference/examples/simple_gp_classification.ipynb))
- Multitask GP regression ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/multitask_gp_regression.ipynb))
- Extrapolation using the spectral mixture kernel ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/spectral_mixture_gp_regression.ipynb))
- Scalable GP regression using kernel interpolation ([example here](https://nbviewer.jupyter.org/github/jrg365/gpytorch/blob/master/examples/kissgp_gp_regression.ipynb))
## Installation

Make sure you have PyTorch (>= 0.2.0) installed.

In addition, you will need libfftw3 (>= 3.3.6) installed on your machine. This can be downloaded [here](http://www.fftw.org/download.html), or installed for example on Ubuntu using

```bash
sudo apt-get install libfftw3-3
```

If you install libfftw3 from source, be sure to run `configure` with `--enable-shared`. Our build script by default looks for libraries in `/usr/local/lib`, which is the default installation
location for libfftw3. If it is installed elsewhere, however, be sure to either add the new location to your `LD_LIBRARY_PATH` environment variable, or add the new location to `build.py` in
`library_dirs`.

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
