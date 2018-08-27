# GPyTorch (Alpha Release)
[![Build status](https://travis-ci.org/cornellius-gp/gpytorch.svg?branch=master)](https://travis-ci.org/cornellius-gp/gpytorch)

GPyTorch is a Gaussian process library implemented using PyTorch. GPyTorch is designed for creating scalable, flexible, and modular Gaussian process models with ease. 

Internally, GPyTorch differs from many existing approaches to GP inference by performing all inference operations using modern numerical linear algebra techniques like preconditioned conjugate gradients. Implementing a scalable GP method is as simple as providing a matrix multiplication routine with the kernel matrix and its derivative via our `LazyVariable` interface, or by composing many of our already existing `LazyVariables`. This allows not only for easy implementation of popular scalable GP techniques, but often also for significantly improved utilization of GPU computing compared to solvers based on the Cholesky decomposition. 

GPyTorch provides (1) significant GPU acceleration (through MVM based inference); (2) state-of-the-art implementations of the latest algorithmic advances for scalability and flexibility ([SKI/KISS-GP](http://proceedings.mlr.press/v37/wilson15.pdf), [stochastic Lanczos expansions](https://arxiv.org/abs/1711.03481), [LOVE](https://arxiv.org/pdf/1803.06058.pdf), [SKIP](https://arxiv.org/pdf/1802.08903.pdf), [stochastic variational](https://arxiv.org/pdf/1611.00336.pdf) [deep kernel learning](http://proceedings.mlr.press/v51/wilson16.pdf), ...); (3) easy integration with deep learning frameworks.

## Examples and Tutorials

Right now, the package is in alpha release, and while we believe that the interface is reasonably stable, things may change. For now, see our numerous [**examples and tutorials**](http://github.com/cornellius-gp/gpytorch/blob/master/examples) on how to construct all sorts of models in GPyTorch. These example notebooks and a walk through of GPyTorch are also available at our **ReadTheDocs page [here](https://gpytorch.readthedocs.io/en/latest/index.html)**

## Installation

### Global installation

The easiest way to install GPyTorch is by installing `PyTorch >= 0.4.0` using the appropriate command from [here](http://pytorch.org), and then installing 
GPyTorch using pip:

```bash
pip install git+https://github.com/cornellius-gp/gpytorch.git
```
To use packages globally but install GPyTorch as a user-only package, use `pip install --user` above.

### Installation in a conda environment

We also provide two conda environment files, `environment.yml` and `environment_cuda90.yml`. As an example, to install GPyTorch in a conda environment with cuda support, run:

```bash
git clone git+https://github.com/cornellius-gp/gpytorch.git
conda create -f gpytorch/environment_cuda.yml
source activate gpytorch
pip install gpytorch/
```

## Citing Us

If you use GPyTorch, please cite the following papers:
> [Gardner, Jacob R., Geoff Pleiss, Ruihan Wu, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Product Kernel Interpolation for Scalable Gaussian Processes." In *AISTATS* (2018).](https://arxiv.org/abs/1802.08903)
```
@inproceedings{gardner2018product,
  title={Product Kernel Interpolation for Scalable Gaussian Processes},
  author={Gardner, Jacob R and Pleiss, Geoff and Wu, Ruihan and Weinberger, Kilian Q and Wilson, Andrew Gordon},
  booktitle={AISTATS},
  year={2018}
}
```
> [Pleiss, Geoff, Jacob R. Gardner, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Constant-Time Predictive Distributions for Gaussian Processes." In *ICML* (2018).](https://arxiv.org/abs/1803.06058)
```
@inproceedings{pleiss2018constant,
  title={Constant-Time Predictive Distributions for Gaussian Processes},
  author={Pleiss, Geoff and Gardner, Jacob R and Weinberger, Kilian Q and Wilson, Andrew Gordon},
  booktitle={ICML},
  year={2018}
}
```

## Documentation

- For **tutorials and examples**, check out [the examples folder](https://github.com/cornellius-gp/gpytorch/tree/master/examples).
- For in-depth **documentation**, check out our [read the docs](http://gpytorch.readthedocs.io/).


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

## Founding Team

GPyTorch is developed at Cornell University by
- [Jake Gardner](http://github.com/jacobrgardner) (lead developer)
- [Geoff Pleiss](http://github.com/gpleiss) (lead developer)
- [Kilian Weinberger](http://kilian.cs.cornell.edu/)
- [Andrew Gordon Wilson](https://people.orie.cornell.edu/andrew/)

<img width="300" src=https://brand.cornell.edu/assets/images/downloads/logos/cornell_logo_simple/cornell_logo_simple.svg alt="Cornell Logo" />

We would like to thank our other contributors including (but not limited to) Max Balandat, Ruihan Wu, Bram Wallace, Jared Frank.

## Acknowledgements
Development of GPyTorch is supported by funding from the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/).
