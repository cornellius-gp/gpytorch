# GPyTorch (Beta Release)
[![Build status](https://travis-ci.org/cornellius-gp/gpytorch.svg?branch=master)](https://travis-ci.org/cornellius-gp/gpytorch)
[![Documentation Status](https://readthedocs.org/projects/gpytorch/badge/?version=latest)](https://gpytorch.readthedocs.io/en/latest/?badge=latest)

**News!**
 - The Beta release is currently out! Note that it **requires PyTorch >= 1.0.0**
 - If you need to install the alpha release (we recommend you use the latest version though!), check out [the alpha release](https://github.com/cornellius-gp/gpytorch/tree/alpha).

GPyTorch is a Gaussian process library implemented using PyTorch. GPyTorch is designed for creating scalable, flexible, and modular Gaussian process models with ease. 

Internally, GPyTorch differs from many existing approaches to GP inference by performing all inference operations using modern numerical linear algebra techniques like preconditioned conjugate gradients. Implementing a scalable GP method is as simple as providing a matrix multiplication routine with the kernel matrix and its derivative via our `LazyTensor` interface, or by composing many of our already existing `LazyTensors`. This allows not only for easy implementation of popular scalable GP techniques, but often also for significantly improved utilization of GPU computing compared to solvers based on the Cholesky decomposition. 

GPyTorch provides (1) significant GPU acceleration (through MVM based inference); (2) state-of-the-art implementations of the latest algorithmic advances for scalability and flexibility ([SKI/KISS-GP](http://proceedings.mlr.press/v37/wilson15.pdf), [stochastic Lanczos expansions](https://arxiv.org/abs/1711.03481), [LOVE](https://arxiv.org/pdf/1803.06058.pdf), [SKIP](https://arxiv.org/pdf/1802.08903.pdf), [stochastic variational](https://arxiv.org/pdf/1611.00336.pdf) [deep kernel learning](http://proceedings.mlr.press/v51/wilson16.pdf), ...); (3) easy integration with deep learning frameworks.

## Examples and Tutorials

See our numerous [**examples and tutorials**](http://github.com/cornellius-gp/gpytorch/blob/master/examples) on how to construct all sorts of models in GPyTorch. These example notebooks and a walk through of GPyTorch are also available at our **ReadTheDocs page [here](https://gpytorch.readthedocs.io/en/latest/index.html)**.

## Installation

**Requirements**:
- Python >= 3.6
- PyTorch >= 1.0

**N.B.** GPyTorch will not run on PyTorch 0.4.1 or earlier versions.

First make sure that you have PyTorch (`>= 1.0.0`) installed using the appropriate command from [here](https://pytorch.org/get-started/locally/).

Then install GPyTorch using pip or conda:

```bash
pip install gpytorch
conda install gpytorch -c gpytorch
```

To use packages globally but install GPyTorch as a user-only package, use `pip install --user` above.

#### Latest (unstable) version

To get the latest (unstable) version, run

```bash
pip install git+https://github.com/cornellius-gp/gpytorch.git
```

## Citing Us

If you use GPyTorch, please cite the following papers:
> [Gardner, Jacob R., Geoff Pleiss, David Bindel, Kilian Q. Weinberger, and Andrew Gordon Wilson. "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration." In Advances in Neural Information Processing Systems (2018).](https://arxiv.org/abs/1809.11165)
```
@inproceedings{gardner2018gpytorch,
  title={GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration},
  author={Gardner, Jacob R and Pleiss, Geoff and Bindel, David and Weinberger, Kilian Q and Wilson, Andrew Gordon},
  booktitle={Advances in Neural Information Processing Systems},
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

## The Team

GPyTorch is primarily maintained by:
- [Jake Gardner](http://github.com/jacobrgardner) (Cornell University)
- [Geoff Pleiss](http://github.com/gpleiss) (Cornell University)
- [Kilian Weinberger](http://kilian.cs.cornell.edu/) (Cornell University)
- [Andrew Gordon Wilson](https://people.orie.cornell.edu/andrew/) (Cornell University)
- [Max Balandat](https://research.fb.com/people/balandat-max/) (Facebook)

<img width="300" src=https://brand.cornell.edu/assets/images/downloads/logos/cornell_logo_simple/cornell_logo_simple.svg alt="Cornell Logo" />

We would like to thank our other contributors including (but not limited to) Eytan Bakshy, David Arbour, Ruihan Wu, Bram Wallace, Sam Stanton, and Jared Frank.

## Acknowledgements
Development of GPyTorch is supported by funding from [Facebook](https://research.fb.com/), the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/), the [National Science Foundation](https://www.nsf.gov/), and [SAP](https://www.sap.com/index.html).
