# GPyTorch

---
__News: GPyTorch v1.0.0__

GPyTorch v1.0.0 has just been released. This release marks our exit from beta status and in to what we consider stable software. This means that we do not expect you to encounter any major bugs when using stable features. Check out the [release notes](https://github.com/cornellius-gp/gpytorch/releases/tag/v1.0.0), as well as our fully revamped documentation and example notebooks.

---

[![Build status](https://travis-ci.org/cornellius-gp/gpytorch.svg?branch=master)](https://travis-ci.org/cornellius-gp/gpytorch)
[![Documentation Status](https://readthedocs.org/projects/gpytorch/badge/?version=latest)](https://gpytorch.readthedocs.io/en/latest/?badge=latest)

GPyTorch is a Gaussian process library implemented using PyTorch. GPyTorch is designed for creating scalable, flexible, and modular Gaussian process models with ease.

Internally, GPyTorch differs from many existing approaches to GP inference by performing all inference operations using modern numerical linear algebra techniques like preconditioned conjugate gradients. Implementing a scalable GP method is as simple as providing a matrix multiplication routine with the kernel matrix and its derivative via our `LazyTensor` interface, or by composing many of our already existing `LazyTensors`. This allows not only for easy implementation of popular scalable GP techniques, but often also for significantly improved utilization of GPU computing compared to solvers based on the Cholesky decomposition.

GPyTorch provides (1) significant GPU acceleration (through MVM based inference); (2) state-of-the-art implementations of the latest algorithmic advances for scalability and flexibility ([SKI/KISS-GP](http://proceedings.mlr.press/v37/wilson15.pdf), [stochastic Lanczos expansions](https://arxiv.org/abs/1711.03481), [LOVE](https://arxiv.org/pdf/1803.06058.pdf), [SKIP](https://arxiv.org/pdf/1802.08903.pdf), [stochastic variational](https://arxiv.org/pdf/1611.00336.pdf) [deep kernel learning](http://proceedings.mlr.press/v51/wilson16.pdf), ...); (3) easy integration with deep learning frameworks.

## Examples, Tutorials, and Documentation

See our numerous [**examples and tutorials**](https://gpytorch.readthedocs.io/en/latest/) on how to construct all sorts of models in GPyTorch.

## Installation

**Requirements**:
- Python >= 3.6
- PyTorch >= 1.3

Install GPyTorch using pip or conda:

```bash
pip install gpytorch
conda install gpytorch -c gpytorch
```

(To use packages globally but install GPyTorch as a user-only package, use `pip install --user` above.)

#### Latest (unstable) version

To upgrade to the latest (unstable) version, run

```bash
pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git
```

#### ArchLinux Package
Note: Experimental AUR package. For most users, we recommend installation by conda or pip.

GPyTorch is also available on the [ArchLinux User Repository](https://wiki.archlinux.org/index.php/Arch_User_Repository) (AUR).
You can install it with an [AUR helper](https://wiki.archlinux.org/index.php/AUR_helpers), like [`yay`](https://aur.archlinux.org/packages/yay/), as follows:

```bash
yay -S python-gpytorch
```
To discuss any issues related to this AUR package refer to the comments section of
[`python-gpytorch`](https://aur.archlinux.org/packages/python-gpytorch/).

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

If you plan on submitting a pull request, please make use of our pre-commit hooks to ensure that your commits adhere
to the general style guidelines enforced by the repo. To do this, navigate to your local repository and run:
```bash
pip install pre-commit
pre-commit install
```
From then on, this will automatically run flake8, isort, black and other tools over the files you commit each time you commit to gpytorch or a fork of it.

## The Team

GPyTorch is primarily maintained by:
- [Jake Gardner](http://github.com/jacobrgardner) (Uber AI Labs)
- [Geoff Pleiss](http://github.com/gpleiss) (Cornell University)
- [Kilian Weinberger](http://kilian.cs.cornell.edu/) (Cornell University)
- [Andrew Gordon Wilson](https://people.orie.cornell.edu/andrew/) (Cornell University)
- [Max Balandat](https://research.fb.com/people/balandat-max/) (Facebook)

<img width="300" src=https://brand.cornell.edu/assets/images/downloads/logos/cornell_logo_simple/cornell_logo_simple.svg alt="Cornell Logo" />
<img width="300" src=https://raw.githubusercontent.com/cornellius-gp/cornellius-gp.github.io/master/static/media/facebook_logo.2835357a.png alt="Facebook Logo" />
<img width="300" src=https://gpytorch.ai/static/media/uber_ai_horizontal.fe9ab653.png alt="Uber AI Logo" />
We would like to thank our other contributors including (but not limited to)  David Arbour, Eytan Bakshy, David Eriksson, Jared Frank, Sam Stanton, Bram Wallace, Ke Alexander Wang, Ruihan Wu.

## Acknowledgements
Development of GPyTorch is supported by funding from the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/), the [National Science Foundation](https://www.nsf.gov/), and [SAP](https://www.sap.com/index.html).
