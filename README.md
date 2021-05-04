# GPyTorch

---
![GPyTorch Unit Tests](https://github.com/cornellius-gp/gpytorch/workflows/GPyTorch%20Unit%20Tests/badge.svg)
![GPyTorch Examples](https://github.com/cornellius-gp/gpytorch/workflows/GPyTorch%20Examples/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/gpytorch/badge/?version=latest)](https://gpytorch.readthedocs.io/en/latest/?badge=latest)

GPyTorch is a Gaussian process library implemented using PyTorch. GPyTorch is designed for creating scalable, flexible, and modular Gaussian process models with ease.

Internally, GPyTorch differs from many existing approaches to GP inference by performing all inference operations using modern numerical linear algebra techniques like preconditioned conjugate gradients. Implementing a scalable GP method is as simple as providing a matrix multiplication routine with the kernel matrix and its derivative via our `LazyTensor` interface, or by composing many of our already existing `LazyTensors`. This allows not only for easy implementation of popular scalable GP techniques, but often also for significantly improved utilization of GPU computing compared to solvers based on the Cholesky decomposition.

GPyTorch provides (1) significant GPU acceleration (through MVM based inference); (2) state-of-the-art implementations of the latest algorithmic advances for scalability and flexibility ([SKI/KISS-GP](http://proceedings.mlr.press/v37/wilson15.pdf), [stochastic Lanczos expansions](https://arxiv.org/abs/1711.03481), [LOVE](https://arxiv.org/pdf/1803.06058.pdf), [SKIP](https://arxiv.org/pdf/1802.08903.pdf), [stochastic variational](https://arxiv.org/pdf/1611.00336.pdf) [deep kernel learning](http://proceedings.mlr.press/v51/wilson16.pdf), ...); (3) easy integration with deep learning frameworks.

## Examples, Tutorials, and Documentation

See our numerous [**examples and tutorials**](https://gpytorch.readthedocs.io/en/latest/) on how to construct all sorts of models in GPyTorch.

## Installation

**Requirements**:
- Python >= 3.6
- PyTorch >= 1.7

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
- [Jake Gardner](https://www.cis.upenn.edu/~jacobrg/index.html) (University of Pennsylvania)
- [Geoff Pleiss](http://github.com/gpleiss) (Columbia University)
- [Kilian Weinberger](http://kilian.cs.cornell.edu/) (Cornell University)
- [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/) (New York University)
- [Max Balandat](https://research.fb.com/people/balandat-max/) (Facebook)

We would like to thank our other contributors including (but not limited to)  David Arbour, Eytan Bakshy, David Eriksson, Jared Frank, Sam Stanton, Bram Wallace, Ke Alexander Wang, Ruihan Wu.

## Acknowledgements
Development of GPyTorch is supported by funding from the [Bill and Melinda Gates Foundation](https://www.gatesfoundation.org/), the [National Science Foundation](https://www.nsf.gov/), and [SAP](https://www.sap.com/index.html).
