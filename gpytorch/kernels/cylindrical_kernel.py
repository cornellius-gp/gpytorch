#!/usr/bin/env python3

import torch
from .kernel import Kernel
from ..utils.transforms import _get_inv_param_transform
from torch.nn.functional import softplus
from .. import settings


class CylindricalKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Cylindrical Kernel between
    inputs :math:`mathbf{x_1}` and :math:`mathbf{x_2}`.
    It was proposed in `BOCK: Bayesian Optimization with Cylindrical Kernels`.
    See http://proceedings.mlr.press/v80/oh18a.html for more details

    .. note::
        The data must lie completely within the unit ball.

    Args:
        :attr:`num_angular_weights` (int):
            The number of components in the angular kernel
        :attr:`radial_base_kernel` (gpytorch.kernel):
            The base kernel for computing the radial kernel
        :attr:`batch_size` (int, optional):
            Set this if the data is batch of input data.
            It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `1`
        :attr:`eps` (float):
            Small floating point number used to improve numerical stability
            in kernel computations. Default: `1e-6`
        :attr:`param_transform` (function, optional):
            Set this if you want to use something other than softplus to ensure positiveness of parameters.
        :attr:`inv_param_transform` (function, optional):
            Set this to allow setting parameters directly in transformed space and sampling from priors.
            Automatically inferred for common transformations such as torch.exp or torch.nn.functional.softplus.

    """
    def __init__(
        self,
        num_angular_weights,
        radial_base_kernel,
        batch_size=1,
        eps=1e-6,
        angular_weights_prior=None,
        alpha_prior=None,
        beta_prior=None,
        param_transform=softplus,
        inv_param_transform=None,
        **kwargs
    ):
        super().__init__(has_lengthscale=False, batch_size=batch_size)
        self.num_angular_weights = num_angular_weights
        self.radial_base_kernel = radial_base_kernel
        self.eps = eps
        self._param_transform = param_transform
        self._inv_param_transform = _get_inv_param_transform(param_transform, inv_param_transform)
        self.register_parameter(name="raw_angular_weights",
                                parameter=torch.nn.Parameter(torch.zeros(batch_size, num_angular_weights)))
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(batch_size)))
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(batch_size)))
        if angular_weights_prior is not None:
            self.register_prior(
                "angular_weights_prior", angular_weights_prior,
                lambda: self.angular_weights, lambda v: self._set_angular_weights(v)
            )
        if alpha_prior is not None:
            self.register_prior(
                "alpha_prior", alpha_prior, lambda: self.alpha, lambda v: self._set_alpha(v)
            )
        if beta_prior is not None:
            self.register_prior(
                "beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v)
            )

    @property
    def angular_weights(self):
        return self._param_transform(self.raw_angular_weights)

    @angular_weights.setter
    def angular_weights(self, value):
        self._set_param(value, 'angular_weights')

    @property
    def alpha(self):
        return self._param_transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_param(value, 'alpha')

    @property
    def beta(self):
        return self._param_transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self._set_param(value, 'beta')

    def _set_param(self, value, param_name):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(**{'raw_{}'.format(param_name): self._inv_param_transform(value)})

    def forward(self, x1, x2, diag=False, **params):
        x1_, x2_ = x1.clone(), x2.clone()
        # Jitter datapoints that are exactly 0
        x1_[x1_ == 0], x2_[x2_ == 0] = x1_[x1_ == 0] + self.eps, x2_[x2_ == 0] + self.eps
        r1, r2 = x1_.norm(dim=-1, keepdim=True), x2_.norm(dim=-1, keepdim=True)
        a1, a2 = x1.div(r1), x2.div(r2)
        if not diag:
            gram_mat = a1.matmul(a2.transpose(-2, -1))
            # Note: This can be made more efficient with a custom backard and
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None, None].mul(gram_mat.pow(p))
        else:
            gram_mat = a1.mul(a2).sum(-1)
            for p in range(self.num_angular_weights):
                if p == 0:
                    angular_kernel = self.angular_weights[..., 0, None]
                else:
                    angular_kernel = angular_kernel + self.angular_weights[..., p, None].mul(gram_mat.pow(p))

        with settings.lazily_evaluate_kernels(False):
            radial_kernel = self.radial_base_kernel(self.kuma(r1), self.kuma(r2), diag=diag, **params)
        return radial_kernel.mul(angular_kernel)

    def kuma(self, x):
        # Add numerical stability to gradient computation
        res = 1 - (1 - x.pow(self.alpha) + self.eps).pow(self.beta)
        return res

    def size(self, x1, x2):
        return self.radial_base_kernel.size(x1, x2)
