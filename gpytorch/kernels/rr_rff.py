#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:15:05 2020

@author: danbiderman
this is as modification of the RFF kernel, that should be combined with an appropriate
GPRegressionModel_RR_RFF model object, that contains the russian roulette weights.
"""

import math
from typing import Optional

import torch
from torch import Tensor

# from ..lazy import MatmulLazyTensor, LowRankRootLazyTensor
from ..lazy import MatmulLazyTensor, RootLazyTensor
from ..models.exact_prediction_strategies import RFFPredictionStrategy
from .kernel import Kernel


class RR_RFF_Kernel(Kernel):
    r"""
    Computes a covariance matrix based on Random Fourier Features with the RBFKernel.
    Random Fourier features was originally proposed in
    'Random Features for Large-Scale Kernel Machines' by Rahimi and Recht (2008).
    Instead of the shifted cosine features from Rahimi and Recht (2008), we use
    the sine and cosine features which is a lower-variance estimator --- see
    'On the Error of Random Fourier Features' by Sutherland and Schneider (2015).
    By Bochner's theorem, any continuous kernel :math:`k` is positive definite
    if and only if it is the Fourier transform of a non-negative measure :math:`p(\omega)`, i.e.
    .. math::
        \begin{equation}
            k(x, x') = k(x - x') = \int p(\omega) e^{i(\omega^\top (x - x'))} d\omega.
        \end{equation}
    where :math:`p(\omega)` is a normalized probability measure if :math:`k(0)=1`.
    For the RBF kernel,
    .. math::
        \begin{equation}
        k(\Delta) = \exp{(-\frac{\Delta^2}{2\sigma^2})}$ and $p(\omega) =
        \exp{(-\frac{\sigma^2\omega^2}{2})}
        \end{equation}
    where :math:`\Delta = x - x'`.
    Given datapoint :math:`x\in \mathbb{R}^d`, we can construct its random Fourier features
    :math:`z(x) \in \mathbb{R}^{2D}` by
    .. math::
        \begin{equation}
        z(x) = \sqrt{\frac{1}{D}}
        \begin{bmatrix}
            \cos(\omega_1^\top x)\\
            \sin(\omega_1^\top x)\\
            \cdots \\
            \cos(\omega_D^\top x)\\
            \sin(\omega_D^\top x)
        \end{bmatrix}, \omega_1, \ldots, \omega_D \sim p(\omega)
        \end{equation}
    such that we have an unbiased Monte Carlo estimator
    .. math::
        \begin{equation}
            k(x, x') = k(x - x') \approx z(x)^\top z(x') =
            \frac{1}{D}\sum_{i=1}^D \cos(\omega_i^\top (x - x')).
        \end{equation}
    .. note::
        When this kernel is used in batch mode, the random frequencies are drawn
        independently across the batch dimension as well by default.
    :param num_samples: Number of random frequencies to draw. This is :math:`D` in the above
        papers. This will produce :math:`D` sine features and :math:`D` cosine
        features for a total of :math:`2D` random Fourier features.
    :type num_samples: int
    :param num_dims: (Default `None`.) Dimensionality of the data space.
        This is :math:`d` in the above papers. Note that if you want an
        independent lengthscale for each dimension, set `ard_num_dims` equal to
        `num_dims`. If unspecified, it will be inferred the first time `forward`
        is called.
    :type num_dims: int, optional
    :var torch.Tensor randn_weights: The random frequencies that are drawn once and then fixed.
    Example:
        >>> # This will infer `num_dims` automatically
        >>> kernel= gpytorch.kernels.RFFKernel(num_samples=5)
        >>> x = torch.randn(10, 3)
        >>> kxx = kernel(x, x).evaluate()
        >>> print(kxx.randn_weights.size())
        torch.Size([3, 5])
    """

    has_lengthscale = True

    def __init__(self, single_sample: bool = False,
                 min_val: Optional[int] = None,
                 num_dims: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.num_samples = None
        self.single_sample = single_sample
        if num_dims is not None:
            self._init_weights(num_dims, self.num_samples)

    def _init_weights(self, num_dims: Optional[int] = None, num_samples: Optional[int] = None,
                      randn_weights: Optional[Tensor] = None):
        if num_dims is not None and num_samples is not None:
            d = num_dims
            D = num_samples
        if randn_weights is None:
            randn_shape = torch.Size([*self._batch_shape, 3, d, D])
            randn_weights = torch.randn(randn_shape,
                                        dtype=self.raw_lengthscale.dtype,
                                        device=self.raw_lengthscale.device)
        self.register_buffer("randn_weights", randn_weights)

    def forward(self, x1: Tensor, x2: Tensor,
                diag: bool = False,
                last_dim_is_batch: bool = False,
                **kwargs) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        num_dims = x1.size(-1)
        self._init_weights(num_dims, self.num_samples)
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self._featurize(x1, normalize=False)
        z1 = self.expand_z(z1)
        if not x1_eq_x2:
            z2 = self._featurize(x2, normalize=False)
        else:
            z2 = z1
        D = float(self.num_samples)
        if diag:
            return (z1 * z2).sum(-1) / D
        if x1_eq_x2:
            # return LowRankRootLazyTensor(z1)
            return RootLazyTensor(z1)
        else:
            print('Warning: x1!=x2 case is not supported for RR.')
            return MatmulLazyTensor(z1, z2.transpose(-1, -2))

    def expand_z(self, z_pre):
        D = int(z_pre.shape[-1] / 2)
        ones = torch.ones(1, self.min_val, dtype=z_pre.dtype, device=z_pre.device)
        zeros = torch.zeros(1, D - self.min_val, dtype=z_pre.dtype, device=z_pre.device)
        mask = torch.cat([ones, zeros], dim=1)
        mask /= torch.sqrt(self.min_val.float())

        ones = torch.ones(1, D - 1, dtype=z_pre.dtype, device=z_pre.device)
        zeros = torch.zeros(1, 1, dtype=z_pre.dtype, device=z_pre.device)
        aux = torch.cat([ones, zeros], dim=1)
        aux /= torch.sqrt(torch.tensor(D - 1, dtype=z_pre.dtype, device=z_pre.device))

        ones = torch.ones(1, D, dtype=z_pre.dtype, device=z_pre.device)
        ones /= torch.sqrt(torch.tensor(D, dtype=z_pre.dtype, device=z_pre.device))
        mask = torch.cat([mask, aux, ones], dim=0)

        mask = mask.unsqueeze(-2)
        mask = torch.cat([mask, mask], dim=-1)

        z_new = z_pre * mask
        return z_new

    def _featurize(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Recompute division each time to allow backprop through lengthscale
        # Transpose lengthscale to allow for ARD
        x = x.matmul(self.randn_weights / self.lengthscale.transpose(-1, -2))
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        if normalize:
            D = self.num_samples
            z = z / math.sqrt(D)
        return z

    @staticmethod
    def prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood):
        # Allow for fast sampling
        return RFFPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)

# # test expand_z
# D = 7 # num_samples
# d = 3 # input dims
# torch.manual_seed(12)
# z_pre = torch.randn(size = (d, 2*D))
# z_new_test = expand_z(z_pre)
# print(z_new_test.shape)
# out_lazy = RootLazyTensor(z_new_test)
# cov = out_lazy.evaluate()
# print(cov.shape)
