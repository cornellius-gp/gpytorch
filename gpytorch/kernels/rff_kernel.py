#!/usr/bin/env python3

import math
from typing import Optional

import torch
from torch import Tensor

from ..lazy import MatmulLazyTensor, RootLazyTensor
from .kernel import Kernel


class RFFKernel(Kernel):
    r"""
    Computes a covariance matrix based on Random Fourier Features with the
    RBFKernel.

    See Random Features for Large-Scale Kernel Machines by Rahimi and Recht

    Here we use sine and cosine features which gives a lower-variance estimator.
    See On the Error of Random Fourier Features by Sutherland and Schneider.

    Args:
        :attr:`num_samples` (int):
            Number of random frequencies to draw. This is :math:`D` in the above
            papers. This will produce :math:`D` sine features and :math:`D`
            cosine features.
        :attr:`num_dims` (Optional[int]):
            Dimensionality of the data space. This is :math:`d` in the above
            papers. Note that if you want an independent lengthscale for each
            dimension, set `ard_num_dims` equal to `num_dims`. If unspecified,
            it will be inferred the first time `forward` is called.

    Attributes:
        :attr:`randn_weights` (Tensor):
            The random frequencies that are drawn once at initialization and
            fixed.
    """

    has_lengthscale = True

    def __init__(self, num_samples: int, num_dims: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        if num_dims is not None:
            self._init_weights(num_dims, num_samples)

    def _init_weights(
        self, num_dims: Optional[int] = None, num_samples: Optional[int] = None, randn_weights: Optional[Tensor] = None
    ):
        if num_dims is not None and num_samples is not None:
            d = num_dims
            D = num_samples
        if randn_weights is None:
            randn_shape = torch.Size([*self._batch_shape, d, D])
            randn_weights = torch.randn(
                randn_shape, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device
            )
        self.register_buffer("randn_weights", randn_weights)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **kwargs) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        num_dims = x1.size(-1)
        if not hasattr(self, "randn_weights"):
            self._init_weights(num_dims, self.num_samples)
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self._featurize(x1, normalize=False)
        if not x1_eq_x2:
            z2 = self._featurize(x2, normalize=False)
        else:
            z2 = z1
        D = float(self.num_samples)
        if diag:
            return (z1 * z2).sum(-1) / D
        if x1_eq_x2:
            return RootLazyTensor(z1 / math.sqrt(D))
        else:
            return MatmulLazyTensor(z1 / D, z2.transpose(-1, -2))

    def _featurize(self, x: Tensor, normalize: bool = False) -> Tensor:
        # Recompute division each time to allow backprop through lengthscale
        # Transpose lengthscale to allow for ARD
        x = x.matmul(self.randn_weights / self.lengthscale.transpose(-1, -2))
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        if normalize:
            D = self.num_samples
            z = z / math.sqrt(D)
        return z
