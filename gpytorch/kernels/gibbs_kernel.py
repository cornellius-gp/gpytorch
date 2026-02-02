#!/usr/bin/env python3


import torch

from .kernel import Kernel


class GibbsKernel(Kernel):
    r"""Computes a covariance matrix based on the gibbs kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    """

    has_lengthscale = False

    def __init__(
        self, lengthscale1, lengthscale2, ard_num_dims: int = 1, batch_shape: torch.Size = torch.Size([]), **kwargs
    ):
        super().__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)
        self.lengthscale1 = lengthscale1
        self.lengthscale2 = lengthscale2

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1
        x2_ = x2
        diff = (x1_.unsqueeze(-2) - x2_.unsqueeze(-3)).pow(2)
        square_term = (self.lengthscale1).pow(2).unsqueeze(-2) + (self.lengthscale2).pow(2).unsqueeze(-3)
        prod_term = 2 * (self.lengthscale1) * (self.lengthscale2)
        res = (prod_term / square_term).pow(0.5).prod(dim=-1) * ((-(diff / square_term).sum(dim=-1)).exp_())
        if diag:
            res = res.squeeze(0)
        return res
