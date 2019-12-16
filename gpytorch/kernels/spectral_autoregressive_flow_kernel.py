#!/usr/bin/env python3

import math

import torch
from pyro import distributions as dist
from pyro.distributions.transforms import BlockAutoregressive

from ..lazy import MatmulLazyTensor, RootLazyTensor
from .kernel import Kernel


class SpectralAutoregressiveFlowKernel(Kernel):
    def __init__(self, num_dims, stack_size=1, **kwargs):
        super(SpectralAutoregressiveFlowKernel, self).__init__(has_lengthscale=True, **kwargs)
        if stack_size > 1:
            self.dsf = torch.nn.ModuleList([BlockAutoregressive(num_dims, **kwargs) for _ in range(stack_size)])
        else:
            self.dsf = BlockAutoregressive(num_dims, **kwargs)
        self.num_dims = num_dims

    def _create_input_grid(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the gridded `x1` and `x2`.
            The shape depends on the kernel's mode

            * `full_covar`: (`b x n x 1 x d` and `b x 1 x m x d`)
            * `full_covar` with `last_dim_is_batch=True`: (`b x k x n x 1 x 1` and `b x k x 1 x m x 1`)
            * `diag`: (`b x n x d` and `b x n x d`)
            * `diag` with `last_dim_is_batch=True`: (`b x k x n x 1` and `b x k x n x 1`)
        """
        x1_, x2_ = x1, x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            if torch.equal(x1, x2):
                x2_ = x1_
            else:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        if diag:
            return x1_, x2_
        else:
            return x1_.unsqueeze(-2), x2_.unsqueeze(-3)

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, diag=diag)

        diffs = x1_ - x2_
        base_dist = dist.Normal(
            torch.zeros(self.num_dims, device=x1.device, dtype=x1.dtype),
            torch.ones(self.num_dims, device=x1.device, dtype=x1.dtype),
        )
        if isinstance(self.dsf, torch.nn.ModuleList):
            dsf = list(self.dsf)
        else:
            dsf = self.dsf
        dsf_dist = dist.TransformedDistribution(base_dist, dsf)
        if self.training:
            Z = dsf_dist.rsample(torch.Size([2000]))
        else:
            Z = dsf_dist.rsample(torch.Size([2000]))

        if diag:
            diffs_times_Z = (Z.unsqueeze(-2) * diffs.unsqueeze(-3)).sum(-1)
            K = diffs_times_Z.mul(2 * math.pi).cos().mean(dim=-2)
        else:
            diffs_times_Z = (Z.unsqueeze(-2).unsqueeze(-2) * diffs.unsqueeze(-4)).sum(-1)
            K = diffs_times_Z.mul(2 * math.pi).cos().mean(dim=-3)
        return K


class NSSpectralAutoregressiveFlowKernel(SpectralAutoregressiveFlowKernel):
    def __init__(self, num_dims, stack_size=1, **kwargs):
        Kernel.__init__(self, has_lengthscale=True, **kwargs)
        if stack_size > 1:
            self.dsf = torch.nn.ModuleList([BlockAutoregressive(2 * num_dims, **kwargs) for _ in range(stack_size)])
        else:
            self.dsf = BlockAutoregressive(2 * num_dims, **kwargs)

        self.num_dims = num_dims

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, diag=diag)

        base_dist = dist.Normal(
            torch.zeros(self.num_dims * 2, device=x1.device, dtype=x1.dtype),
            torch.ones(self.num_dims * 2, device=x1.device, dtype=x1.dtype),
        )

        if isinstance(self.dsf, torch.nn.ModuleList):
            dsf = list(self.dsf)
        else:
            dsf = self.dsf

        dsf_dist = dist.TransformedDistribution(base_dist, dsf)

        Z = dsf_dist.rsample(torch.Size([128]))
        Z1 = Z[:, : self.num_dims]
        Z2 = Z[:, self.num_dims :]

        Z1_ = torch.cat(([Z1, Z2, -Z1, -Z2]))
        Z2_ = torch.cat(([Z1, Z2, -Z1, -Z2]))

        Z1_ = Z1_.unsqueeze(1)
        Z2_ = Z2_.unsqueeze(1)
        if not diag:
            Z1_ = Z1_.unsqueeze(1)
            Z2_ = Z2_.unsqueeze(1)

        x1z1 = x1_ * Z1_  # s x n x 1 x d
        x2z2 = x2_ * Z2_  # s x 1 x n x d

        x1z1 = x1z1.sum(-1)  # s x n x 1
        x2z2 = x2z2.sum(-1)  # s x 1 x n

        diff = x1z1 - x2z2  # s x n x n
        K = diff.mul(2 * math.pi).cos().mean(0)  # n x n

        return K


class NSSpectralDeltaKernel(SpectralAutoregressiveFlowKernel):
    def __init__(self, num_dims, num_deltas=128, **kwargs):
        Kernel.__init__(self, has_lengthscale=True, **kwargs)

        self.Z = torch.nn.Parameter(torch.randn(num_deltas, 2 * num_dims))

        self.num_dims = num_dims

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        x1_, x2_ = self._create_input_grid(x1_, x2_, diag=diag)

        Z = self.Z

        Z1 = Z[:, : self.num_dims]
        Z2 = Z[:, self.num_dims :]

        Z1_ = torch.cat(([Z1, Z2, -Z1, -Z2]))
        Z2_ = torch.cat(([Z1, Z2, -Z1, -Z2]))

        Z1_ = Z1_.unsqueeze(1)
        Z2_ = Z2_.unsqueeze(1)
        if not diag:
            Z1_ = Z1_.unsqueeze(1)
            Z2_ = Z2_.unsqueeze(1)

        x1z1 = x1_ * Z1_  # s x n x 1 x d
        x2z2 = x2_ * Z2_  # s x 1 x n x d

        x1z1 = x1z1.sum(-1)  # s x n x 1
        x2z2 = x2z2.sum(-1)  # s x 1 x n

        diff = x1z1 - x2z2  # s x n x n
        K = diff.mul(2 * math.pi).cos().mean(0)  # n x n

        return K


class RFNSSpectralDeltaKernel(NSSpectralDeltaKernel):
    def __init__(self, num_dims, num_deltas=128, nonstationary=False, **kwargs):
        Kernel.__init__(self, has_lengthscale=True, **kwargs)

        if nonstationary:
            self.Z = torch.nn.Parameter(torch.randn(num_deltas, 2 * num_dims))
        else:
            self.Z = torch.nn.Parameter(torch.randn(num_deltas, num_dims))

        self.nonstationary = nonstationary
        self.num_dims = num_dims

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        Z = self.Z

        if self.nonstationary:
            Z1 = Z[:, : self.num_dims]
            Z2 = Z[:, self.num_dims :]

            Z1_ = torch.cat(([Z1, Z2, -Z1, -Z2]))
            Z2_ = torch.cat(([Z1, Z2, -Z1, -Z2]))
        else:
            Z1_ = Z
            Z2_ = Z

        # Z1_ and Z2_ are s x d
        x1z1 = x1_.matmul(Z1_.transpose(-2, -1))  # n x s
        x2z2 = x2_.matmul(Z2_.transpose(-2, -1))  # n x s

        x1z1 = x1z1 * 2 * math.pi
        x2z2 = x2z2 * 2 * math.pi

        x1z1 = torch.cat([x1z1.cos(), x1z1.sin()], dim=-1) / math.sqrt(x1z1.size(-1))
        x2z2 = torch.cat([x2z2.cos(), x2z2.sin()], dim=-1) / math.sqrt(x2z2.size(-1))

        if x1.size() == x2.size() and torch.equal(x1, x2):
            prod = RootLazyTensor(x1z1)
        else:
            prod = MatmulLazyTensor(x1z1, x2z2.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod


class RFNSSpectralNFKernel(NSSpectralDeltaKernel):
    def __init__(self, num_dims, stack_size=1, nonstationary=False, **kwargs):
        Kernel.__init__(self, has_lengthscale=True, **kwargs)

        if nonstationary:
            ndims = 2 * num_dims
        else:
            ndims = num_dims

        if stack_size > 1:
            self.dsf = torch.nn.ModuleList([BlockAutoregressive(ndims, **kwargs) for _ in range(stack_size)])
        else:
            self.dsf = BlockAutoregressive(ndims, **kwargs)

        self.num_dims = num_dims
        self.ndims = ndims
        self.nonstationary = nonstationary

    def Z(self, x1, x2):
        base_dist = dist.Normal(
            torch.zeros(self.ndims, device=x1.device, dtype=x1.dtype),
            torch.ones(self.ndims, device=x1.device, dtype=x1.dtype),
        )

        if isinstance(self.dsf, torch.nn.ModuleList):
            dsf = list(self.dsf)
        else:
            dsf = self.dsf

        dsf_dist = dist.TransformedDistribution(base_dist, dsf)

        return dsf_dist.rsample(torch.Size([128]))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        Z = self.Z(x1, x2)

        if self.nonstationary:
            Z1 = Z[:, : self.num_dims]
            Z2 = Z[:, self.num_dims :]

            Z1_ = torch.cat(([Z1, Z2, -Z1, -Z2]))
            Z2_ = torch.cat(([Z1, Z2, -Z1, -Z2]))
        else:
            Z1_ = Z
            Z2_ = Z

        # Z1_ and Z2_ are s x d
        x1z1 = x1_.matmul(Z1_.transpose(-2, -1))  # n x s
        x2z2 = x2_.matmul(Z2_.transpose(-2, -1))  # n x s

        x1z1 = torch.cat([x1z1.cos(), x1z1.sin()], dim=-1) / math.sqrt(x1z1.size(-1))
        x2z2 = torch.cat([x2z2.cos(), x2z2.sin()], dim=-1) / math.sqrt(x2z2.size(-1))

        if x1.size() == x2.size() and torch.equal(x1, x2):
            prod = RootLazyTensor(x1z1)
        else:
            prod = MatmulLazyTensor(x1z1, x2z2.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod
