#!/usr/bin/env python3

import torch
from .kernel import Kernel
from ..lazy import KroneckerProductLazyTensor
from .polynomial_kernel import PolynomialKernel
from typing import Optional
from ..priors import Prior
from ..constraints import Positive, Interval


class PolynomialKernelGrad(PolynomialKernel):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: Optional[bool] = False,
        last_dim_is_batch: Optional[bool] = False,
        **params
    ) -> torch.Tensor:
        offset = self.offset.view(*self.batch_shape, 1, 1)

        batch_shape = x1.shape[:-2]
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        if diag:
            raise RuntimeError("None done yet")
            K11 = ((x1 * x2).sum(dim=-1) + self.offset).pow(self.power)
        else:
            base_inner_prod = torch.matmul(x1, x2.transpose(-2, -1)) + offset
            K11 = base_inner_prod.pow(self.power)

            K12_base = self.power * base_inner_prod.pow(self.power - 1)
            K12 = torch.zeros((*batch_shape, n1, n2 * d))
            for i in range(d):
                K12[:, n2*i:n2*(i+1)] =  K12_base * torch.ger(x1[:, i], torch.ones(n2,))

            K21 = torch.zeros((*batch_shape, n1 * d, n2))
            for i in range(d):
                K21[n1*i:n1*(i+1), :] = torch.ger(torch.ones(n1,), x2[:, i]) * K12_base

            K22_base = self.power * (self.power - 1) * base_inner_prod.pow(self.power - 2)
            K22 = torch.zeros(n1*d, n2*d)
            for i in range(d):
                for j in range(d):
                    K22[n1*i:n1*(i+1), n2*j:n2*(j+1)] = K22_base * torch.ger(x1[:, j], x2[:, i]) + (i == j) * K12_base

            K = torch.cat([torch.cat([K11, K12], dim=-1), torch.cat([K21, K22], dim=-1)])

            # Apply perfect shuffle
            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().contiguous().view((n1 * (d + 1)))
            pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            K = K[..., pi1, :][..., :, pi2]

            return K

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1) + 1
