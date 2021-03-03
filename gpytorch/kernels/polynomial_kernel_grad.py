#!/usr/bin/env python3

from typing import Optional

import torch

from .polynomial_kernel import PolynomialKernel


class PolynomialKernelGrad(PolynomialKernel):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: Optional[bool] = False,
        last_dim_is_batch: Optional[bool] = False,
        **params,
    ) -> torch.Tensor:
        offset = self.offset.view(*self.batch_shape, 1, 1)

        batch_shape = x1.shape[:-2]
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        if diag:
            base_diag = (x1 * x2).sum(dim=-1) + self.offset
            K11_diag = base_diag.pow(self.power)

            all_outers_diag = (x1 * x2).transpose(-2, -1).reshape(*batch_shape, -1)
            K22_base_diag = self.power * (self.power - 1) * base_diag.pow(self.power - 2)
            K12_base_diag = self.power * base_diag.pow(self.power - 1)

            K22_diag = torch.add(
                all_outers_diag * K22_base_diag.repeat(*([1] * (K22_base_diag.dim() - 1)), d),
                K12_base_diag.repeat(*([1] * (K12_base_diag.dim() - 1)), d),
            )

            K_diag = torch.cat([K11_diag, K22_diag], dim=-1)
            # Apply perfect shuffle
            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
            K_diag = K_diag[..., pi1]
            return K_diag

        else:
            base_inner_prod = torch.matmul(x1, x2.transpose(-2, -1)) + offset
            K11 = base_inner_prod.pow(self.power)

            K12_base = self.power * base_inner_prod.pow(self.power - 1)
            K12 = torch.zeros(*batch_shape, n1, n2 * d, dtype=x1.dtype, device=x1.device)

            ones_ = torch.ones(*batch_shape, d, 1, n2, dtype=x1.dtype, device=x1.device)
            K12_outer_prods = torch.matmul(x1.transpose(-2, -1).unsqueeze(-1), ones_)
            K12 = (K12_base.unsqueeze(-3) * K12_outer_prods).transpose(-3, -2).reshape(*batch_shape, n1, d * n2)

            ones_ = torch.ones(*batch_shape, d, n1, 1, dtype=x1.dtype, device=x1.device)
            K21_outer_prods = torch.matmul(ones_, x2.transpose(-2, -1).unsqueeze(-2))
            K21 = (K12_base.unsqueeze(-3) * K21_outer_prods).view(*batch_shape, d * n1, n2)

            K22_base = self.power * (self.power - 1) * base_inner_prod.pow(self.power - 2)
            K22 = torch.zeros(*batch_shape, n1 * d, n2 * d, dtype=x1.dtype, device=x1.device)
            all_outers = x1.unsqueeze(-2).unsqueeze(-2).transpose(-2, -1).matmul(x2.unsqueeze(-3).unsqueeze(-2))
            all_outers = all_outers.transpose(-4, -2).transpose(-3, -1)
            K22 = K22_base.unsqueeze(-3).unsqueeze(-3) * all_outers  # d x d x n1 x n2

            # Can't avoid this for loop without unnecessary memory duplication, which is worse.
            for i in range(d):
                K22[..., i, i, :, :] = K22[..., i, i, :, :] + K12_base

            K22 = K22.transpose(-4, -3).transpose(-3, -2).reshape(*batch_shape, n1 * d, n2 * d)

            K = torch.cat([torch.cat([K11, K12], dim=-1), torch.cat([K21, K22], dim=-1)], dim=-2)

            # Apply perfect shuffle
            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
            pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
            K = K[..., pi1, :][..., :, pi2]

            return K

    def num_outputs_per_input(self, x1, x2):
        return x1.size(-1) + 1
