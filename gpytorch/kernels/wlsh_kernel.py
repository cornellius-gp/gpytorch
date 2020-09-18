#!/usr/bin/env python3

from typing import Optional

import torch
from torch import Tensor

from ..distributions import Distribution
from ..lazy import DiagLazyTensor, InterpolatedLazyTensor
from .kernel import Kernel


class WLSHKernel(Kernel):
    r"""
    """

    has_lengthscale = True

    def __init__(
        self,
        num_samples: int,
        num_dims: Optional[int] = None,
        ard_num_dims: int = 1,
        hash_distribution: Optional[Distribution] = None,
        smooth=True,
        **kwargs,
    ):
        if num_dims is None and ard_num_dims == 1:
            raise ValueError("Either num_dims or ard_num_dims must be supplied to the constructor.")
        elif ard_num_dims != 1 and num_dims != ard_num_dims:
            raise ValueError(f"ard_num_dims and num_dims do not match: got {ard_num_dims} and {num_dims}")
        else:
            num_dims = num_dims or ard_num_dims

        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        self.num_samples = num_samples
        self.num_dims = num_dims
        self.smooth = smooth

        # (Maybe) construct hashing distribuiton
        if hash_distribution is None:
            concentration = torch.full((self.num_dims,), fill_value=5.0 if smooth else 2.0)
            rate = torch.ones_like(concentration)
            hash_distribution = torch.distributions.Gamma(concentration=concentration, rate=rate)
        self.hash_distribution = hash_distribution

        # Construct hash bins and locations
        hash_bins_ws = self.hash_distribution.sample(torch.Size([self.num_samples])).unsqueeze(-2)
        hash_bins_zs = torch.rand_like(hash_bins_ws).mul(hash_bins_ws)
        self.register_buffer("hash_bins_ws", hash_bins_ws)
        self.register_buffer("hash_bins_zs", hash_bins_zs)

    def _smoothed_rect(self, x):
        if self.smooth:
            scalar = lambda i: torch.tensor(i, dtype=x.dtype, device=x.device)
            y_sign = torch.where(
                x.abs().lt(0.375),
                torch.where(x.abs().lt(0.25), scalar(0.0), scalar(-0.5)),
                torch.where(x.abs().gt(0.5), scalar(0.0), scalar(0.5)),
            )
            x_offset = torch.where(
                x.lt(0),
                torch.where(x.lt(-0.375), scalar(0.5), scalar(0.25)),
                torch.where(x.gt(0.375), scalar(-0.5), scalar(-0.25)),
            )
            y_offset = torch.where(x.abs().lt(0.375), scalar(1.0), scalar(0.0))
            y = (8 * (x + x_offset)) ** 2 * y_sign + y_offset
        else:  # use rect
            y = ((x > -0.5) & (x < 0.5)).type_as(x)  # not actually necessary since arg always in domain i think
        return y.prod(dim=-1)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **kwargs) -> Tensor:
        x1_eq_x2 = torch.equal(x1, x2)
        x1_ = x1.div(self.lengthscale).sub(self.hash_bins_zs).div(self.hash_bins_ws)
        x2_ = x1_ if x1_eq_x2 else x2.div(self.lengthscale).sub(self.hash_bins_zs).div(self.hash_bins_ws)

        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        # Compute rounded hash_ids
        hash_ids1 = x1_.round()
        hash_ids2 = hash_ids1 if x1_eq_x2 else x2_.round()

        # Compute kernel features
        features1 = self._smoothed_rect(hash_ids1 - x1_)
        features2 = features1 if x1_eq_x2 else self._smoothed_rect(hash_ids2 - x2_)

        # Assign hash bins to each input
        with torch.no_grad():
            flattened_hash_ids1 = hash_ids1.view(-1, *hash_ids1.shape[-2:])
            flattened_hash_bins1 = torch.stack(
                [
                    torch.unique(hash_ids, sorted=False, return_inverse=True, dim=-2)[1]
                    for hash_ids in flattened_hash_ids1
                ]
            )
            hash_bins1 = flattened_hash_bins1.view(*flattened_hash_ids1.shape[:-1])
            if x1_eq_x2:
                hash_bins2 = hash_bins1
            else:
                flattened_hash_ids2 = hash_ids2.view(-1, *hash_ids2.shape[-2:])
                flattened_hash_bins2 = torch.stack(
                    [
                        torch.unique(hash_ids, sorted=False, return_inverse=True, dim=-2)[1]
                        for hash_ids in flattened_hash_ids2
                    ]
                )
                hash_bins2 = flattened_hash_bins2.view(*flattened_hash_ids2.shape[:-1])

        # Compute interpolated lazy tensor
        num_bins = max(hash_bins1.max().item(), hash_bins2.max().item()) + 1
        res = (
            InterpolatedLazyTensor(
                base_lazy_tensor=DiagLazyTensor(torch.ones(num_bins, dtype=x1.dtype, device=x1.device)),
                left_interp_indices=hash_bins1.unsqueeze(-1),
                left_interp_values=features1.unsqueeze(-1),
                right_interp_indices=hash_bins2.unsqueeze(-1),
                right_interp_values=features2.unsqueeze(-1),
            )
            .evaluate()
            .sum(dim=0)
            .mul(1.0 / self.num_samples)
        )

        if diag:
            return res.diag()
        return res
