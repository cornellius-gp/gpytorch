#!/usr/bin/env python3

import torch
import math

from .kernel import Kernel
from ..lazy import MatmulLazyTensor

class RFFKernel(Kernel):

    has_lengthscale = True

    def __init__(self, input_dim, num_samples, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.input_dim = input_dim

        d = input_dim
        D = num_samples
        randn_shape = torch.Size([d, D])
        randn_weights = torch.randn(randn_shape, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        self.register_buffer("randn_weights", randn_weights)

    def forward(self, x1, x2, **kwargs):
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self._featurize(x1, normalize=False)
        if not x1_eq_x2:
            z2 = self._featurize(x2, normalize=False)
        else:
            z2 = z1
        D = self.num_samples
        return MatmulLazyTensor(z1 / D, z2.transpose(-1, -2))

    def _featurize(self, x, normalize=False):
        # Recompute division each time to allow backprop through lengthscale
        # Transpose lengthscale to allow for ARD
        x = x.matmul(self.randn_weights / self.lengthscale.transpose(-1, -2))
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        if normalize:
            D = self.num_samples
            z = z / math.sqrt(D)
        return z
