#!/usr/bin/env python3

import math

import torch

from ..lazy import MatmulLazyTensor, RootLazyTensor
from ..models.exact_prediction_strategies import RFFPredictionStrategy
from .kernel import Kernel


class RFFKernel(Kernel):

    has_lengthscale = True

    def __init__(self, ard_num_dims, num_samples, **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)
        self.num_samples = num_samples
        self.num_dims = ard_num_dims

        d = ard_num_dims
        D = num_samples
        randn_shape = torch.Size([d, D])
        randn_weights = torch.randn(randn_shape, dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        self.register_buffer("randn_weights", randn_weights)

        outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        if last_dim_is_batch:
            raise RuntimeError("last_dim_is_batch not implemented for RFF")
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

    def _featurize(self, x, normalize=False):
        # Recompute division each time to allow backprop through lengthscale
        # Transpose lengthscale to allow for ARD
        x = x.matmul(self.randn_weights / self.lengthscale.transpose(-1, -2))
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        if normalize:
            D = self.num_samples
            z = z / math.sqrt(D)
        return z

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return RFFPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
