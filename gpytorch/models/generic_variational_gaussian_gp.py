#!/usr/bin/env python3
import math
import torch
import pyro
from .. import Module
from ..lazy import RootLazyTensor, DiagLazyTensor, BlockDiagLazyTensor
from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from . import GP
from .. import settings
from ..variational import CholeskyVariationalDistribution
from .generic_variational_particle_gp import GenericVariationalParticleGP


class GenericVariationalGaussianGP(GenericVariationalParticleGP):
    def __init__(self, inducing_points, likelihood, num_data, name_prefix="",
                 mode="predictive",beta=1.0, divbeta=0.1):
        super().__init__(
            inducing_points,
            likelihood,
            num_data,
            name_prefix=name_prefix,
            mode=mode,
            beta=beta,
            divbeta=divbeta,
        )

        self.variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(-2))

    def guide(self):
        beta = self.beta if self.beta > 0.0 else 1.0e-20
        with pyro.poutine.scale(scale=beta / self.num_data):
            return pyro.sample(self.name_prefix + ".inducing_values", self.variational_distribution)

    def __call__(self, input, *args, **kwargs):
        inducing_points = self.inducing_points
        inducing_batch_shape = inducing_points.shape[:-2]
        if inducing_batch_shape < input.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], input.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            input = input.expand(*batch_shape, *input.shape[-2:])

        # Draw samples from p(u) for KL divergence computation
        inducing_values_samples = self.sample_inducing_values()

        var_dist = self.variational_distribution.variational_distribution
        var_mean = var_dist.mean
        var_covar = var_dist.lazy_covariance_matrix

        # Get function dist
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, input], dim=-2)
        full_output = self.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        test_mean = full_output.mean[..., num_induc:]
        L = full_covar[..., :num_induc, :num_induc].add_jitter().cholesky().evaluate()
        cross_covar = full_covar[..., :num_induc, num_induc:].evaluate()  # K_{ux}
        scaled_cross_covar = torch.triangular_solve(cross_covar, L, upper=False)[0]  # L^{-1}K_{ux}
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # K_xuL^{-1}SL^{-1}K_ux <=> K_{xu}K_{uu}^{-1}\hat{S}K_{uu}^{-1}K_{ux}
        covar_term = scaled_cross_covar.transpose(-2, -1) @ var_covar.matmul(scaled_cross_covar)

        function_dist = MultivariateNormal(
            (scaled_cross_covar.transpose(-1, -2) @ var_mean),  # K_{xu}L^{-1}m  <=> K_{xu}K_{uu}^{-1}\hat{m},
            data_data_covar + covar_term + RootLazyTensor(scaled_cross_covar.transpose(-1, -2)).mul(-1),
        )
        return function_dist
