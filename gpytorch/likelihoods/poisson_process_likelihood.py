#!/usr/bin/env python3

import torch
import numpy as np
from scipy.special import hyp1f1
from .likelihood import Likelihood
from ..utils.interpolation import Interpolation, left_interp
from ..lazy import LazyEvaluatedKernelTensor
from .. import settings


EULER_MASCHERONI_CONSTANT = 0.5772156649015329


def hypergeo_approx(a, b, z, eps=10e-10):
    left_approx = (hyp1f1(a, b, z) - hyp1f1(a - eps, b, z)) / eps
    return left_approx


class PoissonProcessLikelihood(Likelihood):
    def __init__(
        self,
        min_bounds,
        max_bounds,
        hypergeo_grid_size=30000,
        hypergeo_grid_lower=-1000,
        hypergeo_grid_upper=-1e-5
    ):
        super().__init__()

        self.hypergeo_grid_size = hypergeo_grid_size
        self.hypergeo_grid_lower = hypergeo_grid_lower
        self.hypergeo_grid_upper = hypergeo_grid_upper

        x_grid_np = np.linspace(hypergeo_grid_lower, hypergeo_grid_upper, hypergeo_grid_size)
        y_grid_np = np.array([hypergeo_approx(0, 0.5, i) for i in x_grid_np])

        x_grid = torch.from_numpy(x_grid_np).float().unsqueeze(-1)
        y_grid = torch.from_numpy(y_grid_np).float()

        self.register_buffer("hypergeom_grid", x_grid)
        self.register_buffer("hypergeom_vals", y_grid)

        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def _volume(self):
        return (self.max_bounds - self.min_bounds).prod()

    def _hypergeo(self, x):
        x = x.clamp(self.hypergeo_grid_lower, self.hypergeo_grid_upper)
        squeeze_at_end = False
        if x.dim() == 1:
            squeeze_at_end = True
            x = x.unsqueeze(-1)
        interp_inds, interp_values = Interpolation().interpolate(self.hypergeom_grid, x)
        res = left_interp(interp_inds, interp_values, self.hypergeom_vals)
        if squeeze_at_end:
            res = res.squeeze(-1)
        return res

    def _integrate_moments(self, model):
        variational_dist = model.variational_strategy.variational_distribution.variational_distribution
        variational_mean = variational_dist.mean.unsqueeze(-1)
        chol_variational_covar = variational_dist.lazy_covariance_matrix._chol

        inducing_points = model.variational_strategy.inducing_points
        inducing_dist = model.forward(inducing_points)
        inducing_covar = inducing_dist.lazy_covariance_matrix

        if not isinstance(inducing_covar, LazyEvaluatedKernelTensor):
            raise RuntimeError('Models using a Poisson process likelihood must return a covar call directly.')

        kernel = inducing_covar.kernel
        inducing_covar = inducing_covar.add_jitter()
        psi_matrix = kernel.integrate_inner(
            inducing_points,
            inducing_points,
            self.min_bounds,
            self.max_bounds
        ).squeeze(0)

        # TODO: Obviously, do this a better way.
        Kzz_inv = inducing_covar.evaluate().inverse()

        mu = variational_mean.squeeze()
        S = chol_variational_covar.matmul(chol_variational_covar.transpose(-2, -1))

        int_mean = mu.matmul(Kzz_inv.matmul(psi_matrix.matmul(Kzz_inv.matmul(mu))))
        int_var = kernel.outputscale.item() * self._volume() - torch.trace(Kzz_inv.matmul(psi_matrix)) + torch.trace(Kzz_inv.matmul(S.matmul(Kzz_inv.matmul(psi_matrix))))

        return int_mean + int_var

    def forward(self, input):
        raise NotImplementedError

    def variational_log_probability(self, latent_func, target, model):
        q_mu_squared = latent_func.mean.pow(2)
        q_var = latent_func.variance
        scaled_mean = q_mu_squared / (2 * q_var)

        num_samples = 1000
        samples = latent_func.rsample(torch.Size([num_samples]))
        expected_log_probs_sample = samples.pow(2).log().mean(-2).sum()

        # Easy term
        expected_log_probs = -self._hypergeo(-scaled_mean) + (q_var / 2).log() - EULER_MASCHERONI_CONSTANT
        expected_log_prob = expected_log_probs.sum()

        # print(expected_log_prob, expected_log_probs_sample)

        # Hard term
        moment_integral = self._integrate_moments(model)
        log_prob = (-moment_integral + expected_log_prob)
        print(expected_log_probs_sample.item(), moment_integral.item())
        return log_prob

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        raise NotImplementedError
