#!/usr/bin/env python3

import torch
import pyro
from .abstract_variational_gp import AbstractVariationalGP


class QuadratureDist(pyro.distributions.Distribution):
    def __init__(self, likelihood, function_dist):
        self.likelihood = likelihood
        self.function_dist = function_dist

    def log_prob(self, target):
        return self.likelihood.expected_log_prob(target, self.function_dist)

    def sample(self, sample_shape=torch.Size()):
        pass



class PyroVariationalGP(AbstractVariationalGP):
    def __init__(self, variational_strategy, likelihood, num_data, name_prefix=""):
        super(PyroVariationalGP, self).__init__(variational_strategy)
        from pyro.nn import AutoRegressiveNN
        import pyro.distributions as dist
        self.name_prefix = name_prefix
        self.likelihood = likelihood
        self.num_data = num_data
        self.num_inducing = variational_strategy.inducing_points.size(-2)
        print(self.num_inducing)
        self.iaf1 = dist.InverseAutoregressiveFlow(AutoRegressiveNN(self.num_inducing, [self.num_inducing]))

    @property
    def variational_distribution(self):
        pyro.module("my_iaf1", self.iaf1)  # doctest: +SKIP
        import pyro.distributions as dist
        base_dist = dist.Normal(torch.zeros(self.num_inducing).cuda(), torch.ones(self.num_inducing).cuda())
        return dist.TransformedDistribution(base_dist, [self.iaf1])

    def guide(self, input, output, *params, **kwargs):
        # Draw samples from q(u) for KL divergence computation
        self.sample_inducing_values(self.variational_distribution)
        self.likelihood.guide(*params, **kwargs)

    def __call__(self, input):
        inducing_points = self.variational_strategy.inducing_points
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, input], dim=-2)
        full_output = self.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        induc_mean = full_mean[..., :num_induc]
        test_mean = full_mean[..., num_induc:]

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        from ..lazy import CholLazyTensor, DiagLazyTensor
        induc_induc_covar = CholLazyTensor(induc_induc_covar.cholesky())
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Prior distribution + samples
        inducing_values_samples = self.sample_inducing_values(self.variational_distribution)

        if inducing_values_samples.dim() == 1:
            means = induc_induc_covar.inv_matmul(
                induc_data_covar,
                inducing_values_samples.unsqueeze(-2)
            ).squeeze(-2)
        elif inducing_values_samples.dim() == 3:
            means = induc_induc_covar.inv_matmul(
                induc_data_covar,
                inducing_values_samples.squeeze(-2)
            )
        else:
            means = induc_induc_covar.inv_matmul(
                induc_data_covar,
                inducing_values_samples
            )
        f_samples = full_output.__class__(
            means,
            DiagLazyTensor((
                data_data_covar.diag() - induc_induc_covar.inv_quad(induc_data_covar, reduce_inv_quad=False)
            ).clamp_min(0))
        )
        return f_samples

    def model(self, input, output, *params, **kwargs):
        pyro.module(self.name_prefix + ".gp_prior", self)

        inducing_points = self.variational_strategy.inducing_points
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, input], dim=-2)
        full_output = self.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        induc_mean = full_mean[..., :num_induc]
        test_mean = full_mean[..., num_induc:]

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        from ..lazy import CholLazyTensor
        induc_induc_covar = CholLazyTensor(induc_induc_covar.cholesky())
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Prior distribution + samples
        prior_distribution = full_output.__class__(induc_mean, induc_induc_covar)
        inducing_values_samples = self.sample_inducing_values(prior_distribution)

        if inducing_values_samples.dim() == 1:
            means = induc_induc_covar.inv_matmul(
                induc_data_covar,
                inducing_values_samples.unsqueeze(-2)
            ).squeeze(-2)
        elif inducing_values_samples.dim() == 3:
            means = induc_induc_covar.inv_matmul(
                induc_data_covar,
                inducing_values_samples.squeeze(-2)
            )
        else:
            means = induc_induc_covar.inv_matmul(
                induc_data_covar,
                inducing_values_samples
            )
        f_samples = pyro.distributions.Normal(
            means,
            torch.sqrt((
                data_data_covar.diag() - induc_induc_covar.inv_quad(induc_data_covar, reduce_inv_quad=False)
            ).clamp_min(0))
        )

        with pyro.plate(self.name_prefix + ".data_plate", f_samples.batch_shape[-1], dim=-1):
            with pyro.poutine.scale(scale=float(self.num_data / input.size(-2))):
                out_dist = QuadratureDist(self.likelihood, f_samples)
                return pyro.sample(self.name_prefix + ".output_value", out_dist, obs=output)

    def sample_inducing_values(self, inducing_values_dist):
        """
        Sample values from the inducing point distribution `p(u)` or `q(u)`.
        This should only be re-defined to note any conditional independences in
        the `inducing_values_dist` distribution. (By default, all batch dimensions
        are not marked as conditionally indendent.)
        """
        reinterpreted_batch_ndims = len(inducing_values_dist.batch_shape)
        samples = pyro.sample(
            self.name_prefix + ".inducing_values",
            inducing_values_dist.to_event(reinterpreted_batch_ndims)
        )
        return samples

    def transform_function_dist(self, function_dist):
        """
        Transform the function_dist from `gpytorch.distributions.MultivariateNormal` into
        some other variant of a Normal distribution.

        This is useful for marking conditional independencies (useful for inference),
        marking that the distribution contains multiple outputs, etc.

        By default, this funciton transforms a multivariate normal into a set of conditionally
        independent normals when performing inference, and keeps the distribution multivariate
        normal for predictions.
        """
        if self.training:
            return pyro.distributions.Normal(function_dist.mean, function_dist.variance)
        else:
            return function_dist
