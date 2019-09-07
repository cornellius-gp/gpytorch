#!/usr/bin/env python3

import torch
import pyro
from .. import Module
from ..lazy import RootLazyTensor, DiagLazyTensor, BlockDiagLazyTensor
from ..distributions import MultivariateNormal, MultitaskMultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from . import GP
from .. import settings


class SteinVariationalGP(Module):
    def __init__(self, inducing_points, likelihood, num_data, name_prefix=""):
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        super().__init__()
        self.likelihood = likelihood
        self.num_data = num_data
        self.name_prefix = name_prefix

        # Cheap buffers
        self.register_parameter("inducing_points", torch.nn.Parameter(inducing_points))
        self.register_buffer("prior_mean", torch.zeros(inducing_points.shape[:-1]))
        self.register_buffer("prior_var", torch.ones(inducing_points.shape[:-1]))

    def model(self, input, output, *params, **kwargs):
        predict = kwargs.pop('predict', False)
        pyro.module(self.name_prefix + ".gp_prior", self)

        function_dist = self(input, *params, **kwargs)

        # Go from function -> output
        num_minibatch = function_dist.event_shape[0]
        scale_factor = float(self.num_data / num_minibatch)

        if predict:
            print("in predict branch")
            assert False, "actually we are never in this branch..."
            likelihood_dist = pyro.distributions.Normal(function_dist.mean,
                (function_dist.variance + self.likelihood.noise).sqrt()).to_event(1)
            with pyro.poutine.scale(scale=scale_factor):
                return pyro.sample(self.name_prefix + ".output_values", likelihood_dist, obs=output)
        else:
            obs_dist = torch.distributions.Normal(function_dist.mean, self.likelihood.noise.sqrt())
            factor1 = obs_dist.log_prob(output).sum(-1)
            factor2 = 0.5 * function_dist.variance.sum(1) / self.likelihood.noise
            factor = scale_factor * (factor1 - factor2)
            pyro.factor(self.name_prefix + ".output_values", factor)

    def sample_inducing_values(self):
        """
        Sample values from the inducing point distribution `p(u)` or `q(u)`.
        This should only be re-defined to note any conditional independences in
        the `inducing_values_dist` distribution. (By default, all batch dimensions
        are not marked as conditionally indendent.)
        """
        prior_dist = MultivariateNormal(self.prior_mean, DiagLazyTensor(self.prior_var))
        samples = pyro.sample(self.name_prefix + ".inducing_values", prior_dist)
        return samples

    def __call__(self, input, *args, **kwargs):
        inducing_points = self.inducing_points
        inducing_batch_shape = inducing_points.shape[:-2]
        if inducing_batch_shape < input.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], input.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            input = input.expand(*batch_shape, *input.shape[-2:])
        # Draw samples from p(u) for KL divergence computation
        inducing_values_samples = self.sample_inducing_values()

        # Get function dist
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, input], dim=-2)
        full_output = self.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        test_mean = full_output.mean[..., num_induc:]
        L = full_covar[..., :num_induc, :num_induc].add_jitter().cholesky().evaluate()
        cross_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        scaled_cross_covar = torch.triangular_solve(cross_covar, L, upper=False)[0]
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        function_dist = MultivariateNormal(
            (scaled_cross_covar.transpose(-1, -2) @ inducing_values_samples.unsqueeze(-1)).squeeze(-1),
            data_data_covar + RootLazyTensor(scaled_cross_covar.transpose(-1, -2)).mul(-1)
        )
        return function_dist

class SteinVariationalDeepGPLayer(SteinVariationalGP):
    def __init__(self, input_dims, output_dims, inducing_points, likelihood, num_data, name_prefix=""):
        if inducing_points.size(-1) != input_dims:
            raise RuntimeError(
                "Inducing point dimensionality must match specified input_dim."
            )

        if output_dims is not None and inducing_points.size(-3) != output_dims:
            raise RuntimeError(
                "Inducing point batch size must match specified output_dims."
            )
        super().__init__(inducing_points, likelihood, num_data, name_prefix)
        self.input_dims = input_dims
        self.output_dims = output_dims

    def __call__(self, inputs, are_samples=False, *args, **kwargs):
        deterministic_input = not are_samples
        if isinstance(inputs, MultitaskMultivariateNormal):
            inputs = torch.distributions.Normal(inputs.mean, inputs.variance.sqrt()).rsample()
            deterministic_input = False

        if settings.debug.on():
            if not torch.is_tensor(inputs):
                raise ValueError(
                    "`inputs` should either be a MultitaskMultivariateNormal or a Tensor, got "
                    f"{inputs.__class__.__Name__}"
                )

            if inputs.size(-1) != self.input_dims:
                raise RuntimeError(
                    f"Input shape did not match self.input_dims. Got total feature dims [{inputs.size(-1)}],"
                    f" expected [{self.input_dims}]"
                )

        if self.output_dims is not None:
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])

        output = super().__call__(inputs)

        if self.output_dims is not None:
            mean = output.loc.transpose(-2, -1)
            covar = BlockDiagLazyTensor(output.lazy_covariance_matrix, block_dim=-3)
            output = MultitaskMultivariateNormal(mean, covar, interleaved=False)

        if deterministic_input:
            mean = mean.expand(torch.Size([settings.num_likelihood_samples.value()]) + mean.shape)
            if len(output.batch_shape) > 0:
                mean = mean.transpose(0, 1)
            output = MultitaskMultivariateNormal(mean, covar, interleaved=False)

        return output

class SteinVariationalDeepGP(GP):
    def __init__(self, num_data, likelihood, name_prefix=""):
        super().__init__()
        self.name_prefix = name_prefix
        self.num_data = num_data
        self.likelihood = likelihood

    def forward(self, input):
        """
        Assume forward passes through all hidden GP layers.
        """
        raise NotImplementedError

    def model(self, input, output, *params, **kwargs):
        pyro.module(self.name_prefix + ".gp_prior", self)

        # Function dist will be e.g. gp2(gp1(x)), and be f_samples x minibatch_size MVN
        # pyro sample calls on p(u) will have happened in all layers already.
        function_dist = self(input, *params, **kwargs)

        # Go from function -> output
        num_minibatch = function_dist.event_shape[0]
        num_samples = settings.num_likelihood_samples.value()
        scale_factor = self.num_data / (num_samples * num_minibatch)
        with pyro.poutine.scale(scale=float(scale_factor)):
            likelihood_dist = pyro.distributions.Normal(
                function_dist.mean,
                (function_dist.variance + self.likelihood.noise).sqrt()
            ).to_event(1)
            return pyro.sample(self.name_prefix + ".output_values", likelihood_dist, obs=output)
