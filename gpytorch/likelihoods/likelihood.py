#!/usr/bin/env python3

import torch
from ..module import Module
from ..distributions import MultivariateNormal
from ..utils.quadrature import GaussHermiteQuadrature1D
from .. import settings
import warnings


class _Likelihood(Module):
    """
    A Likelihood in GPyTorch specifies the mapping from latent function values
    f to observed labels y.

    For example, in the case of regression this might be a Gaussian
    distribution, as y(x) is equal to f(x) plus Gaussian noise:

    .. math::
        y(x) = f(x) + \epsilon, \epsilon ~ N(0,\sigma^{2}_{n} I)

    In the case of classification, this might be a Bernoulli distribution,
    where the probability that y=1 is given by the latent function
    passed through some sigmoid or probit function:

        y(x) = 1 w/ probability \sigma(f(x)), -1 w/ probability 1-\sigma(f(x))

    In either case, to implement a (non-Gaussian) likelihood function, GPyTorch
    requires a :attr:`forward` method that computes the conditional distribution
    :math:`p(y \mid f)`.

    Calling this object does one of two things:

        - If likelihood is called with a :class:`torch.Tensor` object, then it is
          assumed that the input is samples from :math:`f(x)`. This
          returns the *conditional* distribution `p(y|f(x))`.
        - If likelihood is called with a :class:`gpytorch.distribution.MultivariateNormal` object,
          then it is assumed that the input is the distribution :math:`f(x)`.
          This returns the *marginal* distribution `p(y|x)`.
    """
    def __init__(self):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, function_samples, *params, **kwargs):
        """
        Computes the conditional distribution p(y|f) that defines the likelihood.

        Args:
            :attr:`function_samples`
                Samples from the function `f`
            :attr:`kwargs`

        Returns:
            Distribution object (with same shape as :attr:`function_samples`)
        """
        raise NotImplementedError

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        """
        Computes the expected log likelihood (used for variational inference):

        .. math::
            \mathbb{E}_{f(x)} \left[ \log p \left( y \mid f(x) \right) \right]

        Args:
            :attr:`function_dist` (:class:`gpytorch.distributions.MultivariateNormal`)
                Distribution for :math:`f(x)`.
            :attr:`observations` (:class:`torch.Tensor`)
                Values of :math:`y`.
            :attr:`kwargs`

        Returns
            `torch.Tensor` (log probability)
        """
        log_prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob.sum(tuple(range(len(function_dist.event_shape))))

    def marginal(self, function_dist, *params, **kwargs):
        """
        Computes a predictive distribution :math:`p(y*|x*)` given either a posterior
        distribution :math:`p(f|D,x)` or a prior distribution :math:`p(f|x)` as input.

        With both exact inference and variational inference, the form of
        :math:`p(f|D,x)` or :math:`p(f|x)` should usually be Gaussian. As a result, input
        should usually be a MultivariateNormal specified by the mean and
        (co)variance of :math:`p(f|...)`.

        Args:
            :attr:`function_dist` (:class:`gpytorch.distributions.MultivariateNormal`)
                Distribution for :math:`f(x)`.
            :attr:`kwargs`

        Returns
            Distribution object (the marginal distribution, or samples from it)
        """
        sample_shape = torch.Size((settings.num_likelihood_samples,))
        function_samples = function_dist.rsample(sample_shape)
        return self.forward(function_samples)

    def variational_log_probability(self, function_dist, observations):
        warnings.warn(
            "Likelihood.variational_log_probability is deprecated. Use Likelihood.expected_log_prob instead.",
            DeprecationWarning
        )
        return self.expected_log_prob(observations, function_dist)

    def __call__(self, input, *params, **kwargs):
        # Conditional
        if torch.is_tensor(input):
            return super().__call__(input, *params, **kwargs)
        # Marginal
        elif isinstance(input, MultivariateNormal):
            return self.marginal(input, *params, **kwargs)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateNormal input to make marginal predictions, or a "
                "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
            )


try:
    import pyro

    class Likelihood(_Likelihood):
        def pyro_sample_outputs(self, observations, function_dist, *params, **kwargs):
            name_prefix = kwargs.pop("name_prefix", "")
            with pyro.plate(name_prefix + ".output_values_plate", function_dist.batch_shape[-1], dim=-1):
                with pyro.poutine.block():
                    function_samples = pyro.sample(name_prefix + ".function_values", function_dist)
                output_dist = self(function_samples, *params, **kwargs)
                samples = pyro.sample(name_prefix + ".output_values", output_dist, obs=observations)
                return samples

except ImportError:
    class Likelihood(_Likelihood):
        def pyro_sample_outputs(self, *args, **kwargs):
            raise ImportError("Failed to import pyro. Is it installed correctly?")
