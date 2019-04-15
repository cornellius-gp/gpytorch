#!/usr/bin/env python3

from copy import deepcopy

import functools
import torch
import warnings
from ..module import Module
from ..distributions import MultivariateNormal
from ..utils.deprecation import _ClassWithDeprecatedBatchSize
from ..utils.quadrature import GaussHermiteQuadrature1D
from .. import settings


class _Likelihood(Module, _ClassWithDeprecatedBatchSize):
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
        self._register_load_state_dict_pre_hook(self._batch_shape_state_dict_hook)
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
        return log_prob.sum(tuple(range(-1, -len(function_dist.event_shape) - 1, -1)))

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
        sample_shape = torch.Size([settings.num_likelihood_samples.value()])
        function_samples = function_dist.rsample(sample_shape)
        return self.forward(function_samples)

    def variational_log_probability(self, function_dist, observations):
        warnings.warn(
            "Likelihood.variational_log_probability is deprecated. Use Likelihood.expected_log_prob instead.",
            DeprecationWarning
        )
        return self.expected_log_prob(observations, function_dist)

    def get_fantasy_likelihood(self, **kwargs):
        return deepcopy(self)

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
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._max_plate_nesting = 1

        @property
        def max_plate_nesting(self):
            """
            How many batch dimensions are plated (default = 1)
            This should be modified if thew likelihood uses other plated random variables
            """
            return self._max_plate_nesting

        @max_plate_nesting.setter
        def max_plate_nesting(self, val):
            self._max_plate_nesting = val

        def pyro_sample_output(self, observations, function_dist, *params, **kwargs):
            r"""
            Returns observed pyro samples :math:`p(y)` from the likelihood distribution,
            given the function distribution :math:`f`

            .. math::
                \mathbb{E}_{f(x)} \left[ \log p \left( y \mid f(x) \right) \right]

            Args:
                :attr:`observations` (:class:`torch.Tensor`) Values of :math:`y`.
                :attr:`function_dist` (:class:`pyro.distributions`) Distribution for :math:`f(x)`.
                :attr:`params`
                :attr:`kwargs`

            Returns:
                `pyro.sample`
            """
            name_prefix = kwargs.get("name_prefix", "")

            # Get the correct sample shape
            # The default sample shape includes all the batch dimensions that can be smapled from
            sample_shape = kwargs.get("sample_shape", torch.Size([1] * len(function_dist.batch_shape)))
            sample_shape = sample_shape[:-len(function_dist.batch_shape)]
            function_samples = function_dist(sample_shape)
            output_dist = self(function_samples, *params, **kwargs)
            with pyro.plate(name_prefix + ".output_values_plate", function_dist.batch_shape[-1], dim=-1):
                samples = pyro.sample(name_prefix + ".output_values", output_dist, obs=observations)
                return samples

        def guide(self, *param, **kwargs):
            """
            Guide function for the likelihood
            This should be defined if the likelihood contains any random variables that need to be infered.

            If `forward` call to the likelihood function should contains any `pyro.sample` calls, then
            the `guide` call should contain the same sample calls.
            """
            pass

        def marginal(self, function_dist, *params, **kwargs):
            name_prefix = kwargs.get("name_prefix", "")
            num_samples = settings.num_likelihood_samples.value()
            with pyro.plate(name_prefix + ".num_particles_vectorized", num_samples, dim=(-self.max_plate_nesting - 1)):
                function_samples_shape = torch.Size(
                    [num_samples] + [1] * (self.max_plate_nesting - len(function_dist.batch_shape) - 1)
                )
                function_samples = function_dist(function_samples_shape)
                if self.training:
                    return self(function_samples, *params, **kwargs)
                else:
                    guide_trace = pyro.poutine.trace(self.guide).get_trace(*params, **kwargs)
                    marginal_fn = functools.partial(self.__call__, function_samples)
                    return pyro.poutine.replay(marginal_fn, trace=guide_trace)(*params, **kwargs)

        def __call__(self, input, *params, **kwargs):
            # Conditional
            if torch.is_tensor(input):
                return super().__call__(input, *params, **kwargs)
            # Marginal
            elif any([
                isinstance(input, MultivariateNormal),
                isinstance(input, pyro.distributions.Normal),
                (
                    isinstance(input, pyro.distributions.Independent)
                    and isinstance(input.base_dist, pyro.distributions.Normal)
                ),
            ]):
                return self.marginal(input, *params, **kwargs)
            # Error
            else:
                raise RuntimeError(
                    "Likelihoods expects a MultivariateNormal or Normal input to make marginal predictions, or a "
                    "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
                )

except ImportError:
    class Likelihood(_Likelihood):
        def pyro_sample_output(self, *args, **kwargs):
            raise ImportError("Failed to import pyro. Is it installed correctly?")
