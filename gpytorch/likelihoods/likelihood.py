#!/usr/bin/env python3

from copy import deepcopy

import math
import functools
import torch
from ..module import Module
from ..distributions import base_distributions, MultivariateNormal
from ..utils.deprecation import _ClassWithDeprecatedBatchSize
from ..utils.quadrature import GaussHermiteQuadrature1D
from .. import settings
from abc import ABC, abstractmethod


class _Likelihood(Module, ABC, _ClassWithDeprecatedBatchSize):
    def __init__(self, max_plate_nesting=1):
        super().__init__()
        self._register_load_state_dict_pre_hook(self._batch_shape_state_dict_hook)
        self.max_plate_nesting = max_plate_nesting

    def _draw_likelihood_samples(self, function_dist, *args, sample_shape=None, **kwargs):
        if sample_shape is None:
            sample_shape = torch.Size(
                [settings.num_likelihood_samples.value()]
                + [1] * (self.max_plate_nesting - len(function_dist.batch_shape) - 1)
            )
        else:
            sample_shape = sample_shape[:-len(function_dist.batch_shape) - 1]
        if self.training:
            num_event_dims = len(function_dist.event_shape)
            function_dist = base_distributions.Normal(function_dist.mean, function_dist.variance.sqrt())
            function_dist = base_distributions.Independent(function_dist, num_event_dims - 1)
        function_samples = function_dist.rsample(sample_shape)
        return self.forward(function_samples, *args, **kwargs)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        likelihood_samples = self._draw_likelihood_samples(function_dist, *args, **kwargs)
        res = likelihood_samples.log_prob(observations).mean(dim=0)
        return res

    @abstractmethod
    def forward(self, function_samples, *args, **kwargs):
        raise NotImplementedError

    def get_fantasy_likelihood(self, **kwargs):
        return deepcopy(self)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        likelihood_samples = self._draw_likelihood_samples(function_dist, *args, **kwargs)
        log_probs = likelihood_samples.log_prob(observations)
        res = log_probs.sub(math.log(log_probs.size(0))).logsumexp(dim=0)
        return res

    def marginal(self, function_dist, *args, **kwargs):
        res = self._draw_likelihood_samples(function_dist, *args, **kwargs)
        return res

    def __call__(self, input, *args, **kwargs):
        # Conditional
        if torch.is_tensor(input):
            return super().__call__(input, *args, **kwargs)
        # Marginal
        elif isinstance(input, MultivariateNormal):
            return self.marginal(input, *args, **kwargs)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateNormal input to make marginal predictions, or a "
                "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
            )


try:
    import pyro

    class Likelihood(_Likelihood):
        r"""
        A Likelihood in GPyTorch specifies the mapping from latent function values
        :math:`f(\mathbf X)` to observed labels :math:`y`.

        For example, in the case of regression this might be a Gaussian
        distribution, as :math:`y(\mathbf x)` is equal to :math:`f(\mathbf x)` plus Gaussian noise:

        .. math::
            y(\mathbf x) = f(\mathbf x) + \epsilon, \:\:\:\: \epsilon ~ N(0,\sigma^{2}_{n} \mathbf I)

        In the case of classification, this might be a Bernoulli distribution,
        where the probability that :math:`y=1` is given by the latent function
        passed through some sigmoid or probit function:

        .. math::
            y(\mathbf x) = \begin{cases}
                1 & \text{w/ probability} \:\: \sigma(f(\mathbf x)) \\
                0 & \text{w/ probability} \:\: 1-\sigma(f(\mathbf x))
            \end{cases}

        In either case, to implement a likelihood function, GPyTorch only
        requires a :attr:`forward` method that computes the conditional distribution
        :math:`p(y \mid f(\mathbf x))`.

        Calling this object does one of two things:

            - If likelihood is called with a :class:`torch.Tensor` object, then it is
              assumed that the input is samples from :math:`f(\mathbf x)`. This
              returns the *conditional* distribution `p(y|f(\mathbf x))`.
            - If likelihood is called with a :class:`~gpytorch.distribution.MultivariateNormal` object,
              then it is assumed that the input is the distribution :math:`f(\mathbf x)`.
              This returns the *marginal* distribution `p(y|\mathbf x)`.

        Args:
            :attr:`max_plate_nesting` (int, default=1)
                (For Pyro integration only). How many batch dimensions are in the function.
                This should be modified if thew likelihood uses plated random variables.
        """
        def _draw_likelihood_samples(self, function_dist, *args, **kwargs):
            # Hack to get the current plating structure
            sample_shape = pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1)).shape
            if not len(sample_shape):
                sample_shape = None
            return super()._draw_likelihood_samples(function_dist, *args, **kwargs, sample_shape=sample_shape)

        def expected_log_prob(self, observations, function_dist, *args, **kwargs):
            r"""
            (Used by :obj:`~gpytorch.mlls.VariationalELBO` for variational inference.)

            Computes the expected log likelihood, where the expectation is over the GP variational distribution.

            .. math::
                \sum_{\mathbf x, y} \mathbb{E}_{q\left( f(\mathbf x) \right)}
                \left[ \log p \left( y \mid f(\mathbf x) \right) \right]

            Args:
                :attr:`observations` (:class:`torch.Tensor`)
                    Values of :math:`y`.
                :attr:`function_dist` (:class:`~gpytorch.distributions.MultivariateNormal`)
                    Distribution for :math:`f(x)`.
                :attr:`args`, :attr:`kwargs`
                    Passed to the `forward` function

            Returns
                `torch.Tensor` (log probability)
            """
            return super().expected_log_prob(observations, function_dist, *args, **kwargs)

        @abstractmethod
        def forward(self, function_samples, *args, **kwargs):
            """
            Computes the conditional distribution p(y|f) that defines the likelihood.

            Args:
                :attr:`function_samples` (:obj:`torch.Tensor`)
                    Samples from the function `f`
                :attr:`args`, :attr:`kwargs`
                    Any additional arguments and keyword arguments

            Returns:
                Distribution object (with same shape as :attr:`function_samples`)
            """
            raise NotImplementedError

        def get_fantasy_likelihood(self, **kwargs):
            """
            """
            return super().get_fantasy_likelihood(**kwargs)

        def guide(self, *args, **kwargs):
            """
            (For Pyro integration only).

            Guide function for the likelihood
            This should be defined if the likelihood contains any random variables that need to be infered.
            In other words, if `forward` call to the likelihood function should contains any `pyro.sample` calls,
            then the `guide` call should contain the same sample calls.

            Args:
                :attr:`args`, :attr:`kwargs`
                    Passed to the `forward` function
            """
            pass

        def log_marginal(self, observations, function_dist, *args, **kwargs):
            r"""
            (Used by :obj:`~gpytorch.mlls.PredictiveLogLikelihood` for approximate inference.)

            Computes the log marginal likelihood of the approximate predictive distribution

            .. math::
                \sum_{\mathbf x, y} \log \mathbb{E}_{q\left( f(\mathbf x) \right)}
                \left[ p \left( y \mid f(\mathbf x) \right) \right]

            Note that this differs from :meth:`expected_log_prob` because the :math:`log` is on the outside
            of the expectation.

            Args:
                :attr:`observations` (:class:`torch.Tensor`)
                    Values of :math:`y`.
                :attr:`function_dist` (:class:`~gpytorch.distributions.MultivariateNormal`)
                    Distribution for :math:`f(x)`.
                :attr:`args`, :attr:`kwargs`
                    Passed to the `forward` function

            Returns
                `torch.Tensor` (log probability)
            """
            return super().log_marginal(observations, function_dist, *args, **kwargs)

        def marginal(self, function_dist, *args, **kwargs):
            r"""
            Computes a predictive distribution :math:`p(y^* | \mathbf x^*)` given either a posterior
            distribution :math:`p(\mathbf f | \mathcal D, \mathbf x)` or a
            prior distribution :math:`p(\mathbf f|\mathbf x)` as input.

            With both exact inference and variational inference, the form of
            :math:`p(\mathbf f|\mathcal D, \mathbf x)` or :math:`p(\mathbf f|
            \mathbf x)` should usually be Gaussian. As a result, :attr:`function_dist`
            should usually be a :obj:`~gpytorch.distributions.MultivariateNormal` specified by the mean and
            (co)variance of :math:`p(\mathbf f|...)`.

            Args:
                :attr:`function_dist` (:class:`~gpytorch.distributions.MultivariateNormal`)
                    Distribution for :math:`f(x)`.
                :attr:`args`, :attr:`kwargs`
                    Passed to the `forward` function

            Returns:
                Distribution object (the marginal distribution, or samples from it)
            """
            name_prefix = kwargs.get("name_prefix", "")
            plate_name = name_prefix + ".num_particles_vectorized"
            num_samples = settings.num_likelihood_samples.value()
            with pyro.plate(plate_name, size=num_samples, dim=(-self.max_plate_nesting - 1)):
                guide_trace = pyro.poutine.trace(self.guide).get_trace(*args, **kwargs)
                marginal_fn = functools.partial(super().marginal, function_dist)
                res = pyro.poutine.replay(marginal_fn, trace=guide_trace)(*args, **kwargs)
                return res

        def pyro_sample_output(self, observations, function_dist, scale, *args, **kwargs):
            r"""
            Returns observed pyro samples :math:`p(y)` from the likelihood distribution,
            given the function distribution :math:`f`.

            .. math::
                \mathbb{E}_{f(x)} \left[ \log p \left( y \mid f(x) \right) \right]

            Args:
                :attr:`observations` (:class:`torch.Tensor`):
                    Values of :math:`y`.
                :attr:`function_dist` (:class:`pyro.distributions`):
                    Distribution for :math:`f(x)`.
                :attr:`scale` (float):
                    Scale factor to multiply likelihood probabilities by to properly scale stochastic inference.
                    Should be equal to `total_num_data`/`num_minibatch`.
                :attr:`args`, :attr:`kwargs`
                    Passed to the `forward` function

            Returns:
                `pyro.sample`
            """
            name_prefix = kwargs.get("name_prefix", "")

            # Hack to the correct sample shape
            # The default sample shape includes all the batch dimensions that can be smapled from
            sample_shape = pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1)).shape

            # Make sure that the function dist is factored to be independent
            function_dist = pyro.distributions.Normal(
                loc=function_dist.mean,
                scale=function_dist.variance.sqrt()
            ).to_event(len(function_dist.event_shape) - 1)

            # Draw samples from the likelihood dist, which comes from samples of the function dist
            function_samples = function_dist(sample_shape[:-len(function_dist.batch_shape)])
            output_dist = self(function_samples, *args, **kwargs)

            # Condition on these samples
            with pyro.plate(name_prefix + ".output_values_plate", output_dist.batch_shape[-1], dim=-1):
                with pyro.poutine.scale(scale=scale):
                    samples = pyro.sample(name_prefix + ".output_values", output_dist, obs=observations)
                    return samples

        def __call__(self, input, *args, **kwargs):
            # Conditional
            if torch.is_tensor(input):
                return super().__call__(input, *args, **kwargs)
            # Marginal
            elif any([
                isinstance(input, MultivariateNormal),
                isinstance(input, pyro.distributions.Normal),
                (
                    isinstance(input, pyro.distributions.Independent)
                    and isinstance(input.base_dist, pyro.distributions.Normal)
                ),
            ]):
                return self.marginal(input, *args, **kwargs)
            # Error
            else:
                raise RuntimeError(
                    "Likelihoods expects a MultivariateNormal or Normal input to make marginal predictions, or a "
                    "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
                )

except ImportError:
    class Likelihood(_Likelihood):
        pass


class _OneDimensionalLikelihood(Likelihood, ABC):
    r"""
    A specific case of :obj:`~gpytorch.likelihoods.Likelihood` when the GP represents a one-dimensional
    output. (I.e. for a specific :math:`\mathbf x`, :math:`f(\mathbf x) \in \mathbb{R}`.)

    Inheriting from this likelihood reduces the variance when computing approximate GP objective functions
    by using 1D Gauss-Hermite quadrature.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quadrature = GaussHermiteQuadrature1D()

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(observations).exp()
        prob = self.quadrature(prob_lambda, function_dist)
        return prob.log()
