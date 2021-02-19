#!/usr/bin/env python3

import math
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import torch

from .. import settings
from ..distributions import MultivariateNormal, base_distributions
from ..module import Module
from ..utils.quadrature import GaussHermiteQuadrature1D
from ..utils.warnings import GPInputWarning


class _Likelihood(Module, ABC):
    def __init__(self, max_plate_nesting=1):
        super().__init__()
        self.max_plate_nesting = max_plate_nesting

    def _draw_likelihood_samples(self, function_dist, *args, sample_shape=None, **kwargs):
        if sample_shape is None:
            sample_shape = torch.Size(
                [settings.num_likelihood_samples.value()]
                + [1] * (self.max_plate_nesting - len(function_dist.batch_shape) - 1)
            )
        else:
            sample_shape = sample_shape[: -len(function_dist.batch_shape) - 1]
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
              returns the *conditional* distribution :math:`p(y|f(\mathbf x))`.
            - If likelihood is called with a :class:`~gpytorch.distribution.MultivariateNormal` object,
              then it is assumed that the input is the distribution :math:`f(\mathbf x)`.
              This returns the *marginal* distribution :math:`p(y|\mathbf x)`.

        :param max_plate_nesting: (For Pyro integration only). How many batch dimensions are in the function.
            This should be modified if the likelihood uses plated random variables.
        :type max_plate_nesting: int, default=1
        """

        @property
        def num_data(self):
            if hasattr(self, "_num_data"):
                return self._num_data
            else:
                warnings.warn(
                    "likelihood.num_data isn't set. This might result in incorrect ELBO scaling.", GPInputWarning
                )
                return ""

        @num_data.setter
        def num_data(self, val):
            self._num_data = val

        @property
        def name_prefix(self):
            if hasattr(self, "_name_prefix"):
                return self._name_prefix
            else:
                return ""

        @name_prefix.setter
        def name_prefix(self, val):
            self._name_prefix = val

        def _draw_likelihood_samples(self, function_dist, *args, sample_shape=None, **kwargs):
            if self.training:
                num_event_dims = len(function_dist.event_shape)
                function_dist = base_distributions.Normal(function_dist.mean, function_dist.variance.sqrt())
                function_dist = base_distributions.Independent(function_dist, num_event_dims - 1)

            plate_name = self.name_prefix + ".num_particles_vectorized"
            num_samples = settings.num_likelihood_samples.value()
            max_plate_nesting = max(self.max_plate_nesting, len(function_dist.batch_shape))
            with pyro.plate(plate_name, size=num_samples, dim=(-max_plate_nesting - 1)):
                if sample_shape is None:
                    function_samples = pyro.sample(self.name_prefix, function_dist.mask(False))
                    # Deal with the fact that we're not assuming conditional indendence over data points here
                    function_samples = function_samples.squeeze(-len(function_dist.event_shape) - 1)
                else:
                    sample_shape = sample_shape[: -len(function_dist.batch_shape)]
                    function_samples = function_dist(sample_shape)

                if not self.training:
                    function_samples = function_samples.squeeze(-len(function_dist.event_shape) - 1)
                return self.forward(function_samples, *args, **kwargs)

        def expected_log_prob(self, observations, function_dist, *args, **kwargs):
            r"""
            (Used by :obj:`~gpytorch.mlls.VariationalELBO` for variational inference.)

            Computes the expected log likelihood, where the expectation is over the GP variational distribution.

            .. math::
                \sum_{\mathbf x, y} \mathbb{E}_{q\left( f(\mathbf x) \right)}
                \left[ \log p \left( y \mid f(\mathbf x) \right) \right]

            :param torch.Tensor observations: Values of :math:`y`.
            :param ~gpytorch.distributions.MultivariateNormal function_dist: Distribution for :math:`f(x)`.
            :param args: Additional args (passed to the foward function).
            :param kwargs: Additional kwargs (passed to the foward function).
            :rtype: torch.Tensor
            """
            return super().expected_log_prob(observations, function_dist, *args, **kwargs)

        @abstractmethod
        def forward(self, function_samples, *args, data={}, **kwargs):
            r"""
            Computes the conditional distribution :math:`p(\mathbf y \mid
            \mathbf f, \ldots)` that defines the likelihood.

            :param torch.Tensor function_samples: Samples from the function (:math:`\mathbf f`)
            :param data: Additional variables that the likelihood needs to condition
                on. The keys of the dictionary will correspond to Pyro sample sites
                in the likelihood's model/guide.
            :type data: dict {str: torch.Tensor}, optional - Pyro integration only
            :param args: Additional args
            :param kwargs: Additional kwargs
            :rtype: :obj:`Distribution` (with same shape as :attr:`function_samples` )
            """
            raise NotImplementedError

        def get_fantasy_likelihood(self, **kwargs):
            """"""
            return super().get_fantasy_likelihood(**kwargs)

        def log_marginal(self, observations, function_dist, *args, **kwargs):
            r"""
            (Used by :obj:`~gpytorch.mlls.PredictiveLogLikelihood` for approximate inference.)

            Computes the log marginal likelihood of the approximate predictive distribution

            .. math::
                \sum_{\mathbf x, y} \log \mathbb{E}_{q\left( f(\mathbf x) \right)}
                \left[ p \left( y \mid f(\mathbf x) \right) \right]

            Note that this differs from :meth:`expected_log_prob` because the :math:`log` is on the outside
            of the expectation.

            :param torch.Tensor observations: Values of :math:`y`.
            :param ~gpytorch.distributions.MultivariateNormal function_dist: Distribution for :math:`f(x)`.
            :param args: Additional args (passed to the foward function).
            :param kwargs: Additional kwargs (passed to the foward function).
            :rtype: torch.Tensor
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

            :param ~gpytorch.distributions.MultivariateNormal function_dist: Distribution for :math:`f(x)`.
            :param args: Additional args (passed to the foward function).
            :param kwargs: Additional kwargs (passed to the foward function).
            :return: The marginal distribution, or samples from it.
            :rtype: ~gpytorch.distributions.Distribution
            """
            return super().marginal(function_dist, *args, **kwargs)

        def pyro_guide(self, function_dist, target, *args, **kwargs):
            r"""
            (For Pyro integration only).

            Part of the guide function for the likelihood.
            This should be re-defined if the likelihood contains any latent variables that need to be infered.

            :param ~gpytorch.distributions.MultivariateNormal function_dist: Distribution of latent function
                :math:`q(\mathbf f)`.
            :param torch.Tensor target: Observed :math:`\mathbf y`.
            :param args: Additional args (for :meth:`~forward`).
            :param kwargs: Additional kwargs (for :meth:`~forward`).
            """
            with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
                pyro.sample(self.name_prefix + ".f", function_dist)

        def pyro_model(self, function_dist, target, *args, **kwargs):
            r"""
            (For Pyro integration only).

            Part of the model function for the likelihood.
            It should return the
            This should be re-defined if the likelihood contains any latent variables that need to be infered.

            :param ~gpytorch.distributions.MultivariateNormal function_dist: Distribution of latent function
                :math:`p(\mathbf f)`.
            :param torch.Tensor target: Observed :math:`\mathbf y`.
            :param args: Additional args (for :meth:`~forward`).
            :param kwargs: Additional kwargs (for :meth:`~forward`).
            """
            with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
                function_samples = pyro.sample(self.name_prefix + ".f", function_dist)
                output_dist = self(function_samples, *args, **kwargs)
                return self.sample_target(output_dist, target)

        def sample_target(self, output_dist, target):
            scale = (self.num_data or output_dist.batch_shape[-1]) / output_dist.batch_shape[-1]
            with pyro.poutine.scale(scale=scale):
                return pyro.sample(self.name_prefix + ".y", output_dist, obs=target)

        def __call__(self, input, *args, **kwargs):
            # Conditional
            if torch.is_tensor(input):
                return super().__call__(input, *args, **kwargs)
            # Marginal
            elif any(
                [
                    isinstance(input, MultivariateNormal),
                    isinstance(input, pyro.distributions.Normal),
                    (
                        isinstance(input, pyro.distributions.Independent)
                        and isinstance(input.base_dist, pyro.distributions.Normal)
                    ),
                ]
            ):
                return self.marginal(input, *args, **kwargs)
            # Error
            else:
                raise RuntimeError(
                    "Likelihoods expects a MultivariateNormal or Normal input to make marginal predictions, or a "
                    "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
                )


except ImportError:

    class Likelihood(_Likelihood):
        @property
        def num_data(self):
            warnings.warn("num_data is only used for likehoods that are integrated with Pyro.", RuntimeWarning)
            return 0

        @num_data.setter
        def num_data(self, val):
            warnings.warn("num_data is only used for likehoods that are integrated with Pyro.", RuntimeWarning)

        @property
        def name_prefix(self):
            warnings.warn("name_prefix is only used for likehoods that are integrated with Pyro.", RuntimeWarning)
            return ""

        @name_prefix.setter
        def name_prefix(self, val):
            warnings.warn("name_prefix is only used for likehoods that are integrated with Pyro.", RuntimeWarning)


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
