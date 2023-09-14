#!/usr/bin/env python3
import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import torch
from linear_operator.operators import LinearOperator, MaskedLinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.distributions import Distribution, Normal

from .. import settings
from ..constraints import Interval
from ..distributions import base_distributions, MultivariateNormal
from ..priors import Prior
from ..utils.warnings import GPInputWarning
from .likelihood import Likelihood
from .noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise


class _GaussianLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    has_analytic_marginal = True

    def __init__(self, noise_covar: Union[Noise, FixedGaussianNoise], **kwargs: Any) -> None:
        super().__init__()
        param_transform = kwargs.get("param_transform")
        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformaton, specify a different 'noise_constraint' instead.",
                DeprecationWarning,
            )

        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:

        noise = self._shaped_noise_covar(input.mean.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        # Handle NaN values if enabled
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "mask":
            observed = settings.observation_nan_policy._get_observed(target, input.event_shape)
            input = MultivariateNormal(
                mean=input.mean[..., observed],
                covariance_matrix=MaskedLinearOperator(
                    input.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
                ),
            )
            noise = noise[..., observed]
            target = target[..., observed]
        elif nan_policy == "fill":
            missing = torch.isnan(target)
            target = settings.observation_nan_policy._fill_tensor(target)

        mean, variance = input.mean, input.variance
        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        if nan_policy == "fill":
            res = res * ~missing

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(input.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))

        return res

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        return base_distributions.Normal(function_samples, noise.sqrt())

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        marginal = self.marginal(function_dist, *params, **kwargs)

        # Handle NaN values if enabled
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "mask":
            observed = settings.observation_nan_policy._get_observed(observations, marginal.event_shape)
            marginal = MultivariateNormal(
                mean=marginal.mean[..., observed],
                covariance_matrix=MaskedLinearOperator(
                    marginal.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
                ),
            )
            observations = observations[..., observed]
        elif nan_policy == "fill":
            missing = torch.isnan(observations)
            observations = settings.observation_nan_policy._fill_tensor(observations)

        # We're making everything conditionally independent
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        res = indep_dist.log_prob(observations)

        if nan_policy == "fill":
            res = res * ~missing

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(marginal.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


class GaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    The standard likelihood for regression.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. note::
        GaussianLikelihood has an analytic marginal distribution.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param kwargs:

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(
        self,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)


class GaussianLikelihoodWithMissingObs(GaussianLikelihood):
    r"""
    The standard likelihood for regression with support for missing values.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter. Values of y that are nan do
    not impact the likelihood calculation.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. warning::
        This likelihood is deprecated in favor of :class:`gpytorch.settings.observation_nan_policy`.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :type noise_prior: ~gpytorch.priors.Prior, optional
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :type noise_constraint: ~gpytorch.constraints.Interval, optional
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :type batch_shape: torch.Size, optional
    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        GaussianLikelihoodWithMissingObs has an analytic marginal distribution.
    """

    MISSING_VALUE_FILL: float = -999.0

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn(
            "GaussianLikelihoodWithMissingObs is replaced by gpytorch.settings.observation_nan_policy('fill').",
            DeprecationWarning,
        )
        super().__init__(**kwargs)

    def _get_masked_obs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        missing_idx = x.isnan()
        x_masked = x.masked_fill(missing_idx, self.MISSING_VALUE_FILL)
        return missing_idx, x_masked

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        missing_idx, target = self._get_masked_obs(target)
        res = super().expected_log_prob(target, input, *params, **kwargs)
        return res * ~missing_idx

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        missing_idx, observations = self._get_masked_obs(observations)
        res = super().log_marginal(observations, function_dist, *params, **kwargs)
        return res * ~missing_idx

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)


class FixedNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    A Likelihood that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateNormal. This allows for adding known observational noise to test data.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param noise: Known observation noise (variance) for each training example.
    :type noise: torch.Tensor (... x N)
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        FixedNoiseGaussianLikelihood has an analytic marginal distribution.

    Example:
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(gp_model(test_x), noise=test_noises)
    """

    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

        self.second_noise_covar: Optional[HomoskedasticNoise] = None
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
            )

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Union[float, Tensor]:
        if self.second_noise_covar is None:
            return 0.0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs: Any) -> "FixedNoiseGaussianLikelihood":
        if "noise" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None  # pyre-fixme[8]
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)


class DirichletClassificationLikelihood(FixedNoiseGaussianLikelihood):
    r"""
    A classification likelihood that treats the labels as regression targets with fixed heteroscedastic noise.
    From Milios et al, NeurIPS, 2018 [https://arxiv.org/abs/1805.10915].

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param targets: (... x N) Classification labels.
    :param alpha_epsilon: Tuning parameter for the scaling of the likeihood targets. We'd suggest 0.01 or setting
        via cross-validation.
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        DirichletClassificationLikelihood has an analytic marginal distribution.

    Example:
        >>> train_x = torch.randn(55, 1)
        >>> labels = torch.round(train_x).long()
        >>> likelihood = DirichletClassificationLikelihood(targets=labels, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 1)
        >>> test_labels = torch.round(test_x).long()
        >>> pred_y = likelihood(gp_model(test_x), targets=labels)
    """

    def _prepare_targets(
        self, targets: Tensor, alpha_epsilon: float = 0.01, dtype: torch.dtype = torch.float
    ) -> Tuple[Tensor, Tensor, int]:
        num_classes = int(targets.max() + 1)
        # set alpha = \alpha_\epsilon
        alpha = alpha_epsilon * torch.ones(targets.shape[-1], num_classes, device=targets.device, dtype=dtype)

        # alpha[class_labels] = 1 + \alpha_\epsilon
        alpha[torch.arange(len(targets)), targets] = alpha[torch.arange(len(targets)), targets] + 1.0

        # sigma^2 = log(1 / alpha + 1)
        sigma2_i = torch.log(alpha.reciprocal() + 1.0)

        # y = log(alpha) - 0.5 * sigma^2
        transformed_targets = alpha.log() - 0.5 * sigma2_i

        return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.type(dtype), num_classes

    def __init__(
        self,
        targets: Tensor,
        alpha_epsilon: float = 0.01,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: torch.Size = torch.Size(),
        dtype: torch.dtype = torch.float,
        **kwargs: Any,
    ) -> None:
        sigma2_labels, transformed_targets, num_classes = self._prepare_targets(
            targets, alpha_epsilon=alpha_epsilon, dtype=dtype
        )
        super().__init__(
            noise=sigma2_labels,
            learn_additional_noise=learn_additional_noise,
            batch_shape=torch.Size((num_classes,)),
            **kwargs,
        )
        self.transformed_targets: Tensor = transformed_targets.transpose(-2, -1)
        self.num_classes: int = num_classes
        self.targets: Tensor = targets
        self.alpha_epsilon: float = alpha_epsilon

    def get_fantasy_likelihood(self, **kwargs: Any) -> "DirichletClassificationLikelihood":
        # we assume that the number of classes does not change.

        if "targets" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `targets` kwarg")

        old_noise_covar = self.noise_covar
        self.noise_covar = None  # pyre-fixme[8]
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_targets = kwargs.get("noise")
        new_noise, new_targets, _ = fantasy_liklihood._prepare_targets(new_targets, self.alpha_epsilon)
        fantasy_liklihood.targets = torch.cat([fantasy_liklihood.targets, new_targets], -1)

        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])

        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)

    def __call__(self, input: Union[Tensor, MultivariateNormal], *args: Any, **kwargs: Any) -> Distribution:
        if "targets" in kwargs:
            targets = kwargs.pop("targets")
            dtype = self.transformed_targets.dtype
            new_noise, _, _ = self._prepare_targets(targets, dtype=dtype)
            kwargs["noise"] = new_noise
        return super().__call__(input, *args, **kwargs)
