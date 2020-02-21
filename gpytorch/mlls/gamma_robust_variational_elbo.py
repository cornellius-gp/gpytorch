#!/usr/bin/env python3

import math

import numpy as np
import torch

from ..likelihoods import _GaussianLikelihoodBase
from ._approximate_mll import _ApproximateMarginalLogLikelihood


class GammaRobustVariationalELBO(_ApproximateMarginalLogLikelihood):
    r"""
    An alternative to the variational evidence lower bound (ELBO), proposed by `Knoblauch, 2019`_.
    It is derived by replacing the log-likelihood term in the ELBO with a `\gamma` divergence:

    .. math::

       \begin{align*}
          \mathcal{L}_{\gamma} &=
          \sum_{i=1}^N \mathbb{E}_{q( \mathbf u)} \left[
            -\frac{\gamma}{\gamma - 1}
            \frac{
                p( y_i \! \mid \! \mathbf u, x_i)^{\gamma - 1}
            }{
                \int p(y \mid \mathbf u, x_i)^{\gamma} \: dy
            }
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the number of datapoints, :math:`\gamma` is a hyperparameter,
    :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, and :math:`p(\mathbf u)` is the prior distribution for the inducing function
    values.

    :math:`\beta` is a scaling constant for the KL divergence.

    .. note::
        This module will only work with :obj:`~gpytorch.likelihoods.GaussianLikelihood`.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The likelihood for the model
    :param ~gpytorch.models.ApproximateGP model: The approximate GP model
    :param int num_data: The total number of training data points (necessary for SGD)
    :param float beta: (optional, default=1.) A multiplicative factor for the KL divergence term.
        Setting it to anything less than 1 reduces the regularization effect of the model
        (similarly to what was proposed in `the beta-VAE paper`_).
    :param float gamma: (optional, default=1.03) The :math:`\gamma`-divergence hyperparameter.
    :param bool combine_terms: (default=True): Whether or not to sum the
        expected NLL with the KL terms (default True)

    Example:
        >>> # model is a gpytorch.models.ApproximateGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.GammaRobustVariationalELBO(likelihood, model, num_data=100, beta=0.5, gamma=1.03)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()

    .. _Knoblauch, 2019:
        https://arxiv.org/pdf/1904.02303.pdf
    .. _Knoblauch, Jewson, Damoulas 2019:
        https://arxiv.org/pdf/1904.02063.pdf
    """

    def __init__(self, likelihood, model, gamma=1.03, *args, **kwargs):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super().__init__(likelihood, model, *args, **kwargs)
        if gamma <= 1.0:
            raise ValueError("gamma should be > 1.0")
        self.gamma = gamma

    def _log_likelihood_term(self, variational_dist_f, target, *args, **kwargs):
        shifted_gamma = self.gamma - 1

        muf, varf = variational_dist_f.mean, variational_dist_f.variance

        # Get noise from likelihood
        noise = self.likelihood._shaped_noise_covar(muf.shape, *args, **kwargs).diag()
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *variational_dist_f.event_shape)

        # adapted from https://github.com/JeremiasKnoblauch/GVIPublic/
        mut = shifted_gamma * target / noise + muf / varf
        sigmat = 1.0 / (shifted_gamma / noise + 1.0 / varf)
        log_integral = -0.5 * shifted_gamma * torch.log(2.0 * math.pi * noise) - 0.5 * np.log1p(shifted_gamma)
        log_tempered = (
            -math.log(shifted_gamma)
            - 0.5 * shifted_gamma * torch.log(2.0 * math.pi * noise)
            - 0.5 * torch.log1p(shifted_gamma * varf / noise)
            - 0.5 * (shifted_gamma * target.pow(2.0) / noise)
            - 0.5 * muf.pow(2.0) / varf
            + 0.5 * mut.pow(2.0) * sigmat
        )

        factor = log_tempered + shifted_gamma / self.gamma * log_integral
        factor = self.gamma * factor.exp()

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(variational_dist_f.event_shape)
        if num_event_dim > 1:
            factor = factor.sum(list(range(-1, -num_event_dim, -1)))

        return factor.sum(-1)
