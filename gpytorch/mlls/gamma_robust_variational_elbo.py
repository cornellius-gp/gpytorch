#!/usr/bin/env python3

import math
import torch
from ._approximate_mll import _ApproximateMarginalLogLikelihood
from ..likelihoods import _GaussianLikelihoodBase


class GammaRobustVariationalELBO(_ApproximateMarginalLogLikelihood):
    r"""
    An alternative to the variational evidence lower bound (ELBO), proposed by `Knoblauch, 2019`_.
    It is derived by replacing the log-likelihood term in the ELBO with a `\gamma` divergence:

    .. math::

       \begin{align*}
          \mathcal{L}_{\gamma} &=
          \sum_{i=1}^N \mathbb{E}_{q( \mathbf u)} \left[
            -\frac{\gamma}{1 - \gamma}
            \frac{
                p( y_i \! \mid \! \mathbf u)^{\gamma - 1}
            }{
                \int p(\mathbf y \mid \mathbf u) \: d \mathbf y
            }
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the amount of data, :math:`\gamma` is a hyperparameter,
    :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, and `p(\mathbf u)` is the prior distribution for the inducing function
    values.

    :math:`\beta` is a scaling constant for the KL divergence.

    Args:
        :attr:`likelihood` (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`model` (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        :attr:`num_data` (int):
            The total number of training data points (necessary for SGD)
        :attr:`beta` (float - default 1.):
            A multiplicative factor for the KL divergence term.
        :attr:`gamma` (float - default 1.):
            Controls the :math:`\gamma` term in the robust ELBO.
        :attr:`combine_terms` (bool):
            Whether or not to sum the expected NLL with the KL terms (default True)

    Example:
        >>> # model is a gpytorch.models.VariationalGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=100, beta=0.5)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()

    .. _Knoblauch, 2019:
        https://arxiv.org/pdf/1904.02303.pdf
    .. _Knoblauch, Jewson, Damoulas 2019:
        https://arxiv.org/pdf/1904.02063.pdf
    """
    def __init__(self, likelihood, model, gamma=1.0, *args, **kwargs):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super().__init__(likelihood, model, *args, **kwargs)
        self.register_buffer("gamma", torch.tensor(gamma))

    def _log_likelihood_term(self, variational_dist_f, target, *args, **kwargs):
        muf, varf = variational_dist_f.mean, variational_dist_f.variance

        # Get noise from likelihood
        noise = self.likelihood._shaped_noise_covar(muf.shape, *args, **kwargs).diag()
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *variational_dist_f.event_shape)

        # adapted from https://github.com/JeremiasKnoblauch/GVIPublic/
        mut = self.gamma * target / noise + muf / varf
        sigmat = 1.0 / (self.gamma / noise + 1.0 / varf)
        log_integral = -0.5 * self.gamma * torch.log(2.0 * math.pi * noise) - 0.5 * torch.log1p(self.gamma)
        log_tempered = (
            -math.log(self.gamma)
            - 0.5 * self.gamma * torch.log(2.0 * math.pi * noise)
            - 0.5 * torch.log1p(self.gamma * varf / noise)
            - 0.5 * (self.gamma * target.pow(2.0) / noise)
            - 0.5 * muf.pow(2.0) / varf
            + 0.5 * mut.pow(2.0) * sigmat
        )

        factor = log_tempered + self.gamma / (1.0 + self.gamma) * log_integral + (1.0 + self.gamma)

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(variational_dist_f.event_shape)
        if num_event_dim > 1:
            factor = factor.sum(list(range(-1, -num_event_dim, -1)))

        return factor.sum(-1)
