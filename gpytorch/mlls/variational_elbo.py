#!/usr/bin/env python3

import torch
from ._approximate_mll import _ApproximateMarginalLogLikelihood
from .. import settings


class VariationalELBO(_ApproximateMarginalLogLikelihood):
    r"""
    The variational evidence lower bound (ELBO). This is used to optimize
    variational Gaussian processes (with or without stochastic optimization).

    .. math::

       \begin{align*}
          \mathcal{L}_\text{ELBO} &=
          \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
            \log \mathbb{E}_{q(\mathbf u)} \left[  p( y \! \mid \! \mathbf u) \right]
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
          \\
          &\approx \sum_{i=1}^N \mathbb{E}_{q( \mathbf u)} \left[
            \log \int p( y_i \! \mid \! f_i) p(f_i \! \mid \! \mathbf u) \: d \mathbf f_i
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the amount of data, :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, and `p(\mathbf u)` is the prior distribution for the inducing function
    values.

    :math:`\beta` is a scaling constant that reduces the regularization effect of the KL
    divergence. Setting :math:`\beta=1` (default) results in the true variational ELBO.

    For more information on this derivation, see `Scalable Variational Gaussian Process Classification`_
    (Hensman et al., 2015).

    Args:
        :attr:`likelihood` (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`model` (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        :attr:`num_data` (int):
            The total number of training data points (necessary for SGD)
        :attr:`beta` (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).
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

    .. _Scalable Variational Gaussian Process Classification:
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _the beta-VAE paper:
        https://openreview.net/pdf?id=Sy2fzU9gl
    """

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs)

    def forward(self, variational_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`variational_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        return super().forward(variational_dist_f, target, **kwargs)


class VariationalELBOEmpirical(VariationalELBO):
    def __init__(self, likelihood, model, num_data):
        """
        A special MLL designed for variational inference.
        This computes an empirical (rather than exact) estimate of the KL divergence

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        """
        super(VariationalELBOEmpirical, self).__init__(likelihood, model, num_data, combine_terms=True)

    def forward(self, variational_dist_f, target, **kwargs):
        num_batch = variational_dist_f.event_shape[0]
        variational_dist_u = self.model.variational_strategy.variational_distribution
        prior_dist = self.model.variational_strategy.prior_distribution

        log_likelihood = self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs)
        log_likelihood = log_likelihood.div(num_batch)

        num_samples = settings.num_likelihood_samples.value()
        variational_samples = variational_dist_u.rsample(torch.Size([num_samples]))
        kl_divergence = (
            variational_dist_u.log_prob(variational_samples) - prior_dist.log_prob(variational_samples)
        ).mean(0)
        kl_divergence = kl_divergence.div(self.num_data)

        res = log_likelihood - kl_divergence
        for _, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure()).sum().div(self.num_data))
        return res
