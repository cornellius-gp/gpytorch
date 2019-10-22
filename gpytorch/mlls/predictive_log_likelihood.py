#!/usr/bin/env python3

import torch
from ._approximate_mll import _ApproximateMarginalLogLikelihood
from ..distributions import MultivariateNormal


class PredictiveLogLikelihood(_ApproximateMarginalLogLikelihood):
    r"""
    An alternative objective function for approximate GPs, proposed in `Jankowiak et al., 2019`_.
    It typically produces predictive variances than the :obj:`gpytorch.mlls.VariationalELBO` objective.

    .. math::

       \begin{align*}
          \mathcal{L}_\text{ELBO} &=
          \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
            \log p( y \! \mid \! \mathbf x)
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
          \\
          &\approx \sum_{i=1}^N \log \mathbb{E}_{q(\mathbf u)} \left[
            \int p( y_i \! \mid \! f_i) p(f_i \! \mid \! \mathbf u) \: d f_i
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the amount of data, :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, and `p(\mathbf u)` is the prior distribution for the inducing function
    values.

    :math:`\beta` is a scaling constant that reduces the regularization effect of the KL
    divergence. Setting :math:`\beta=1` (default) results in the true variational ELBO.

    .. note::
        This objective is very similar to the variational ELBO.
        The only difference is that the :math:`log` occurs *outside* the expectation :math:`\mathbb E_{q(\mathbf u}`.
        This difference results in very different predictive performance (see `Jankowiak et al., 2019`_).

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
        >>> mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=100, beta=0.5)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()

    .. _Jankowiak et al., 2019:
        http://bit.ly/predictive_gp
    """

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        # Compute predictive distribution
        target_dist = self.likelihood(approximate_dist_f, **kwargs)
        num_sample_dims = len(target_dist.batch_shape) - len(approximate_dist_f.batch_shape)

        # Make sure that predictive distribution factorizes over inputs (if we have a MultivariateNormal)
        if isinstance(target_dist, MultivariateNormal):
            target_dist = torch.distributions.Normal(target_dist.mean, target_dist.variance.sqrt())

        # Compute log p(y | x)
        res = target_dist.log_prob(target).sum(-1)

        # Sum over sample dimensions
        if num_sample_dims:
            res = res.sum(dim=list(range(num_sample_dims)))

        # We're done!
        return res

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the predictive cross entropy given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `forward` function.

        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `forward` function.
        """
        return super().forward(approximate_dist_f, target, **kwargs)
