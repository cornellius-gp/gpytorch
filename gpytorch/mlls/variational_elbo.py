#!/usr/bin/env python3

from ._approximate_mll import _ApproximateMarginalLogLikelihood


class VariationalELBO(_ApproximateMarginalLogLikelihood):
    r"""
    The variational evidence lower bound (ELBO). This is used to optimize
    variational Gaussian processes (with or without stochastic optimization).

    .. math::

       \begin{align*}
          \mathcal{L}_\text{ELBO} &=
          \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
            \mathbb{E}_{p(f \mid \mathbf u, \mathbf x) q(\mathbf u)} \left[  \log p( y \! \mid \! f) \right]
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
          \\
          &\approx \sum_{i=1}^N \mathbb{E}_{q( f_i)} \left[
            \log p( y_i \! \mid \! f_i) \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the number of datapoints, :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, :math:`q(f_i)` is the marginal of
    :math:`p(f_i \mid \mathbf u, \mathbf x_i) q(\mathbf u)`,
    and :math:`p(\mathbf u)` is the prior distribution for the inducing function values.

    :math:`\beta` is a scaling constant that reduces the regularization effect of the KL
    divergence. Setting :math:`\beta=1` (default) results in the true variational ELBO.

    For more information on this derivation, see `Scalable Variational Gaussian Process Classification`_
    (Hensman et al., 2015).

    :param ~gpytorch.likelihoods.Likelihood likelihood: The likelihood for the model
    :param ~gpytorch.models.ApproximateGP model: The approximate GP model
    :param int num_data: The total number of training data points (necessary for SGD)
    :param float beta: (optional, default=1.) A multiplicative factor for the KL divergence term.
        Setting it to 1 (default) recovers true variational inference
        (as derived in `Scalable Variational Gaussian Process Classification`_).
        Setting it to anything less than 1 reduces the regularization effect of the model
        (similarly to what was proposed in `the beta-VAE paper`_).
    :param bool combine_terms: (default=True): Whether or not to sum the
        expected NLL with the KL terms (default True)

    Example:
        >>> # model is a gpytorch.models.ApproximateGP
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
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs).sum(-1)

    def forward(self, variational_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and :math:`\mathbf y`.
        Calling this function will call the likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob`
        function.

        :param ~gpytorch.distributions.MultivariateNormal variational_dist_f: :math:`q(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param kwargs: Additional arguments passed to the
            likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob` function.
        :rtype: torch.Tensor
        :return: Variational ELBO. Output shape corresponds to batch shape of the model/input data.
        """
        return super().forward(variational_dist_f, target, **kwargs)
