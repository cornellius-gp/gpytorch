#!/usr/bin/env python3

from ._approximate_mll import _ApproximateMarginalLogLikelihood


class PredictiveLogLikelihood(_ApproximateMarginalLogLikelihood):
    r"""
    An alternative objective function for approximate GPs, proposed in `Jankowiak et al., 2020`_.
    It typically produces better predictive variances than the :obj:`gpytorch.mlls.VariationalELBO` objective.

    .. math::

       \begin{align*}
          \mathcal{L}_\text{ELBO} &=
          \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
            \log p( y \! \mid \! \mathbf x)
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
          \\
          &\approx \sum_{i=1}^N \log \mathbb{E}_{q(\mathbf u)} \left[
            \int p( y_i \! \mid \! f_i) p(f_i \! \mid \! \mathbf u, \mathbf x_i) \: d f_i
          \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
       \end{align*}

    where :math:`N` is the total number of datapoints, :math:`q(\mathbf u)` is the variational distribution for
    the inducing function values, and :math:`p(\mathbf u)` is the prior distribution for the inducing function
    values.

    :math:`\beta` is a scaling constant that reduces the regularization effect of the KL
    divergence. Setting :math:`\beta=1` (default) results in an objective that can be motivated by a connection
    to Stochastic Expectation Propagation (see `Jankowiak et al., 2020`_ for details).

    .. note::
        This objective is very similar to the variational ELBO.
        The only difference is that the :math:`log` occurs *outside* the expectation :math:`\mathbb{E}_{q(\mathbf u)}`.
        This difference results in very different predictive performance (see `Jankowiak et al., 2020`_).

    :param ~gpytorch.likelihoods.Likelihood likelihood: The likelihood for the model
    :param ~gpytorch.models.ApproximateGP model: The approximate GP model
    :param int num_data: The total number of training data points (necessary for SGD)
    :param float beta: (optional, default=1.) A multiplicative factor for the KL divergence term.
        Setting it to anything less than 1 reduces the regularization effect of the model
        (similarly to what was proposed in `the beta-VAE paper`_).
    :param bool combine_terms: (default=True): Whether or not to sum the
        expected NLL with the KL terms (default True)

    Example:
        >>> # model is a gpytorch.models.ApproximateGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=100, beta=0.5)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()

    .. _Jankowiak et al., 2020:
        https://arxiv.org/abs/1910.07123
    """

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        return self.likelihood.log_marginal(target, approximate_dist_f, **kwargs).sum(-1)

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the predictive cross entropy given :math:`q(\mathbf f)` and :math:`\mathbf y`.
        Calling this function will call the likelihood's
        :meth:`~gpytorch.likelihoods.Likelihood.forward` function.

        :param ~gpytorch.distributions.MultivariateNormal variational_dist_f: :math:`q(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param kwargs: Additional arguments passed to the
            likelihood's :meth:`~gpytorch.likelihoods.Likelihood.forward` function.
        :rtype: torch.Tensor
        :return: Predictive log likelihood. Output shape corresponds to batch shape of the model/input data.
        """
        return super().forward(approximate_dist_f, target, **kwargs)
