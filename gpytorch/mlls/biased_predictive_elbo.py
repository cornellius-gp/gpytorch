#!/usr/bin/env python3

from ._approximate_mll import _ApproximateMarginalLogLikelihood
import torch
import math


class BiasedPredictiveELBO(_ApproximateMarginalLogLikelihood):
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        logmarginal = self.likelihood.log_marginal(target, variational_dist_f, **kwargs)
        marginal = torch.logsumexp(logmarginal, dim=0) - math.log(logmarginal.size(0))
        return marginal.sum()

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
