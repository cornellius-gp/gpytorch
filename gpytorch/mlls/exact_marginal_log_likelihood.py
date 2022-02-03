#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from .marginal_log_likelihood import MarginalLogLikelihood


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    """
    The exact marginal log likelihood (MLL) for an exact Gaussian process with a
    Gaussian likelihood.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model
    :param ~bool combine_terms (optional): If `False`, the MLL call returns each MLL term separately

    Example:
        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model, combine_terms=True):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model, combine_terms)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for name, module, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure(module)).sum())

        return res

    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        split_terms = output.log_prob_terms(target)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()

        if self.combine_terms:
            term_sum = sum(split_terms)
            term_sum = self._add_other_terms(term_sum, params)
            return term_sum.div(num_data)
        else:
            norm_const = split_terms[-1]
            other_terms = torch.zeros_like(norm_const)
            other_terms = self._add_other_terms(other_terms, params)
            split_terms.append(other_terms)
            return [term.div(num_data) for term in split_terms]
