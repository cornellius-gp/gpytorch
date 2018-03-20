from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .marginal_log_likelihood import MarginalLogLikelihood
from ..lazy import LazyVariable
from ..likelihoods import GaussianLikelihood
from ..utils import function_factory


_exact_gp_mll_class = function_factory.exact_gp_mll_factory()


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood, model):
        """
        A special MLL designed for exact inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the exact GP model
        """

        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError('Likelihood must be Gaussian for exact inference')
        super(ExactMarginalLogLikelihood, self).__init__(likelihood, model)

    def forward(self, output, target):
        mean, covar = self.likelihood(output).representation()
        n_data = target.size(-1)
        if isinstance(covar, LazyVariable):
            res = covar.exact_gp_marginal_log_likelihood(target - mean)
        else:
            res = _exact_gp_mll_class()(covar, target - mean)
        return res.div(n_data)
