import math
from .marginal_log_likelihood import MarginalLogLikelihood
from ..lazy import LazyVariable, NonLazyVariable
from ..likelihoods import GaussianLikelihood


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
        if not isinstance(covar, LazyVariable):
            covar = NonLazyVariable(covar)

        # Get log determininat and first part of quadratic form
        inv_quad, log_det = covar.inv_quad_log_det(inv_quad_rhs=target.unsqueeze(-1), log_det=True)
        res = -0.5 * sum([
            inv_quad,
            log_det,
            n_data * math.log(2 * math.pi)
        ])
        return res.div(n_data)
