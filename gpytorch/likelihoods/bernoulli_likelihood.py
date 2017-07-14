import torch
from gpytorch.math.functions import NormalCDF, LogNormalCDF
from gpytorch.random_variables import GaussianRandomVariable, IndependentRandomVariables, BernoulliRandomVariable
from .likelihood import Likelihood


class BernoulliLikelihood(Likelihood):
    """
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF \Phi(x)). Given the identity \Phi(-x) =
    1-\Phi(x), we can write the likelihood compactly as:

    p(Y=y|f)=\Phi(yf)
    """

    def forward(self, input):
        """
        Computes predictive distributions p(y|x) given a latent distribution
        p(f|x). To do this, we solve the integral:

            p(y|x) = \int p(y|f)p(f|x) df

        Given that p(y=1|f) = \Phi(f), this integral is analytically tractable,
        and if \mu_f and \sigma^2_f are the mean and variance of p(f|x), the
        solution is given by:

            p(y|x) = \Phi(\frac{\mu}{\sqrt{1+\sigma^2_f}})
        """
        if not isinstance(input, GaussianRandomVariable):
            raise RuntimeError(' '.join([
                'BernoulliLikelihood expects a Gaussian',
                'distributed latent function to make predictions',
            ]))

        mean = input.mean()
        var = input.var()

        link = mean.div(torch.sqrt(1 + var))

        output_probs = NormalCDF()(link)
        return IndependentRandomVariables([BernoulliRandomVariable(output_prob)
                                           for output_prob in output_probs])

    def log_probability(self, f, y):
        """
        Computes the log probability \sum_{i} \log \Phi(y_{i}f_{i}), where
        \Phi(y_{i}f_{i}) is computed by averaging over a set of s samples of
        f_{i} drawn from p(f|x).
        """
        return LogNormalCDF()(f.mul(y.unsqueeze(1).expand_as(f))).mean(1).sum(0)
