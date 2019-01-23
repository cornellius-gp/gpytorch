#!/usr/bin/env python3

import torch
from torch.distributions import Bernoulli
from ..distributions import MultivariateNormal
from ..utils.quadrature import GaussHermiteQuadrature1D
from ..functions import log_normal_cdf, normal_cdf
from .likelihood import Likelihood


class BernoulliLikelihood(Likelihood):
    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF \Phi(x)). Given the identity \Phi(-x) =
    1-\Phi(x), we can write the likelihood compactly as:

    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi(yf)
        \end{equation*}
    """
    def __init__(self):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, input):
        if not isinstance(input, MultivariateNormal):
            raise RuntimeError(
                "BernoulliLikelihood expects a multi-variate normally distributed latent function to make predictions"
            )

        mean = input.mean
        var = input.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = normal_cdf(link)
        return Bernoulli(probs=output_probs)

    def variational_log_probability(self, latent_func, target):
        likelihood_func = lambda locs: log_normal_cdf(locs.mul(target.unsqueeze(-1)))
        res = self.quadrature(likelihood_func, latent_func)
        return res.sum()

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        import pyro

        f_samples = variational_dist_f(sample_shape)
        y_prob_samples = torch.distributions.Normal(0, 1).cdf(f_samples)
        y_dist = pyro.distributions.Bernoulli(y_prob_samples)
        pyro.sample(name_prefix + "._training_labels", y_dist.independent(1), obs=y_obs)
