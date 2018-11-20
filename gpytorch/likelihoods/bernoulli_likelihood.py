#!/usr/bin/env python3

import torch
from torch.distributions import Bernoulli

from .. import settings
from ..distributions import MultivariateNormal
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
        num_samples = settings.num_likelihood_samples.value()
        samples = latent_func.rsample(torch.Size([num_samples])).view(-1)
        target = target.unsqueeze(0).repeat(num_samples, 1).view(-1)
        return log_normal_cdf(samples.mul(target)).sum().div(num_samples)

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        import pyro

        f_samples = variational_dist_f(sample_shape)
        y_prob_samples = torch.distributions.Normal(0, 1).cdf(f_samples)
        y_dist = pyro.distributions.Bernoulli(y_prob_samples)
        pyro.sample(name_prefix + "._training_labels", y_dist.independent(1), obs=y_obs)
