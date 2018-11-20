#!/usr/bin/env python3

from ..module import Module


class Likelihood(Module):
    """
    A Likelihood in GPyTorch specifies the mapping from latent function values
    f to observed labels y.

    For example, in the case of regression this might be a Gaussian
    distribution, as y(x) is equal to f(x) plus Gaussian noise:

            y(x) = f(x) + \epsilon, \epsilon ~ N(0,\sigma^{2}_{n} I)

    In the case of classification, this might be a Bernoulli distribution,
    where the probability that y=1 is given by the latent function
    passed through some sigmoid or probit function:

            y(x) = 1 w/ probability \sigma(f(x)), -1 w/ probability 1-\sigma(f(x))

    In either case, to implement a (non-Gaussian) likelihood function, GPyTorch
    requires that two methods be implemented:

    1. A forward method that computes predictions p(y*|x*) given a distribution
       over the latent function p(f*|x*). Typically, this solves or
       approximates the integral:

            p(y*|x*) = \int p(y*|f*)p(f*|x*) df*

    2. A variational_log_probability method that computes the log probability
        \log p(y|f) from a set of samples of f. This is only used for variational
        inference.
    """

    def forward(self, *inputs, **kwargs):
        """
        Computes a predictive distribution p(y*|x*) given either a posterior
        distribution p(f|D,x) or a prior distribution p(f|x) as input.

        With both exact inference and variational inference, the form of
        p(f|D,x) or p(f|x) should usually be Gaussian. As a result, input
        should usually be a MultivariateNormal specified by the mean and
        (co)variance of p(f|...).
        """
        raise NotImplementedError

    def variational_log_probability(self, f, y):
        """
        Compute the log likelihood p(y|f) given y and averaged over a set of
        latent function samples.

        For the purposes of our variational inference implementation, y is an
        n-by-1 label vector, and f is an n-by-s matrix of s samples from the
        variational posterior, q(f|D).
        """
        raise NotImplementedError

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        raise NotImplementedError
