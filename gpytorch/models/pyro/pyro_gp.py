#!/usr/bin/env python3

import pyro
from ..approximate_gp import ApproximateGP


class PyroGP(ApproximateGP):
    """
    A :obj:`~gpytorch.models.ApproximateGP` designed to work with Pyro.

    This module makes it possible to include GP models with more complex probablistic models,
    or to use likelihood functions with additional variational/approximate distributions.

    The parameters of these models are learned using Pyro's inference tools, unlike other models
    that optimize models with respect to a :obj:`~gpytorch.mlls.MarginalLogLikelihood`.
    See `the Pyro examples <examples/09_Pyro_Integration/index.html>`_ for detailed examples.

    Args:
        :attr:`variational_strategy` (:obj:`~gpytorch.variational.VariationalStrategy`):
            The variational strategy that defines the variational distribution and
            the marginalization strategy.
        :attr:`likelihood` (:obj:`~gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        :attr:`num_data` (int):
            The total number of training data points (necessary for SGD)
        :attr:`name_prefix` (str, optional):
            A prefix to put in front of pyro sample/plate sites
        :attr:`beta` (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).

    Example:
        >>> class MyVariationalGP(gpytorch.models.PyroGP):
        >>>     # implementation
        >>>
        >>> # variational_strategy = ...
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>> model = MyVariationalGP(variational_strategy, likelihood, train_y.size())
        >>>
        >>> optimizer = pyro.optim.Adam({"lr": 0.01})
        >>> elbo = pyro.infer.Trace_ELBO(num_particles=64, vectorize_particles=True)
        >>> svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)
        >>>
        >>> # Optimize variational parameters
        >>> for _ in range(n_iter):
        >>>    loss = svi.step(train_x, train_y)

    .. _Scalable Variational Gaussian Process Classification:
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _the beta-VAE paper:
        https://openreview.net/pdf?id=Sy2fzU9gl
    """
    def __init__(self, variational_strategy, likelihood, num_data, name_prefix="", beta=1.0):
        super().__init__(variational_strategy)
        self.name_prefix = name_prefix
        self.likelihood = likelihood
        self.num_data = num_data
        self.beta = beta

    def guide(self, input, target, *args, **kwargs):
        """
        Gude function for Pyro inference.
        Includes the guide for the GP's likelihood function as well.

        Args:
            :attr:`input` (`torch.Tensor`):
                :math:`\mathbf X` The input values values
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`*args`, :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `forward` function.
        """
        # Hack for getting correct sampling shape
        pyro.sample("__throwaway__", pyro.distributions.Normal(0, 1)).shape

        # Draw samples from q(u) for KL divergence computation
        with pyro.poutine.scale(scale=self.beta):
            inducing_dist = self.variational_strategy.variational_distribution
            # Ensure that no batch dimensions interfere with any plating
            inducing_dist = inducing_dist.to_event(len(inducing_dist.batch_shape))
            pyro.sample(self.name_prefix + ".u", inducing_dist)

        self.likelihood.guide(*args, **kwargs)

    def model(self, input, target, *args, **kwargs):
        """
        Model function for Pyro inference.
        Includes the model for the GP's likelihood function as well.

        Args:
            :attr:`input` (`torch.Tensor`):
                :math:`\mathbf X` The input values values
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`*args`, :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `forward` function.
        """
        pyro.module(self.name_prefix + ".gp_prior", self)

        # Get the variational distribution for the function
        function_dist = self(input)

        # Draw samples from p(u) for KL divergence computation
        with pyro.poutine.scale(scale=self.beta):
            inducing_dist = self.variational_strategy.prior_distribution
            # Ensure that no batch dimensions interfere with any plating
            inducing_dist = inducing_dist.to_event(len(inducing_dist.batch_shape))
            pyro.sample(self.name_prefix + ".u", inducing_dist)

        # Draw samples from the likelihood
        num_minibatch = function_dist.event_shape[0]
        scale = self.num_data / num_minibatch
        return self.likelihood.pyro_sample_output(
            target, function_dist, scale, *args, name_prefix=self.name_prefix, **kwargs
        )
