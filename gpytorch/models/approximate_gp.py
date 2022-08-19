#!/usr/bin/env python3

from .gp import GP
from .pyro import _PyroMixin  # This will only contain functions if Pyro is installed


class ApproximateGP(GP, _PyroMixin):
    r"""
    The base class for any Gaussian process latent function to be used in conjunction
    with approximate inference (typically stochastic variational inference).
    This base class can be used to implement most inducing point methods where the
    variational parameters are learned directly.

    :param ~gpytorch.variational._VariationalStrategy variational_strategy: The strategy that determines
        how the model marginalizes over the variational distribution (over inducing points)
        to produce the approximate posterior distribution (over data)

    The :meth:`forward` function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Example:
        >>> class MyVariationalGP(gpytorch.models.PyroGP):
        >>>     def __init__(self, variational_strategy):
        >>>         super().__init__(variational_strategy)
        >>>         self.mean_module = gpytorch.means.ZeroMean()
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>>
        >>>     def forward(self, x):
        >>>         mean = self.mean_module(x)
        >>>         covar = self.covar_module(x)
        >>>         return gpytorch.distributions.MultivariateNormal(mean, covar)
        >>>
        >>> # variational_strategy = ...
        >>> model = MyVariationalGP(variational_strategy)
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>>
        >>> # optimization loop for variational parameters...
        >>>
        >>> # test_x = ...;
        >>> model(test_x)  # Returns the approximate GP latent function at test_x
        >>> likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    """

    def __init__(self, variational_strategy):
        super().__init__()
        self.variational_strategy = variational_strategy

    def forward(self, x):
        raise NotImplementedError

    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        r"""
        (For Pyro integration only). The component of a `pyro.guide` that
        corresponds to drawing samples from the latent GP function.

        :param torch.Tensor input: The inputs :math:`\mathbf X`.
        :param float beta: (default=1.) How much to scale the :math:`\text{KL} [ q(\mathbf f) \Vert p(\mathbf f) ]`
            term by.
        :param str name_prefix: (default="") A name prefix to prepend to pyro sample sites.
        """
        return super().pyro_guide(input, beta=beta, name_prefix=name_prefix)

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        r"""
        (For Pyro integration only). The component of a `pyro.model` that
        corresponds to drawing samples from the latent GP function.

        :param torch.Tensor input: The inputs :math:`\mathbf X`.
        :param float beta: (default=1.) How much to scale the :math:`\text{KL} [ q(\mathbf f) \Vert p(\mathbf f) ]`
            term by.
        :param str name_prefix: (default="") A name prefix to prepend to pyro sample sites.
        :return: samples from :math:`q(\mathbf f)`
        :rtype: torch.Tensor
        """
        return super().pyro_model(input, beta=beta, name_prefix=name_prefix)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        r"""
        Returns a new GP model that incorporates the specified inputs and targets as new training data using
        online variational conditioning (OVC).

        This function first casts the inducing points and variational parameters into pseudo-points before
        returning an equivalent ExactGP model with a specialized likelihood.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
            If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
            are the same for each target batch.

        :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :return: An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated.
        :rtype: ~gpytorch.models.ExactGP

        Reference: "Conditioning Sparse Variational Gaussian Processes for Online Decision-Making,"
            Maddox, Stanton, Wilson, NeurIPS, '21
            https://papers.nips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html

        """
        return self.variational_strategy.get_fantasy_model(inputs=inputs, targets=targets, **kwargs)

    def __call__(self, inputs, prior=False, **kwargs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        return self.variational_strategy(inputs, prior=prior, **kwargs)
