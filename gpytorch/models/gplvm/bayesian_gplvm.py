#!/usr/bin/env python3

from ..approximate_gp import ApproximateGP


class BayesianGPLVM(ApproximateGP):
    """
    The Gaussian Process Latent Variable Model (GPLVM) class for unsupervised learning.
    The class supports

    1. Point estimates for latent X when prior_x = None
    2. MAP Inference for X when prior_x is not None and inference == 'map'
    3. Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

    .. seealso::
        The `GPLVM tutorial
        <examples/04_Variational_and_Approximate_GPs/Gaussian_Process_Latent_Variable_Models_with_Stochastic_Variational_Inference.ipynb>`_
        for use instructions.

    :param X: An instance of a sub-class of the LatentVariable class. One of,
        :class:`~gpytorch.models.gplvm.PointLatentVariable`, :class:`~gpytorch.models.gplvm.MAPLatentVariable`, or
        :class:`~gpytorch.models.gplvm.VariationalLatentVariable`, to facilitate inference with 1, 2, or 3 respectively.
    :type X: ~gpytorch.models.LatentVariable
    :param ~gpytorch.variational._VariationalStrategy variational_strategy: The strategy that determines
        how the model marginalizes over the variational distribution (over inducing points)
        to produce the approximate posterior distribution (over data)
    """

    def __init__(self, X, variational_strategy):
        super().__init__(variational_strategy)

        # Assigning Latent Variable
        self.X = X

    def forward(self):
        raise NotImplementedError

    def sample_latent_variable(self):
        sample = self.X()
        return sample
