#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..lazy import CholLazyTensor
from ..utils.cholesky import psd_safe_cholesky
from ._variational_distribution import _VariationalDistribution


class _ExpectationParams(torch.autograd.Function):
    """
    Given the canonical params, computes the expectation params

    expec_mean = m = -1/2 natural_mat^{-1} natural_vec
    expec_covar = m m^T + S = [ expec_mean expec_mean^T - 0.5 natural_mat^{-1} ]

    On the backward pass, we simply pass the gradients from the expected parameters into the canonical params
    This will allow us to perform NGD
    """

    @staticmethod
    def forward(ctx, natural_vec, natural_mat):
        cov_mat = natural_mat.inverse().mul_(-0.5)
        expec_mean = cov_mat @ natural_vec
        expec_covar = (expec_mean.unsqueeze(-1) @ expec_mean.unsqueeze(-2)).add_(cov_mat)
        return expec_mean, expec_covar

    @staticmethod
    def backward(ctx, expec_mean_grad, expec_covar_grad):
        return expec_mean_grad, expec_covar_grad


class NaturalVariationalDistribution(_VariationalDistribution):
    r"""
    A multivariate normal :obj:`~gpytorch.variational._VariationalDistribution`,
    parameterized by **natural** parameters.

    If the variational distribution is defined by :math:`\mathcal{N}(\mathbf m, \mathbf S)`, then
    a :obj:`~gpytorch.variational.NaturalVariationalDistribution` uses the parameterization:

    .. math::

        \begin{align*}
            \boldsymbol \theta &= \mathbf S^{-1} \mathbf m
            \\
            \boldsymbol \Theta &= -\frac{1}{2} \mathbf S^{-1}.
        \end{align*}

    This is for use with natural gradient descent (see e.g. `Salimbeni et al., 2018`_).

    .. seealso::
        The `natural gradient descent tutorial
        <examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb>`_
        for use instructions.

    .. note::
        Natural gradient descent is very stable with variational regression, but can be unstable with
        non-conjugate likelihoods and alternative objective functions.

    .. _Salimbeni et al., 2018:
        https://arxiv.org/abs/1803.09151

    :param int num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :type batch_shape: :obj:`torch.Size`, optional
    :param float mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        scaled_mean_init = torch.zeros(num_inducing_points)
        neg_prec_init = torch.eye(num_inducing_points, num_inducing_points).mul(-0.5)
        scaled_mean_init = scaled_mean_init.repeat(*batch_shape, 1)
        neg_prec_init = neg_prec_init.repeat(*batch_shape, 1, 1)

        # eta1 and eta2 parameterization of the variational distribution
        self.register_parameter(name="natural_vec", parameter=torch.nn.Parameter(scaled_mean_init))
        self.register_parameter(name="natural_mat", parameter=torch.nn.Parameter(neg_prec_init))

    def forward(self):
        # First - get the natural/canonical parameters (\theta in Hensman, 2013, eqn. 6)
        # This is what's computed by super().forward()
        # natural_vec = S^{-1} m
        # natural_mat = -1/2 S^{-1}
        natural_vec = self.natural_vec
        natural_mat = self.natural_mat

        # From the canonical parameters, compute the expectation parameters (\eta in Hensman, 2013, eqn. 6)
        # expec_mean = m = -1/2 natural_mat^{-1} natural_vec
        # expec_covar = m m^T + S = [ expec_mean expec_mean^T - 0.5 natural_mat^{-1} ]
        expec_mean, expec_covar = _ExpectationParams().apply(natural_vec, natural_mat)

        # Finally, convert the expected parameters into m and S
        # m = expec_mean
        # S = expec_covar - expec_mean expec_mean^T
        mean = expec_mean
        chol_covar = psd_safe_cholesky(expec_covar - expec_mean.unsqueeze(-1) @ expec_mean.unsqueeze(-2), max_tries=4)
        return MultivariateNormal(mean, CholLazyTensor(chol_covar))

    def initialize_variational_distribution(self, prior_dist):
        prior_prec = prior_dist.covariance_matrix.inverse()
        prior_mean = prior_dist.mean
        noise = torch.randn_like(prior_mean).mul_(self.mean_init_std)

        self.natural_vec.data.copy_((prior_prec @ prior_mean).add_(noise))
        self.natural_mat.data.copy_(prior_prec.mul(-0.5))
