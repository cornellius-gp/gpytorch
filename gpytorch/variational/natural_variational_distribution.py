#!/usr/bin/env python3

import abc

import torch

from ..distributions import MultivariateNormal
from ..lazy import CholLazyTensor, TriangularLazyTensor
from ..utils.cholesky import psd_safe_cholesky
from ._variational_distribution import _VariationalDistribution


class _NaturalVariationalDistribution(_VariationalDistribution, abc.ABC):
    r"""Any :obj:`~gpytorch.variational._VariationalDistribution` which calculates
    natural gradients with respect to its parameters.
    """
    pass


class NaturalVariationalDistribution(_NaturalVariationalDistribution):
    r"""A multivariate normal :obj:`~gpytorch.variational._VariationalDistribution`,
    parameterized by **natural** parameters.

    .. note::
       The :obj:`~gpytorch.variational.NaturalVariationalDistribution` can only
       be used with :obj:`gpytorch.optim.NGD`, or other optimizers
       that follow exactly the gradient direction. Failure to do so will cause
       the natural matrix :math:`\mathbf \Theta_\text{mat}` to stop being
       positive definite, and a :obj:`~RuntimeError` will be raised.

    .. seealso::
        The `natural gradient descent tutorial
        <examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb>`_
        for use instructions.

        The :obj:`~gpytorch.variational.TrilNaturalVariationalDistribution` for
        a more numerically stable parameterization, at the cost of needing more
        iterations to make variational regression converge.

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
        mean, chol_covar = _NaturalToMuVarSqrt.apply(self.natural_vec, self.natural_mat)
        res = MultivariateNormal(mean, CholLazyTensor(TriangularLazyTensor(chol_covar)))
        return res

    def initialize_variational_distribution(self, prior_dist):
        prior_prec = prior_dist.covariance_matrix.inverse()
        prior_mean = prior_dist.mean
        noise = torch.randn_like(prior_mean).mul_(self.mean_init_std)

        self.natural_vec.data.copy_((prior_prec @ prior_mean.unsqueeze(-1)).squeeze(-1).add_(noise))
        self.natural_mat.data.copy_(prior_prec.mul(-0.5))


def _triangular_inverse(A, upper=False):
    eye = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return eye.triangular_solve(A, upper=upper).solution


def _phi_for_cholesky_(A):
    "Modifies A to be the phi function used in differentiating through Cholesky"
    A.tril_().diagonal(offset=0, dim1=-2, dim2=-1).mul_(0.5)
    return A


def _cholesky_backward(dout_dL, L, L_inverse):
    # c.f. https://github.com/pytorch/pytorch/blob/25ba802ce4cbdeaebcad4a03cec8502f0de9b7b3/
    #      tools/autograd/templates/Functions.cpp
    A = L.transpose(-1, -2) @ dout_dL
    phi = _phi_for_cholesky_(A)
    grad_input = (L_inverse.transpose(-1, -2) @ phi) @ L_inverse
    # Symmetrize gradient
    return grad_input.add(grad_input.transpose(-1, -2)).mul_(0.5)


class _NaturalToMuVarSqrt(torch.autograd.Function):
    @staticmethod
    def _forward(nat_mean, nat_covar):
        try:
            L_inv = psd_safe_cholesky(-2.0 * nat_covar, upper=False)
        except RuntimeError as e:
            if str(e).startswith("cholesky"):
                raise RuntimeError(
                    "Non-negative-definite natural covariance. You probably "
                    "updated it using an optimizer other than gpytorch.optim.NGD (such as Adam). "
                    "This is not supported."
                )
            else:
                raise e
        L = _triangular_inverse(L_inv, upper=False)
        S = L.transpose(-1, -2) @ L
        mu = (S @ nat_mean.unsqueeze(-1)).squeeze(-1)
        # Two choleskys are annoying, but we don't have good support for a
        # LazyTensor of form L.T @ L
        return mu, psd_safe_cholesky(S, upper=False)

    @staticmethod
    def forward(ctx, nat_mean, nat_covar):
        mu, L = _NaturalToMuVarSqrt._forward(nat_mean, nat_covar)
        ctx.save_for_backward(mu, L)
        return mu, L

    @staticmethod
    def _backward(dout_dmu, dout_dL, mu, L, C):
        """Calculate dout/d(eta1, eta2), which are:
        eta1 = mu
        eta2 = mu*mu^T + LL^T = mu*mu^T + Sigma

        Thus:
        dout/deta1 = dout/dmu + dout/dL dL/deta1
        dout/deta2 = dout/dL dL/deta1

        For L = chol(eta2 - eta1*eta1^T).
        dout/dSigma = _cholesky_backward(dout/dL, L)
        dout/deta2 = dout/dSigma
        dSigma/deta1 = -2* (dout/dSigma) mu
        """
        dout_dSigma = _cholesky_backward(dout_dL, L, C)
        dout_deta1 = dout_dmu - 2 * (dout_dSigma @ mu.unsqueeze(-1)).squeeze(-1)
        return dout_deta1, dout_dSigma

    @staticmethod
    def backward(ctx, dout_dmu, dout_dL):
        "Calculates the natural gradient with respect to nat_mean, nat_covar"
        mu, L = ctx.saved_tensors
        C = _triangular_inverse(L, upper=False)
        return _NaturalToMuVarSqrt._backward(dout_dmu, dout_dL, mu, L, C)
