#!/usr/bin/env python3

from .kernel import Kernel
from ..functions import RBFCovariance


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class WeirdRBFKernel(Kernel):
    r"""
    Computes a covariance matrix based on the RBF (squared exponential) kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::

       \begin{equation*}
          k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left( -\frac{1}{2}
          (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-1} (\mathbf{x_1} - \mathbf{x_2}) \right)
       \end{equation*}

    where :math:`\Theta` is a :attr:`lengthscale` parameter.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `None`
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.
    """

    def __init__(self, **kwargs):
        super(WeirdRBFKernel, self).__init__(has_lengthscale=True, **kwargs)
        self.factor = kwargs.pop('factor', 10.0)

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        k1 = self.covar_dist(x1_, x2_, square_dist=True, diag=diag,
                             dist_postprocess_func=postprocess_rbf, postprocess=True, **params)
        k2 = self.covar_dist(x1_ / self.factor , x2_ / self.factor, square_dist=True, diag=diag,
                             dist_postprocess_func=postprocess_rbf, postprocess=True, **params) / (self.factor * self.factor)
        return k1 + k2

