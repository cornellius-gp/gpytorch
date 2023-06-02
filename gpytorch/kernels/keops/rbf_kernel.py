#!/usr/bin/env python3

# from linear_operator.operators import KeOpsLinearOperator
from linear_operator.operators import KernelLinearOperator

from ... import settings
from ..rbf_kernel import postprocess_rbf
from .keops_kernel import KeOpsKernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class RBFKernel(KeOpsKernel):
        r"""
        Implements the RBF kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for :class:`gpytorch.kernels.RBFKernel` in most cases,
        and supports the same arguments.

        :param ard_num_dims: Set this if you want a separate lengthscale for each input
            dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
        :param batch_shape: Set this if you want a separate lengthscale for each batch of input
            data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
            a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
        :param active_dims: Set this if you want to compute the covariance of only
            a few input dimensions. The ints corresponds to the indices of the
            dimensions. (Default: `None`.)
        :param lengthscale_prior: Set this if you want to apply a prior to the
            lengthscale parameter. (Default: `None`)
        :param lengthscale_constraint: Set this if you want to apply a constraint
            to the lengthscale parameter. (Default: `Positive`.)
        :param eps: The minimum value that the lengthscale can take (prevents
            divide by zero errors). (Default: `1e-6`.)

        :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
            ard_num_dims and batch_shape arguments.
        """

        has_lengthscale = True

        def _nonkeops_covar_func(self, x1, x2, diag=False):
            return postprocess_rbf(self.covar_dist(x1, x2, square_dist=True, diag=diag))

        def covar_func(self, x1, x2, **kwargs):
            if x1.size(-2) < settings.max_cholesky_size.value() or x2.size(-2) < settings.max_cholesky_size.value():
                return self._nonkeops_covar_func(x1, x2)

            x1_ = KEOLazyTensor(x1[..., :, None, :])
            x2_ = KEOLazyTensor(x2[..., None, :, :])

            K = (-((x1_ - x2_) ** 2).sum(-1) / 2).exp()

            return K

        def forward(self, x1, x2, diag=False, **kwargs):

            x1_ = x1 / self.lengthscale
            x2_ = x2 / self.lengthscale

            if diag:
                return self._nonkeops_covar_func(x1_, x2_, diag=diag)

            # return KernelLinearOperator inst only when calculating the whole covariance matrix
            return KernelLinearOperator(x1_, x2_, covar_func=self.covar_func, **kwargs)

except ImportError:

    class RBFKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
