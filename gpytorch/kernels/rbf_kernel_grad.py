#!/usr/bin/env python3
from .rbf_kernel import RBFKernel
import torch
from ..lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor


class RBFKernelGrad(RBFKernel):
    r"""
    Computes a covariance matrix of the RBF kernel that models the covariance
    between the values and partial derivatives for inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`.

    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    Args:
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
             batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([1])`.
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

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())
        >>> covar = covar_module(x)  # Output: LazyTensor of size (60 x 60), where 60 = n * (d + 1)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LazyTensor of size (2 x 60 x 60)
    """
    def forward(self, x1, x2, diag=False, **params):
        b = 1
        if len(x1.size()) == 2:
            n1, d = x1.size()
            n2, d = x2.size()
        else:
            b, n1, d = x1.size()
            _, n2, _ = x2.size()

        K = torch.zeros(b, n1 * (d + 1), n2 * (d + 1), device=x1.device, dtype=x1.dtype)  # batch x n1(d+1) x n2(d+1)
        ell = self.lengthscale.squeeze(-1)

        if not diag:
            # Scale the inputs by the lengthscale (for stability)
            x1_ = x1 / ell
            x2_ = x2 / ell

            # Form all possible rank-1 products for the gradient and Hessian blocks
            outer = x1_.view([b, n1, 1, d]) - x2_.view([b, 1, n2, d])
            outer = torch.transpose(outer, -1, -2).contiguous()

            # 1) Kernel block
            diff = self._covar_dist(x1_, x2_, square_dist=True, **params)
            K_11 = diff.div_(-2).exp_()
            K[..., :n1, :n2] = K_11

            # 2) First gradient block
            outer1 = outer.view([b, n1, n2 * d]) / ell
            K[..., :n1, n2:] = outer1 * K_11.repeat([1, 1, d])

            # 3) Second gradient block
            outer2 = outer.transpose(-1, -3).contiguous().view([b, n2, n1 * d])
            outer2 = outer2.transpose(-1, -2) / ell
            K[..., n1:, :n2] = -outer2 * K_11.repeat([1, d, 1])

            # 4) Hessian block
            outer3 = outer1.repeat([1, d, 1]) * outer2.repeat([1, 1, d])
            kp = KroneckerProductLazyTensor(
                torch.eye(d, d, device=x1.device, dtype=x1.dtype),
                torch.ones(n1, n2, device=x1.device, dtype=x1.dtype)
            )
            chain_rule = kp.evaluate() / ell.pow(2) - outer3
            K[..., n1:, n2:] = chain_rule * K_11.repeat([1, d, d])

            # Symmetrize for stability
            if n1 == n2 and torch.eq(x1, x2).all():
                K = 0.5 * (K.transpose(-1, -2) + K)

            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().contiguous().view((n1 * (d + 1)))
            pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            K = K[..., pi1, :][..., :, pi2]

            return K

        else:  # TODO: This will change when ARD is supported
            if not (n1 == n2 and torch.eq(x1, x2).all()):
                raise RuntimeError("diag=True only works when x1 == x2")

            kernel_diag = super(RBFKernelGrad, self).forward(x1, x2, diag=True)
            grad_diag = torch.ones(1, n2 * d, device=x1.device, dtype=x1.dtype) / (ell.pow(2))
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            pi = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            return k_diag[..., pi]

    def size(self, x1, x2):
        """
        Given `x_1` with `n_1` data points and `x_2` with `n_2` data points, both in
        `d` dimensions, RBFKernelGrad returns an `n_1(d+1) x n_2(d+1)` kernel matrix.
        """
        non_batch_size = ((x1.size(-1) + 1) * x1.size(-2), (x2.size(-1) + 1) * x2.size(-2))
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)
