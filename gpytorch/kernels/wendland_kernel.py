#!/usr/bin/env python3

import torch

from .kernel import Kernel


def phi_0(r):
    return torch.max(torch.tensor(0.0), 1 - r)  # np.maximum(0, 1-r)


def phi_1(r):
    return phi_0(r) ** 3 * (3 * r + 1)


def phi_2(r):
    return phi_0(r) ** 5 * (8 * r ** 2 + 5 * r + 1)


def wendland_kernel_function(x1, x2, k=0, lambdas=1):
    weighted_element_difference = torch.abs(x1 - x2).div(lambdas)
    if k == 0:
        phi = phi_0(weighted_element_difference)
    elif k == 1:
        phi = phi_1(weighted_element_difference)
    elif k == 2:
        phi = phi_2(weighted_element_difference)
    else:
        raise NotImplementedError()
    return torch.prod(phi, axis=-1)


class WendlandKernel(Kernel):
    r"""
    Computes a covariance matrix based on the tensor product version of the one-dimensional Wendland kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` (the smoothness can be chosen from
    k=0 (C^0 continuity), k=1 (C^2 continuity) and k=2 (C^4 continuity))
    See Wendland, H. (2004). Scattered Data Approximation (Cambridge Monographs on Applied and Computational
    Mathematics). Cambridge: Cambridge University Press. doi:10.1017/CBO9780511617539 p.129:

    Case k=2:
    .. math::

       \begin{equation*}
          r_d = \frac{|x1_d-x2_d|}{\Theta_d}
          k_{\text{Wendland}}(\mathbf{x_1}, \mathbf{x_2}) = \prod_{d=1}^D (1-r_d)_+^5 (8r_d^2+5r_d+1)
       \end{equation*}

    where :math:`\Theta` is a :attr:`lengthscale` parameter (:math:`\Theta_d` can either all be equal
    or different per dimension depending on the parameter :attr:`ard_num_dims`) and D is the number of dimensions.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::
        This kernel is useful for inducing a sparse kernel gram matrix since points that are in at least
        one dimension d further apart than :math:`\Theta_d` are assigned 0 covariance.
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

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WendlandKernel())
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WendlandKernel(ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>>
        >>> # Batch Mode not well tested! But should work like the following:
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WendlandKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.WendlandKernel(batch_shape=torch.Size([2])))
        >>> covar = covar_module(batch_x)  # Output: LazyTensor of size (2 x 10 x 10)
    """
    has_lengthscale = True

    def __init__(self, k=0, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def forward(self, x1, x2, diag=False, vectorized=True, **params):
        """
        Args:
            vectorized (bool, optional):
                Vectorization of the computation will result in much faster computations for moderate dimensions
                at the cost of a somewhat larger memory footprint.
                For very high dimensions (~>1000) the non vectorized mode is usually faster and
                saves a lot of memory.
        """
        x1_eq_x2 = torch.equal(x1, x2)
        if diag:
            if x1_eq_x2:
                return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            raise NotImplementedError
        # Vector computation mode
        if vectorized:
            x1_enlarged = x1.unsqueeze(1)
            dm = wendland_kernel_function(x1_enlarged, x2, k=self.k, lambdas=self.lengthscale)
            return dm
        # Entry-wise computation mode. For a moderate number of dimensions this will be much slower,
        # however, when the number of dimensions is very high this mode can be faster and save a lot of memory.
        dm = torch.empty((x1.shape[0], x2.shape[0]))
        for i in torch.arange(x1.shape[0]):
            for j in torch.arange(x2.shape[0]):
                dm[i, j] = wendland_kernel_function(x1[i], x2[j], self.k, lambdas=self.lengthscale)
        return dm
