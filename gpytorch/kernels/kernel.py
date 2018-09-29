from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod
import torch
from torch.nn import ModuleList
from gpytorch.lazy import LazyTensor, LazyEvaluatedKernelTensor, ZeroLazyTensor
from gpytorch.module import Module
from gpytorch.utils import prod


class Kernel(Module):
    """
    Kernels in GPyTorch are implemented as a :class:`gpytorch.Module` that, when called on two :obj:`torch.tensor`
    objects `x1` and `x2` returns either a :obj:`torch.tensor` or a :obj:`gpytorch.lazy.LazyTensor` that represents
    the covariance matrix between `x1` and `x2`.

    In the typical use case, to extend this class means to implement the :func:`~gpytorch.kernels.Kernel.forward`
    method.

    .. note::
        The :func:`~gpytorch.kernels.Kernel.__call__` does some additional internal work. In particular,
        all kernels are lazily evaluated so that, in some cases, we can index in to the kernel matrix before actually
        computing it. Furthermore, many built in kernel modules return LazyTensors that allow for more efficient
        inference than if we explicitly computed the kernel matrix itselfself.

        As a result, if you want to use a :obj:`gpytorch.kernels.Kernel` object just to get an actual
        :obj:`torch.tensor` representing the covariance matrix, you may need to call the
        :func:`gpytorch.lazy.LazyTensor.evaluate` method on the output.

    This base :class:`Kernel` class includes a lengthscale parameter
    :math:`\Theta`, which is used by many common kernel functions.
    There are a few options for the lengthscale:

    * Default: No lengthscale (i.e. :math:`\Theta` is the identity matrix).

    * Single lengthscale: One lengthscale can be applied to all input dimensions/batches
      (i.e. :math:`\Theta` is a constant diagonal matrix).
      This is controlled by setting `has_lengthscale=True`.

    * ARD: Each input dimension gets its own separate lengthscale
      (i.e. :math:`\Theta` is a non-constant diagonal matrix).
      This is controlled by the `ard_num_dims` keyword argument (as well has `has_lengthscale=True`).

    In batch-mode (i.e. when :math:`x_1` and :math:`x_2` are batches of input matrices), each
    batch of data can have its own lengthscale parameter by setting the `batch_size`
    keyword argument to the appropriate number of batches.

    .. note::

        The :attr:`lengthscale` parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the :attr:`log_lengthscale_prior` argument, but
        be sure that the prior has :attr:`log_transform=True` set.

    Base Args:
        :attr:`has_lengthscale` (bool):
            Set this if the kernel has a lengthscale. Default: `False`.
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each input
            dimension. It should be `d` if :attr:`x1` is a `n x d` matrix.  Default: `None`
        :attr:`batch_size` (int, optional):
            Set this if you want a separate lengthscale for each batch of input
            data. It should be `b` if :attr:`x1` is a `b x n x d` tensor.  Default: `1`
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`log_lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Base Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_size` arguments.

    Example:
        >>> covar_module = gpytorch.kernels.LinearKernel()
        >>> x1 = torch.randn(50, 3)
        >>> lazy_covar_matrix = covar_module(x1) # Returns a RootLazyTensor
        >>> tensor_covar_matrix = lazy_covar_matrix.evaluate() # Gets the actual tensor for this kernel matrix
    """

    def __init__(
        self,
        has_lengthscale=False,
        ard_num_dims=None,
        batch_size=1,
        active_dims=None,
        log_lengthscale_prior=None,
        eps=1e-6,
    ):
        super(Kernel, self).__init__()
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims
        self.batch_size = batch_size
        self.__has_lengthscale = has_lengthscale
        if has_lengthscale:
            self.eps = eps
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="log_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(batch_size, 1, lengthscale_num_dims)),
                prior=log_lengthscale_prior,
            )

    @property
    def has_lengthscale(self):
        return self.__has_lengthscale

    @property
    def lengthscale(self):
        if self.has_lengthscale:
            return self.log_lengthscale.exp().clamp(self.eps, 1e5)
        else:
            return None

    @property
    def has_custom_exact_predictions(self):
        return False

    def size(self, x1, x2):
        non_batch_size = (x1.size(-2), x2.size(-2))
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)

    @abstractmethod
    def forward_diag(self, x1, x2, **params):
        res = super(Kernel, self).__call__(x1.transpose(-2, -3), x2.transpose(-2, -3), **params)
        if isinstance(res, LazyTensor):
            res = res.evaluate()
        res = res.transpose(-2, -3)
        return res

    @abstractmethod
    def forward(self, x1, x2, **params):
        """
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        .. note::

            All non-compositional kernels should use the :meth:`gpytorch.kernels.Kernel._create_input_grid`
            method to create a meshgrid between x1 and x2 (if necessary).

            Do not manually create the grid - this is inefficient and will cause erroneous behavior in certain
            evaluation modes.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
            The exact size depends on the kernel's evaluation mode:

            * `full_covar`: `n x m` or `b x n x m`
            * `full_covar` with `dim_groups=k`: `k x n x m` or `b x k x n x m`
            * `diag`: `n` or `b x n`
            * `diag` with `dim_groups=k`: `k x n` or `b x k x n`
        """
        raise NotImplementedError()

    def _create_input_grid(self, x1, x2):
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.

        The grid dimensions depend on the kernel's evaluation mode.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the gridded `x1` and `x2`.
            The shape depends on the kernel's mode

            * `full_covar`: (`b x n x 1 x d` and `b x 1 x m x d`)
            * `full_covar` with `dim_groups=k`: (`b x k x n x 1 x (d//k)` and `b x k x 1 x m x (d//k)`)
            * `diag`: (`b x n x d` and `b x n x d`)
            * `diag` with `dim_groups=k`: (`b x k x n x (d//k)` and `b x k x n x (d//k)`)
        """
        return x1.unsqueeze(-2), x2.unsqueeze(-3)

    def __call__(self, x1, x2=None, **params):
        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_

        return LazyEvaluatedKernelTensor(self, x1_, x2_, **params)

    def __add__(self, other):
        return AdditiveKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)


class AdditiveKernel(Kernel):
    """
    A Kernel that supports summing over multiple component kernels.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> additive_kernel_matrix = covar_module(x1)
    """

    def __init__(self, *kernels):
        super(AdditiveKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1, x2):
        res = ZeroLazyTensor()
        for kern in self.kernels:
            res = res + kern(x1, x2).evaluate_kernel()
        return res


class ProductKernel(Kernel):
    """
    A Kernel that supports elementwise multiplying multiple component kernels together.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) * RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> kernel_matrix = covar_module(x1) # The RBF Kernel already decomposes multiplicatively, so this is foolish!
    """

    def __init__(self, *kernels):
        super(ProductKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1, x2):
        return prod([k(x1, x2).evaluate_kernel() for k in self.kernels])
