from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod
import torch
from torch.nn import ModuleList
from gpytorch.lazy import LazyVariable, LazyEvaluatedKernelVariable, ZeroLazyVariable
from gpytorch.module import Module
from gpytorch.priors._compatibility import _bounds_to_prior
from gpytorch.utils import prod


class Kernel(Module):
    """
    Kernels in GPyTorch are implemented as a :class:`gpytorch.Module` that, when called on two :obj:`torch.tensor`
    objects `x1` and `x2` returns either a :obj:`torch.tensor` or a :obj:`gpytorch.lazy.LazyVariable` that represents
    the covariance matrix between `x1` and `x2`.

    In the typical use case, to extend this class means to implement the :func:`~gpytorch.kernels.Kernel.forward`
    method.

    .. note::
        The :func:`~gpytorch.kernels.Kernel.__call__` does some additional internal work. In particular,
        all kernels are lazily evaluated so that, in some cases, we can index in to the kernel matrix before actually
        computing it. Furthermore, many built in kernel modules return LazyVariables that allow for more efficient
        inference than if we explicitly computed the kernel matrix itselfself.

        As a result, if you want to use a :obj:`gpytorch.kernels.Kernel` object just to get an actual
        :obj:`torch.tensor` representing the covariance matrix, you may need to call the
        :func:`gpytorch.lazy.LazyVariable.evaluate` method on the output.

    Example:
        >>> covar_module = gpytorch.kernels.LinearKernel()
        >>> x1 = torch.randn(50, 3)
        >>> lazy_covar_matrix = covar_module(x1) # Returns a RootLazyVariable
        >>> tensor_covar_matrix = lazy_covar_matrix.evaluate() # Gets the actual tensor for this kernel matrix
    """

    def __init__(
        self,
        has_lengthscale=False,
        ard_num_dims=None,
        log_lengthscale_prior=None,
        active_dims=None,
        batch_size=1,
        log_lengthscale_bounds=None,
    ):
        """
        The base Kernel class handles both lengthscales and ARD.

        Args:
            has_lengthscale (bool): If True, we will register a :obj:`torch.nn.Parameter` named `log_lengthscale`
            ard_num_dims (int): If not None, the `log_lengthscale` parameter will have this many entries.
            log_lengthscale_prior (:obj:`gpytorch.priors.Prior`): Prior over the log lengthscale
            active_dims (list): List of data dimensions to evaluate this Kernel on.
            batch_size (int): If training or testing multiple GPs simultaneously, this is how many lengthscales to
                              register.
            log_lengthscale_bounds (tuple): Deprecated min and max values for the lengthscales. If supplied, this
                                            now registers a :obj:`gpytorch.priors.SmoothedBoxPrior`
        """
        super(Kernel, self).__init__()
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.active_dims = active_dims
        self.ard_num_dims = ard_num_dims
        self.has_lengthscale = has_lengthscale
        if has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            log_lengthscale_prior = _bounds_to_prior(prior=log_lengthscale_prior, bounds=log_lengthscale_bounds)
            self.register_parameter(
                name="log_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(batch_size, 1, lengthscale_num_dims)),
                prior=log_lengthscale_prior,
            )

    @property
    def lengthscale(self):
        if "log_lengthscale" in self.named_parameters().keys():
            return self.log_lengthscale.exp()
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
        if isinstance(res, LazyVariable):
            res = res.evaluate()
        res = res.transpose(-2, -3)
        return res

    @abstractmethod
    def forward(self, x1, x2, **params):
        raise NotImplementedError()

    def __call__(self, x1_, x2_=None, **params):
        x1, x2 = x1_, x2_

        if self.active_dims is not None:
            x1 = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2 = x2_.index_select(-1, self.active_dims)

        if x2 is None:
            x2 = x1

        # Give x1 and x2 a last dimension, if necessary
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)
        if x2.ndimension() == 1:
            x2 = x2.unsqueeze(1)
        if not x1.size(-1) == x2.size(-1):
            raise RuntimeError("x1 and x2 must have the same number of dimensions!")

        return LazyEvaluatedKernelVariable(self, x1, x2)

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
        res = ZeroLazyVariable()
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
