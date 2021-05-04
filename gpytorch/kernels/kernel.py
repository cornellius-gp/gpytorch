#!/usr/bin/env python3

import warnings
from abc import abstractmethod
from copy import deepcopy

import torch
from torch.nn import ModuleList

from .. import settings
from ..constraints import Positive
from ..lazy import LazyEvaluatedKernelTensor, ZeroLazyTensor, delazify, lazify
from ..models import exact_prediction_strategies
from ..module import Module
from ..utils.broadcasting import _mul_broadcast_shape


def default_postprocess_script(x):
    return x


class Distance(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            x2_norm, x2_pad = x1_norm, x1_pad
        else:
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
        res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if postprocess else res


class Kernel(Module):
    r"""
    Kernels in GPyTorch are implemented as a :class:`gpytorch.Module` that, when called on two :obj:`torch.tensor`
    objects `x1` and `x2` returns either a :obj:`torch.tensor` or a :obj:`gpytorch.lazy.LazyTensor` that represents
    the covariance matrix between `x1` and `x2`.

    In the typical use case, to extend this class means to implement the :func:`~gpytorch.kernels.Kernel.forward`
    method.

    .. note::
        The :func:`~gpytorch.kernels.Kernel.__call__` does some additional internal work. In particular,
        all kernels are lazily evaluated so that, in some cases, we can index in to the kernel matrix before actually
        computing it. Furthermore, many built in kernel modules return LazyTensors that allow for more efficient
        inference than if we explicitly computed the kernel matrix itself.

        As a result, if you want to use a :obj:`gpytorch.kernels.Kernel` object just to get an actual
        :obj:`torch.tensor` representing the covariance matrix, you may need to call the
        :func:`gpytorch.lazy.LazyTensor.evaluate` method on the output.

    This base :class:`Kernel` class includes a lengthscale parameter
    :math:`\Theta`, which is used by many common kernel functions.
    There are a few options for the lengthscale:

    * Default: No lengthscale (i.e. :math:`\Theta` is the identity matrix).

    * Single lengthscale: One lengthscale can be applied to all input dimensions/batches
      (i.e. :math:`\Theta` is a constant diagonal matrix).
      This is controlled by setting the attribute `has_lengthscale=True`.

    * ARD: Each input dimension gets its own separate lengthscale
      (i.e. :math:`\Theta` is a non-constant diagonal matrix).
      This is controlled by the `ard_num_dims` keyword argument (as well as `has_lengthscale=True`).

    In batch-mode (i.e. when :math:`x_1` and :math:`x_2` are batches of input matrices), each
    batch of data can have its own lengthscale parameter by setting the `batch_shape`
    keyword argument to the appropriate number of batches.

    .. note::

        The :attr:`lengthscale` parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the :attr:`lengthscale_prior` argument.

    Base Args:
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each input
            dimension. It should be `d` if :attr:`x1` is a `n x d` matrix.  Default: `None`
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each batch of input
            data. It should be `b1 x ... x bk` if :attr:`x1` is a `b1 x ... x bk x n x d` tensor.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.

    Base Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.

    Example:
        >>> covar_module = gpytorch.kernels.LinearKernel()
        >>> x1 = torch.randn(50, 3)
        >>> lazy_covar_matrix = covar_module(x1) # Returns a RootLazyTensor
        >>> tensor_covar_matrix = lazy_covar_matrix.evaluate() # Gets the actual tensor for this kernel matrix
    """

    has_lengthscale = False

    def __init__(
        self,
        ard_num_dims=None,
        batch_shape=torch.Size([]),
        active_dims=None,
        lengthscale_prior=None,
        lengthscale_constraint=None,
        eps=1e-6,
        **kwargs,
    ):
        super(Kernel, self).__init__()
        self._batch_shape = batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.eps = eps

        param_transform = kwargs.get("param_transform")

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformation, specify a different 'lengthscale_constraint' instead.",
                DeprecationWarning,
            )

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is not None:
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )

            self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    @abstractmethod
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        r"""
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?
            :attr:`last_dim_is_batch` (tuple, optional):
                If this is true, it treats the last dimension of the data as another batch dimension.
                (Useful for additive structure over the dimensions). Default: False

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                * `diag`: `n` or `b x n`
                * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
        """
        raise NotImplementedError()

    @property
    def batch_shape(self):
        kernels = list(self.sub_kernels())
        if len(kernels):
            return _mul_broadcast_shape(self._batch_shape, *[k.batch_shape for k in kernels])
        else:
            return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, val):
        self._batch_shape = val

    @property
    def dtype(self):
        if self.has_lengthscale:
            return self.lengthscale.dtype
        else:
            for param in self.parameters():
                return param.dtype
            return torch.get_default_dtype()

    @property
    def is_stationary(self) -> bool:
        """
        Property to indicate whether kernel is stationary or not.
        """
        return self.has_lengthscale

    @property
    def lengthscale(self):
        if self.has_lengthscale:
            return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
        else:
            return None

    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value):
        if not self.has_lengthscale:
            raise RuntimeError("Kernel has no lengthscale.")

        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)

        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def local_load_samples(self, samples_dict, memo, prefix):
        num_samples = next(iter(samples_dict.values())).size(0)
        self.batch_shape = torch.Size([num_samples]) + self.batch_shape
        super().local_load_samples(samples_dict, memo, prefix)

    def covar_dist(
        self,
        x1,
        x2,
        diag=False,
        last_dim_is_batch=False,
        square_dist=False,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params,
    ):
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in x1 and x2.

        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`diag` (bool):
                Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?
            :attr:`square_dist` (bool):
                Should we square the distance matrix before returning?

        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if not self.distance_module or self.distance_module._postprocess != dist_postprocess_func:
            self.distance_module = Distance(dist_postprocess_func)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                if postprocess:
                    res = dist_postprocess_func(res)
                return res
            else:
                res = torch.norm(x1 - x2, p=2, dim=-1)
                if square_dist:
                    res = res.pow(2)
            if postprocess:
                res = dist_postprocess_func(res)
            return res

        elif square_dist:
            res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
        else:
            res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2)

        return res

    def named_sub_kernels(self):
        for name, module in self._modules.items():
            if isinstance(module, Kernel):
                yield name, module

    def num_outputs_per_input(self, x1, x2):
        """
        How many outputs are produced per input (default 1)
        if x1 is size `n x d` and x2 is size `m x d`, then the size of the kernel
        will be `(n * num_outputs_per_input) x (m * num_outputs_per_input)`
        Default: 1
        """
        return 1

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return exact_prediction_strategies.DefaultPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )

    def sub_kernels(self):
        for _, kernel in self.named_sub_kernels():
            yield kernel

    def __call__(self, x1, x2=None, diag=False, last_dim_is_batch=False, **params):
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

        # Check that ard_num_dims matches the supplied number of dimensions
        if settings.debug.on():
            if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
                raise RuntimeError(
                    "Expected the input to have {} dimensionality "
                    "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.size(-1))
                )

        if diag:
            res = super(Kernel, self).__call__(x1_, x2_, diag=True, last_dim_is_batch=last_dim_is_batch, **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diag()
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = lazify(super(Kernel, self).__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params))
            return res

    def __getstate__(self):
        # JIT ScriptModules cannot be pickled
        self.distance_module = None
        return self.__dict__

    def __add__(self, other):
        kernels = []
        kernels += self.kernels if isinstance(self, AdditiveKernel) else [self]
        kernels += other.kernels if isinstance(other, AdditiveKernel) else [other]
        return AdditiveKernel(*kernels)

    def __mul__(self, other):
        kernels = []
        kernels += self.kernels if isinstance(self, ProductKernel) else [self]
        kernels += other.kernels if isinstance(other, ProductKernel) else [other]
        return ProductKernel(*kernels)

    def __setstate__(self, d):
        self.__dict__ = d

    def __getitem__(self, index):
        if len(self.batch_shape) == 0:
            return self

        new_kernel = deepcopy(self)
        # Process the index
        index = index if isinstance(index, tuple) else (index,)

        for param_name, param in self._parameters.items():
            new_kernel._parameters[param_name].data = param.__getitem__(index)
            ndim_removed = len(param.shape) - len(new_kernel._parameters[param_name].shape)
            new_batch_shape_len = len(self.batch_shape) - ndim_removed
            new_kernel.batch_shape = new_kernel._parameters[param_name].shape[:new_batch_shape_len]

        for sub_module_name, sub_module in self.named_sub_kernels():
            self._modules[sub_module_name] = sub_module.__getitem__(index)

        return new_kernel


class AdditiveKernel(Kernel):
    """
    A Kernel that supports summing over multiple component kernels.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> additive_kernel_matrix = covar_module(x1)
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all components are stationary.
        """
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels):
        super(AdditiveKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1, x2, diag=False, **params):
        res = ZeroLazyTensor() if not diag else 0
        for kern in self.kernels:
            next_term = kern(x1, x2, diag=diag, **params)
            if not diag:
                res = res + lazify(next_term)
            else:
                res = res + next_term

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)

        return new_kernel


class ProductKernel(Kernel):
    """
    A Kernel that supports elementwise multiplying multiple component kernels together.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) * RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> kernel_matrix = covar_module(x1) # The RBF Kernel already decomposes multiplicatively, so this is foolish!
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all components are stationary.
        """
        return all(k.is_stationary for k in self.kernels)

    def __init__(self, *kernels):
        super(ProductKernel, self).__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1, x2, diag=False, **params):
        x1_eq_x2 = torch.equal(x1, x2)

        if not x1_eq_x2:
            # If x1 != x2, then we can't make a MulLazyTensor because the kernel won't necessarily be square/symmetric
            res = delazify(self.kernels[0](x1, x2, diag=diag, **params))
        else:
            res = self.kernels[0](x1, x2, diag=diag, **params)

            if not diag:
                res = lazify(res)

        for kern in self.kernels[1:]:
            next_term = kern(x1, x2, diag=diag, **params)
            if not x1_eq_x2:
                # Again delazify if x1 != x2
                res = res * delazify(next_term)
            else:
                if not diag:
                    res = res * lazify(next_term)
                else:
                    res = res * next_term

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)

        return new_kernel
