#!/usr/bin/env python3

from abc import abstractmethod
import torch
import warnings
from torch.nn import ModuleList
from ..lazy import lazify, delazify, LazyEvaluatedKernelTensor, ZeroLazyTensor
from ..module import Module
from .. import settings
from ..utils.deprecation import _deprecate_kwarg_with_transform
from ..utils import broadcasting
from ..constraints import Positive


def default_postprocess_script(x):
    return x


class Distance(torch.jit.ScriptModule):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    @torch.jit.script_method
    def _jit_sq_dist_x1_neq_x2(self, x1, x2, postprocess):
        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

        res = x1.matmul(x2.transpose(-2, -1)).mul_(-2).add_(x2_norm.transpose(-2, -1)).add_(x1_norm)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_sq_dist_x1_neq_x2_nobatch(self, x1, x2, postprocess):
        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)

        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_sq_dist_x1_eq_x2(self, x1, postprocess):
        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        res = x1.matmul(x1.transpose(-2, -1)).mul_(-2).add_(x1_norm.transpose(-2, -1)).add_(x1_norm)
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_sq_dist_x1_eq_x2_nobatch(self, x1, postprocess):
        # Compute squared distance matrix using quadratic expansion
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x1_norm.transpose(-2, -1), x1, x1.transpose(-2, -1), alpha=-2).add_(x1_norm)
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values
        res.clamp_min_(0)
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_dist_x1_neq_x2(self, x1, x2, postprocess, false_tensor):
        # Need to do a hack to get a False tensor because pytorch stable doesn't
        # support creating of tensors in torch scripts
        res = self._jit_sq_dist_x1_neq_x2(x1, x2, postprocess=false_tensor)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_dist_x1_neq_x2_nobatch(self, x1, x2, postprocess, false_tensor):
        # Need to do a hack to get a False tensor because pytorch stable doesn't
        # support creating of tensors in torch scripts
        res = self._jit_sq_dist_x1_neq_x2_nobatch(x1, x2, postprocess=false_tensor)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_dist_x1_eq_x2(self, x1, postprocess, false_tensor):
        # Need to do a hack to get a False tensor because pytorch stable doesn't
        # support creating of tensors in torch scripts
        res = self._jit_sq_dist_x1_eq_x2(x1, postprocess=false_tensor)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if bool(postprocess) else res

    @torch.jit.script_method
    def _jit_dist_x1_eq_x2_nobatch(self, x1, postprocess, false_tensor):
        # Need to do a hack to get a False tensor because pytorch stable doesn't
        # support creating of tensors in torch scripts
        res = self._jit_sq_dist_x1_eq_x2_nobatch(x1, postprocess=false_tensor)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if bool(postprocess) else res


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
    batch of data can have its own lengthscale parameter by setting the `batch_shape`
    keyword argument to the appropriate number of batches.

    .. note::

        The :attr:`lengthscale` parameter is parameterized on a log scale to constrain it to be positive.
        You can set a prior on this parameter using the :attr:`lengthscale_prior` argument.

    Base Args:
        :attr:`has_lengthscale` (bool):
            Set this if the kernel has a lengthscale. Default: `False`.
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

    def _batch_shape_state_dict_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if not len(self.batch_shape):
            try:
                current_state_dict = self.state_dict()
                for name, param in current_state_dict.items():
                    load_param = state_dict[prefix + name]
                    if load_param.dim() == param.dim() + 1:
                        warnings.warn(
                            f"The supplied state_dict contains a parameter ({prefix + name}) with an extra batch "
                            f"dimension ({load_param.shape} vs {param.shape}).\nDefault batch shapes are now "
                            "deprecated in GPyTorch. You may wish to re-save your model.", DeprecationWarning
                        )
                        load_param.squeeze_(0)
            except Exception:
                pass

    def __init__(
        self,
        has_lengthscale=False,
        ard_num_dims=None,
        batch_shape=torch.Size([]),
        active_dims=None,
        lengthscale_prior=None,
        lengthscale_constraint=None,
        eps=1e-6,
        **kwargs
    ):
        super(Kernel, self).__init__()
        self._register_load_state_dict_pre_hook(self._batch_shape_state_dict_hook)

        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        self.batch_shape = _deprecate_kwarg_with_transform(
            kwargs, "batch_size", "batch_shape", batch_shape, lambda n: torch.Size([n])
        )

        self.__has_lengthscale = has_lengthscale
        self.eps = eps

        param_transform = kwargs.get("param_transform")

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if param_transform is not None:
            warnings.warn("The 'param_transform' argument is now deprecated. If you want to use a different "
                          "transformation, specify a different 'lengthscale_constraint' instead.")

        if has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims))
            )
            if lengthscale_prior is not None:
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda: self.lengthscale, lambda v: self._set_lengthscale(v)
                )

            self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    @property
    def dtype(self):
        if self.has_lengthscale:
            return self.lengthscale.dtype
        else:
            for param in self.parameters():
                return param.dtype
            return torch.get_default_dtype()

    @property
    def has_lengthscale(self):
        return self.__has_lengthscale

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
            value = torch.tensor(value)

        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def size(self, x1, x2):
        expected_size = broadcasting._matmul_broadcast_shape(
            torch.Size([*x1.shape[:-2], x1.size(-2), x1.size(-1)]),
            torch.Size([*x2.shape[:-2], x2.size(-1), x2.size(-2)]),
            error_msg="x1 and x2 were not broadcastable to a proper kernel shape."
            "Got x1.shape = {} and x2.shape = {}".format(str(x1.shape), str(x2.shape))
        )
        expected_size = broadcasting._mul_broadcast_shape(
            expected_size[:-2], self.batch_shape,
            error_msg=(
                f"x1 and x2 were not broadcastable with kernel of batch_shape {self.batch_shape}."
                f"Got x1.shape = {x1.shape} and x2.shape = {x2.shape}"
            )
        ) + expected_size[-2:]
        return expected_size

    @abstractmethod
    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        """
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        Args:
            - :attr:`x1` (Tensor `n x d` or `b x n x d`)
            - :attr:`x2` (Tensor `m x d` or `b x m x d`)
            - :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?
            - :attr:`batch_dims` (tuple, optional):
                If this option is passed in, it will tell the tensor which of the
                three dimensions are batch dimensions.
                Currently accepts: standard mode (either None or (0,))
                or (0, 2) for use with Additive/Multiplicative kernels

        Returns:
            - :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `full_covar` with `batch_dims=(0, 2)`: `k x n x m` or `b x k x n x m`
                * `diag`: `n` or `b x n`
                * `diag` with `batch_dims=(0, 2)`: `k x n` or `b x k x n`
        """
        raise NotImplementedError()

    def __getstate__(self):
        # JIT ScriptModules cannot be pickled
        self.distance_module = None
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def _covar_dist(
        self,
        x1,
        x2,
        diag=False,
        batch_dims=None,
        square_dist=False,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params
    ):
        """
        This is a helper method for computing the Euclidean distance between
        all pairs of points in x1 and x2.
        The dimensionality of the output depends on the
        Args:
            - :attr:`x1` (Tensor `n x d` or `b x n x d`)
            - :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs
            - :attr:`diag` (bool):
                Should we return the whole distance matrix, or just the diagonal?
            - :attr:`batch_dims` (tuple, optional):
                If this option is passed in, it will tell the tensor which of the
                three dimensions are batch dimensions.
                Currently accepts: standard mode (either None or (0,))
                or (0, 2) for use with Additive/Multiplicative kernels
            - :attr:`square_dist` (bool):
                Should we square the distance matrix before returning?
        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False` and `batch_dims=None`: (`b x n x n`)
            * `diag=False` and `batch_dims=(0, 2)`: (`bd x n x n`)
            * `diag=True` and `batch_dims=None`: (`b x n`)
            * `diag=True` and `batch_dims=(0, 2)`: (`bd x n`)
        """
        if batch_dims == (0, 2):
            x1 = x1.unsqueeze(0).transpose(0, -1)
            x2 = x2.unsqueeze(0).transpose(0, -1)

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
            if x1_eq_x2:
                if x1.dim() == 2:
                    res = self.distance_module._jit_sq_dist_x1_eq_x2_nobatch(x1, postprocess)
                else:
                    res = self.distance_module._jit_sq_dist_x1_eq_x2(x1, postprocess)
            else:
                if x1.dim() == 2:
                    res = self.distance_module._jit_sq_dist_x1_neq_x2_nobatch(x1, x2, postprocess)
                else:
                    res = self.distance_module._jit_sq_dist_x1_neq_x2(x1, x2, postprocess)
        else:
            if x1_eq_x2:
                if x1.dim() == 2:
                    res = self.distance_module._jit_dist_x1_eq_x2_nobatch(x1, postprocess, torch.tensor(False))
                else:
                    res = self.distance_module._jit_dist_x1_eq_x2(x1, postprocess, torch.tensor(False))
            else:
                if x1.dim() == 2:
                    res = self.distance_module._jit_dist_x1_neq_x2_nobatch(x1, x2, postprocess, torch.tensor(False))
                else:
                    res = self.distance_module._jit_dist_x1_neq_x2(x1, x2, postprocess, torch.tensor(False))

        return res

    def __call__(self, x1, x2=None, diag=False, batch_dims=None, **params):
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

        # Check batch_dims args
        # Check that ard_num_dims matches the supplied number of dimensions
        if settings.debug.on():
            if batch_dims is not None:
                if not isinstance(batch_dims, tuple) or (batch_dims != (0,) and batch_dims != (0, 2)):
                    raise RuntimeError(
                        "batch_dims currently accepts either None, (0,), or (0, 2). Got {}.".format(batch_dims)
                    )

            if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
                raise RuntimeError(
                    "Expected the input to have {} dimensionality "
                    "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.size(-1))
                )

        if diag:
            res = super(Kernel, self).__call__(x1_, x2_, diag=True, batch_dims=batch_dims, **params)

            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diag()

            # Now we'll make sure that the shape we're getting from diag makes sense
            if settings.debug.on():
                # If we used batch_dims...
                shape = self.size(x1_, x2_)
                if batch_dims == (0, 2):
                    expected_shape = torch.Size((x1.size(-1),) + shape[:-1])
                    if res.shape != expected_shape:
                        raise RuntimeError(
                            "The kernel {} is not equipped to handle batch_dims=(0, 2) "
                            "and diag. Expected size {}. Got size {}.".format(
                                self.__class__.__name__, expected_shape, res.shape
                            )
                        )

                # If we didn't use batch_dims...
                else:
                    expected_shape = shape[:-1]
                    if res.shape != expected_shape:
                        raise RuntimeError(
                            "The kernel {} is not equipped to handle and diag. Expected size {}. "
                            "Got size {}".format(self.__class__.__name__, expected_shape, res.shape)
                        )
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, batch_dims=batch_dims, **params)
            else:
                res = super(Kernel, self).__call__(x1_, x2_, batch_dims=batch_dims, **params)

            # Now we'll make sure that the shape we're getting makes sense
            if settings.debug.on():
                # If we used batch_dims...
                shape = self.size(x1_, x2_)
                if batch_dims == (0, 2):
                    expected_shape = torch.Size((x1_.size(-1),) + shape)
                    if res.shape != expected_shape:
                        raise RuntimeError(
                            "The kernel {} is not equipped to handle batch_dims=(0, 2). Expected size {}. "
                            "Got size {}".format(self.__class__.__name__, expected_shape, res.shape)
                        )

                # If we didn't use batch_dims...
                else:
                    expected_shape = shape
                    if res.shape != expected_shape:
                        raise RuntimeError(
                            "Error with {}.forward. Expected size {}. Got size {}.".format(
                                self.__class__.__name__, expected_shape, res.shape
                            )
                        )
            return res

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

    def forward(self, x1, x2, **params):
        res = ZeroLazyTensor()
        for kern in self.kernels:
            next_term = kern(x1, x2, **params)
            res = res + lazify(next_term)
        return res

    def size(self, x1, x2):
        return self.kernels[0].size(x1, x2)


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

    def forward(self, x1, x2, **params):
        x1_eq_x2 = torch.equal(x1, x2)

        if not x1_eq_x2:
            # If x1 != x2, then we can't make a MulLazyTensor because the kernel won't necessarily be square/symmetric
            res = delazify(self.kernels[0](x1, x2, **params))
        else:
            res = lazify(self.kernels[0](x1, x2, **params))

        for kern in self.kernels[1:]:
            next_term = kern(x1, x2, **params)
            if not x1_eq_x2:
                # Again delazify if x1 != x2
                res = res * delazify(next_term)
            else:
                res = res * lazify(next_term)
        return res

    def size(self, x1, x2):
        return self.kernels[0].size(x1, x2)
