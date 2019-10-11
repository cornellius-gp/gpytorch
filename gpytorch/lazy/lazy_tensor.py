#!/usr/bin/env python3

import math
import warnings
from abc import ABC, abstractmethod

import gpytorch
import torch

from .. import settings
from .. import utils
from ..functions._inv_matmul import InvMatmul
from ..functions._inv_quad import InvQuad
from ..functions._inv_quad_log_det import InvQuadLogDet
from ..functions._matmul import Matmul
from ..functions._root_decomposition import RootDecomposition
from ..utils.broadcasting import _matmul_broadcast_shape, _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.deprecation import _deprecate_renamed_methods
from ..utils.gradients import _ensure_symmetric_grad
from ..utils.getitem import _noop_index, _convert_indices_to_tensors, _compute_getitem_size
from ..utils.memoize import add_to_cache, cached
from .lazy_tensor_representation_tree import LazyTensorRepresentationTree


class LazyTensor(ABC):
    """
    Base class for LazyTensors in GPyTorch.

    In GPyTorch, nearly all covariance matrices for Gaussian processes are handled internally as some variety of
    LazyTensor. A LazyTensor is an object that represents a tensor object, similar to :class:`torch.tensor`, but
    typically differs in two ways:

    #. A tensor represented by a LazyTensor can typically be represented more efficiently than storing a full matrix.
       For example, a LazyTensor representing :math:`K=XX^{\\top}` where :math:`K` is :math:`n \\times n` but
       :math:`X` is :math:`n \\times d` might store :math:`X` instead of :math:`K` directly.
    #. A LazyTensor typically defines a matmul routine that performs :math:`KM` that is more efficient than storing
       the full matrix. Using the above example, performing :math:`KM=X(X^{\\top}M)` requires only :math:`O(nd)` time,
       rather than the :math:`O(n^2)` time required if we were storing :math:`K` directly.

    In order to define a new LazyTensor class that can be used as a covariance matrix in GPyTorch, a user must define
    at a minimum the following methods (in each example, :math:`K` denotes the matrix that the LazyTensor represents)

    * :func:`~gpytorch.lazy.LazyTensor._matmul`, which performs a matrix multiplication :math:`KM`
    * :func:`~gpytorch.lazy.LazyTensor._size`, which returns a :class:`torch.Size` containing the dimensions of
      :math:`K`.
    * :func:`~gpytorch.lazy.LazyTensor._transpose_nonbatch`, which returns a transposed version of the LazyTensor

    In addition to these, the following methods should be implemented for maximum efficiency

    * :func:`~gpytorch.lazy.LazyTensor._quad_form_derivative`, which computes the derivative of a quadratic form
      with the LazyTensor (e.g. :math:`d (a^T X b) / dX`).
    * :func:`~gpytorch.lazy.LazyTensor._get_indices`, which returns a :class:`torch.Tensor` containing elements that
      are given by various tensor indices.
    * :func:`~gpytorch.lazy.LazyTensor._expand_batch`, which expands the batch dimensions of LazyTensors.
    * :func:`~gpytorch.lazy.LazyTensor._check_args`, which performs error checking on the arguments supplied to the
      LazyTensor constructor.

    In addition to these, a LazyTensor *may* need to define the following functions if it does anything interesting
    with the batch dimensions (e.g. sums along them, adds additional ones, etc):
    :func:`~gpytorch.lazy.LazyTensor._unsqueeze_batch`, :func:`~gpytorch.lazy.LazyTensor._getitem`, and
    :func:`~gpytorch.lazy.LazyTensor._permute_batch`.
    See the documentation for these methods for details.

    .. note::
        The base LazyTensor class provides default implementations of many other operations in order to mimic the
        behavior of a standard tensor as closely as possible. For example, we provide default implementations of
        :func:`~gpytorch.lazy.LazyTensor.__getitem__`, :func:`~gpytorch.lazy.LazyTensor.__add__`, etc that either
        make use of other lazy tensors or exploit the functions that **must** be defined above.

        Rather than overriding the public methods, we recommend that you override the private versions associated
        with these methods (e.g. - write a custom `_getitem` verses a custom `__getitem__`). This is because the
        public methods do quite a bit of error checking and casing that doesn't need to be repeated.

    .. note::
        LazyTensors are designed by default to optionally represent batches of matrices. Thus, the size of a
        LazyTensor may be (for example) :math:`b \\times n \\times n`. Many of the methods are designed to efficiently
        operate on these batches if present.
    """
    def _check_args(self, *args, **kwargs):
        """
        (Optional) run checks to see that input arguments and kwargs are valid

        Return:
            None (if all checks pass) or str (error message to raise)
        """
        return None

    def __init__(self, *args, **kwargs):
        if settings.debug.on():
            err = self._check_args(*args, **kwargs)
            if err is not None:
                raise ValueError(err)

        self._args = args
        self._kwargs = kwargs

    ####
    # The following methods need to be defined by the LazyTensor
    ####
    @abstractmethod
    def _matmul(self, rhs):
        """
        Performs a matrix multiplication :math:`KM` with the matrix :math:`K` that this LazyTensor represents. Should
        behave as :func:`torch.matmul`. If the LazyTensor represents a batch of matrices, this method should therefore
        operate in batch mode as well.

        ..note::
            This method is intended to be used only internally by various Functions that support backpropagation
            (e.g., :class:`gpytorch.functions.Matmul`). Once this method is defined, it is strongly recommended that
            one use :func:`~gpytorch.lazy.LazyTensor.matmul` instead, which makes use of this method properly.

        Args:
            rhs (:obj:`torch.tensor`): the matrix :math:`M` to multiply with.

        Returns:
            :obj:`torch.tensor`: matrix * rhs
        """
        raise NotImplementedError("The class {} requires a _matmul function!".format(self.__class__.__name__))

    @abstractmethod
    def _size(self):
        """
        Returns the size of the resulting Tensor that the lazy tensor represents.

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.size`,
            which does some additional work. Calling this method directly is discouraged.

        Returns:
            :obj:`torch.Size`: The size of the matrix :math:`K` represented by this LazyTensor
        """
        raise NotImplementedError("The class {} requires a _size function!".format(self.__class__.__name__))

    @abstractmethod
    def _transpose_nonbatch(self):
        """
        Transposes non-batch dimensions (e.g. last two)
        Implement this method, rather than transpose() or t().

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.transpose`, which
            does some additional work. Calling this method directly is discouraged.
        """
        raise NotImplementedError(
            "The class {} requires a _transpose_nonbatch function!".format(self.__class__.__name__)
        )

    ####
    # The following methods MIGHT have be over-written by LazyTensor subclasses
    # if the LazyTensor does weird things with the batch dimensions
    ####
    def _permute_batch(self, *dims):
        """
        Permute the batch dimensions.
        This probably won't have to be overwritten by LazyTensors, unless they use batch dimensions
        in a special way (e.g. BlockDiagLazyTensor, SumBatchLazyTensor)

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.unsqueeze`,
            which does some additional work. Calling this method directly is discouraged.

        Args:
            dims (tuple of ints):
                The new order for the `self.dim() - 2` dimensions.
                It WILL contain each of the positive batch dimensions exactly once.
        """
        components = []
        for component in self._args:
            if torch.is_tensor(component):
                extra_dims = range(len(dims), component.dim())
                components.append(component.permute(*dims, *extra_dims))
            elif isinstance(component, LazyTensor):
                components.append(component._permute_batch(*dims))
            else:
                components.append(component)

        res = self.__class__(*components, **self._kwargs)
        return res

    def _getitem(self, row_index, col_index, *batch_indices):
        """
        Supports subindexing of the matrix this LazyTensor represents.

        The indices passed into this method will either be:
            Tensor indices
            Slices

        ..note::
            LazyTensor.__getitem__ uses this as a helper method. If you are writing your own custom LazyTensor,
            override this method rather than __getitem__ (so that you don't have to repeat the extra work)

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.__getitem__`,
            which does some additional work. Calling this method directly is discouraged.

        This method has a number of restrictions on the type of arguments that are passed in to reduce
        the complexity of __getitem__ calls in PyTorch. In particular:
            - This method only accepts slices and tensors for the row/column indices (no ints)
            - The row and column dimensions don't dissapear (e.g. from Tensor indexing). These cases are
              handled by the `_getindices` method

        Args:
            :attr:`row_index` (slice, Tensor):
                Index for the row of the LazyTensor
            :attr:`col_index` (slice, Tensor):
                Index for the col of the LazyTensor
            :attr:`batch_indices` (tuple of slice, int, Tensor):
                Indices for the batch dimensions

        Returns:
            `LazyTensor`
        """
        # Special case: if both row and col are not indexed, then we are done
        if (row_index is _noop_index and col_index is _noop_index):
            if len(batch_indices):
                components = [component[batch_indices] for component in self._args]
                res = self.__class__(*components, **self._kwargs)
                return res
            else:
                return self

        # Normal case: we have to do some processing on either the rows or columns
        # We will handle this through "interpolation"
        row_interp_indices = torch.arange(0, self.size(-2), dtype=torch.long, device=self.device).view(-1, 1)
        row_interp_indices = row_interp_indices.expand(*self.batch_shape, -1, 1)
        row_interp_values = torch.tensor(1., dtype=self.dtype, device=self.device).expand_as(row_interp_indices)

        col_interp_indices = torch.arange(0, self.size(-1), dtype=torch.long, device=self.device).view(-1, 1)
        col_interp_indices = col_interp_indices.expand(*self.batch_shape, -1, 1)
        col_interp_values = torch.tensor(1., dtype=self.dtype, device=self.device).expand_as(col_interp_indices)

        # Construct interpolated LazyTensor
        from . import InterpolatedLazyTensor

        res = InterpolatedLazyTensor(self, row_interp_indices, row_interp_values, col_interp_indices, col_interp_values)
        return res._getitem(row_index, col_index, *batch_indices)

    def _unsqueeze_batch(self, dim):
        """
        Unsqueezes a batch dimension (positive-indexed only)
        This probably won't have to be overwritten by LazyTensors, unless they use batch dimensions
        in a special way (e.g. BlockDiagLazyTensor, SumBatchLazyTensor)

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.unsqueeze`,
            which does some additional work. Calling this method directly is discouraged.
        """
        components = [component.unsqueeze(dim) for component in self._args]
        res = self.__class__(*components, **self._kwargs)
        return res

    ####
    # The following methods PROBABLY should be over-written by LazyTensor subclasses for efficiency
    ####
    def _expand_batch(self, batch_shape):
        """
        Expands along batch dimensions.

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.expand`,
            which does some additional work. Calling this method directly is discouraged.
        """
        current_shape = torch.Size([1 for _ in range(len(batch_shape) - self.dim() - 2)] + list(self.batch_shape))
        batch_repeat = torch.Size(
            [expand_size // current_size for expand_size, current_size in zip(batch_shape, current_shape)]
        )
        return self.repeat(*batch_repeat, 1, 1)

    def _get_indices(self, row_index, col_index, *batch_indices):
        """
        This method selects elements from the LazyTensor based on tensor indices for each dimension.
        All indices are tensor indices that are broadcastable.
        There will be exactly one index per dimension of the LazyTensor

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.__getitem__`,
            which does some additional work. Calling this method directly is discouraged.

        Args:
            row_index (LongTensor): indices to select from row of LazyTensor
            row_index (LongTensor): indices to select from col of LazyTensor
            batch_indices (tuple LongTensor): indices to select from batch dimensions.

        Returns:
            Tensor (size determined by broadcasted shape of indices) of selected values
        """
        final_shape = _mul_broadcast_shape(*(index.shape for index in batch_indices), row_index.shape, col_index.shape)
        row_index = row_index.expand(final_shape)
        col_index = col_index.expand(final_shape)
        batch_indices = tuple(index.expand(final_shape) for index in batch_indices)

        base_lazy_tensor = self._getitem(_noop_index, _noop_index, *batch_indices)._expand_batch(final_shape)

        # Create some interoplation indices and values
        row_interp_indices = torch.arange(0, self.size(-2), dtype=torch.long, device=self.device)
        row_interp_indices = row_interp_indices[row_index].unsqueeze_(-1).unsqueeze_(-1)
        row_interp_values = torch.tensor(1., dtype=self.dtype, device=self.device).expand_as(row_interp_indices)

        col_interp_indices = torch.arange(0, self.size(-1), dtype=torch.long, device=self.device)
        col_interp_indices = col_interp_indices[col_index].unsqueeze_(-1).unsqueeze_(-1)
        col_interp_values = torch.tensor(1., dtype=self.dtype, device=self.device).expand_as(col_interp_indices)

        # Construct interpolated LazyTensor
        from . import InterpolatedLazyTensor

        res = InterpolatedLazyTensor(
            base_lazy_tensor, row_interp_indices, row_interp_values, col_interp_indices, col_interp_values
        ).evaluate().squeeze(-2).squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        """
        Given u (left_vecs) and v (right_vecs),
        Computes the derivatives of (u^t K v) w.r.t. K

        ..note::
            This method is intended to be used only internally by various Functions that support backpropagation.
            For example, this method is used internally by :func:`~gpytorch.lazy.LazyTensor.inv_quad_logdet`. It is
            not likely that users will need to call this method directly.

        Returns:
            :obj:`torch.tensor`: derivative with respect to the arguments that are actually used to represent this
                                   this LazyTensor.
        """
        from collections import deque

        args = tuple(self.representation())
        args_with_grads = tuple(arg for arg in args if arg.requires_grad)

        # Easy case: if we don't require any gradients, then just return!
        if not len(args_with_grads):
            return tuple(None for _ in args)

        # Normal case: we'll use the autograd to get us a derivative
        with torch.autograd.enable_grad():
            loss = (left_vecs * self._matmul(right_vecs)).sum()
            loss.requires_grad_(True)
            actual_grads = deque(torch.autograd.grad(loss, args_with_grads, allow_unused=True))

        # Now make sure that the object we return has one entry for every item in args
        grads = []
        for arg in args:
            if arg.requires_grad:
                grads.append(actual_grads.popleft())
            else:
                grads.append(None)

        return tuple(grads)

    ####
    # Class definitions
    ####
    _check_size = True

    ####
    # Standard LazyTensor methods
    ####
    @property
    def _args(self):
        return self._args_memo

    @_args.setter
    def _args(self, args):
        self._args_memo = args

    def _approx_diag(self):
        """
        (Optional) returns an (approximate) diagonal of the matrix

        Sometimes computing an exact diagonal is a bit computationally slow
        When we don't need an exact diagonal (e.g. for the pivoted cholesky
        decomposition, this function is called

        Defaults to calling the exact diagonal function

        Returns:
            tensor: - the diagonal (or batch of diagonals)
        """
        return self.diag()

    @cached(name="cholesky")
    def _cholesky(self):
        """
        (Optional) Cholesky-factorizes the LazyTensor

        ..note::
            This method is used as an internal helper. Calling this method directly is discouraged.

        Returns:
            (LazyTensor) Cholesky factor
        """
        from .non_lazy_tensor import NonLazyTensor
        evaluated_mat = self.evaluate()

        # if the tensor is a scalar, we can just take the square root
        if evaluated_mat.size(-1) == 1:
            return NonLazyTensor(evaluated_mat.clamp_min(0.0).sqrt())

        # NOTE: this hack is in place so that the gradient of the Cholesky factorization is symmetric
        # We can remove this hack once https://github.com/pytorch/pytorch/issues/18825 is merged in
        if evaluated_mat.requires_grad:
            evaluated_mat.register_hook(_ensure_symmetric_grad)

        # contiguous call is necessary here
        cholesky = psd_safe_cholesky(evaluated_mat).contiguous()
        return NonLazyTensor(cholesky)

    def _cholesky_solve(self, rhs):
        """
        (Optional) Assuming that `self` is a Cholesky factor, computes the cholesky solve

        ..note::
            This method is used as an internal helper. Calling this method directly is discouraged.

        Returns:
            (LazyTensor) Cholesky factor
        """
        return torch.cholesky_solve(rhs, self.evaluate())

    def _inv_matmul_preconditioner(self):
        """
        (Optional) define a preconditioner that can be used for linear systems, but not necessarily
        for log determinants. By default, this can call :meth:`~gpytorch.lazy.LazyTensor._preconditioner`.

        Returns:
            function: a function on x which performs P^{-1}(x)
        """
        base_precond, _, _ = self._preconditioner()

        if base_precond is not None:
            return base_precond
        elif gpytorch.beta_features.default_preconditioner.on():
            if hasattr(self, "_default_preconditioner_cache"):
                U, S, V = self._default_preconditioner_cache
            else:
                precond_basis_size = min(gpytorch.settings.max_preconditioner_size.value(), self.size(-1))
                random_basis = torch.randn(
                    self.batch_shape + torch.Size((self.size(-2), precond_basis_size)),
                    device=self.device,
                    dtype=self.dtype,
                )
                projected_mat = self._matmul(random_basis)
                proj_q = torch.qr(projected_mat)
                orthog_projected_mat = self._matmul(proj_q).transpose(-2, -1)
                U, S, V = torch.svd(orthog_projected_mat)
                U = proj_q.matmul(U)

                self._default_preconditioner_cache = (U, S, V)

            def preconditioner(v):
                res = V.transpose(-2, -1).matmul(v)
                res = (1 / S).unsqueeze(-1) * res
                res = U.matmul(res)
                return res

            return preconditioner
        else:
            return None

    def _mul_constant(self, other):
        """
        Multiplies the LazyTensor by a costant.

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.mul`,
            which does some additional work. Calling this method directly is discouraged.

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`
        """
        from .constant_mul_lazy_tensor import ConstantMulLazyTensor
        return ConstantMulLazyTensor(self, other)

    def _mul_matrix(self, other):
        """
        Multiplies the LazyTensor by a (batch of) matrices.

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.mul`,
            which does some additional work. Calling this method directly is discouraged.

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`
        """
        from .non_lazy_tensor import NonLazyTensor
        from .mul_lazy_tensor import MulLazyTensor

        self = self.evaluate_kernel()
        other = other.evaluate_kernel()
        if isinstance(self, NonLazyTensor) or isinstance(other, NonLazyTensor):
            return NonLazyTensor(self.evaluate() * other.evaluate())
        else:
            left_lazy_tensor = self if self._root_decomposition_size() < other._root_decomposition_size() else other
            right_lazy_tensor = other if left_lazy_tensor is self else self
            return MulLazyTensor(left_lazy_tensor.root_decomposition(), right_lazy_tensor.root_decomposition())

    def _preconditioner(self):
        """
        (Optional) define a preconditioner (P) for linear conjugate gradients

        Returns:
            function: a function on x which performs P^{-1}(x)
            scalar: the log determinant of P
        """
        return None, None, None

    def _probe_vectors_and_norms(self):
        return None, None

    def _prod_batch(self, dim):
        """
        Multiply the LazyTensor across a batch dimension (supplied as a positive number).

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.prod`,
            which does some additional work. Calling this method directly is discouraged.

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`
        """
        from .mul_lazy_tensor import MulLazyTensor
        from .root_lazy_tensor import RootLazyTensor

        if self.size(dim) == 1:
            return self.squeeze(dim)

        roots = self.root_decomposition().root.evaluate()
        num_batch = roots.size(dim)

        while True:
            # Take care of extra roots (odd roots), if they exist
            if num_batch % 2:
                shape = list(roots.shape)
                shape[dim] = 1
                extra_root = torch.full(
                    shape, dtype=self.dtype, device=self.device, fill_value=(1.0 / math.sqrt(self.size(-2)))
                )
                roots = torch.cat([roots, extra_root], dim)
                num_batch += 1

            # Divide and conqour
            # Assumes that there's an even number of roots
            part1_index = [_noop_index] * roots.dim()
            part1_index[dim] = slice(None, num_batch // 2, None)
            part1 = roots[tuple(part1_index)].contiguous()
            part2_index = [_noop_index] * roots.dim()
            part2_index[dim] = slice(num_batch // 2, None, None)
            part2 = roots[tuple(part2_index)].contiguous()

            if num_batch // 2 == 1:
                part1 = part1.squeeze(dim)
                part2 = part2.squeeze(dim)
                res = MulLazyTensor(RootLazyTensor(part1), RootLazyTensor(part2))
                break
            else:
                res = MulLazyTensor(RootLazyTensor(part1), RootLazyTensor(part2))
                roots = res.root_decomposition().root.evaluate()
                num_batch = num_batch // 2

        return res

    def _root_decomposition(self):
        """
        Returns the (usually low-rank) root of a lazy tensor of a PSD matrix.

        ..note::
            This method is used internally by the related function
            :func:`~gpytorch.lazy.LazyTensor.root_decomposition`, which does some additional work.
            Calling this method directly is discouraged.

        Returns:
            (Tensor or LazyTensor): The root of the root decomposition
        """
        func = RootDecomposition()
        res, _ = func.apply(
            self.representation_tree(),
            self._root_decomposition_size(),
            self.dtype,
            self.device,
            self.batch_shape,
            self.matrix_shape,
            True,
            False,
            None,
            *self.representation()
        )

        return res

    def _root_decomposition_size(self):
        """
        This is the inner size of the root decomposition.
        This is primarily used to determine if it will be cheaper to compute a
        different root or not
        """
        return settings.max_root_decomposition_size.value()

    def _root_inv_decomposition(self, initial_vectors=None):
        """
        Returns the (usually low-rank) inverse root of a lazy tensor of a PSD matrix.

        ..note::
            This method is used internally by the related function
            :func:`~gpytorch.lazy.LazyTensor.root_inv_decomposition`, which does some additional work.
            Calling this method directly is discouraged.

        Returns:
            (Tensor or LazyTensor): The root of the inverse root decomposition
        """
        from .root_lazy_tensor import RootLazyTensor

        func = RootDecomposition()
        roots, inv_roots = func.apply(
            self.representation_tree(),
            self._root_decomposition_size(),
            self.dtype,
            self.device,
            self.batch_shape,
            self.matrix_shape,
            True,
            True,
            initial_vectors,
            *self.representation()
        )

        if initial_vectors is not None and initial_vectors.size(-1) > 1:
            add_to_cache(self, "root_decomposition", RootLazyTensor(roots[0]))
        else:
            add_to_cache(self, "root_decomposition", RootLazyTensor(roots))

        return inv_roots

    def _solve(self, rhs, preconditioner, num_tridiag=0):
        return utils.linear_cg(
            self._matmul,
            rhs,
            n_tridiag=num_tridiag,
            max_iter=settings.max_cg_iterations.value(),
            max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=preconditioner,
        )

    def _sum_batch(self, dim):
        """
        Sum the LazyTensor across a batch dimension (supplied as a positive number).

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.sum`,
            which does some additional work. Calling this method directly is discouraged.

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`
        """
        from .sum_batch_lazy_tensor import SumBatchLazyTensor
        return SumBatchLazyTensor(self, block_dim=dim)

    def _t_matmul(self, rhs):
        """
        Performs a transpose matrix multiplication :math:`K^{\\top}M` with the matrix :math:`K` that this
        LazyTensor represents.

        Args:
            rhs (:obj:`torch.tensor`): the matrix :math:`M` to multiply with.

        Returns:
            :obj:`torch.tensor`: matrix * rhs
        """
        return self.transpose(-1, -2)._matmul(rhs)

    def add_diag(self, diag):
        """
        Adds an element to the diagonal of the matrix.

        Args:
            - diag (Scalar Tensor)
        """
        from .diag_lazy_tensor import DiagLazyTensor
        from .added_diag_lazy_tensor import AddedDiagLazyTensor

        if not self.is_square:
            raise RuntimeError("add_diag only defined for square matrices")

        try:
            expanded_diag = diag.expand(self.shape[:-1])
        except RuntimeError:
            raise RuntimeError(
                "add_diag for LazyTensor of size {} received invalid diagonal of size {}.".format(
                    self.shape, diag.shape
                )
            )

        return AddedDiagLazyTensor(self, DiagLazyTensor(expanded_diag))

    def add_jitter(self, jitter_val=1e-3):
        """
        Adds jitter (i.e., a small diagonal component) to the matrix this
        LazyTensor represents. This could potentially be implemented as a no-op,
        however this could lead to numerical instabilities, so this should only
        be done at the user's risk.
        """
        diag = torch.tensor(jitter_val, dtype=self.dtype, device=self.device)
        return self.add_diag(diag)

    @property
    def batch_dim(self):
        """
        Returns the dimension of the shape over which the tensor is batched.
        """
        return len(self.batch_shape)

    @property
    def batch_shape(self):
        """
        Returns the shape over which the tensor is batched.
        """
        return self.shape[:-2]

    def cholesky(self, upper=False):
        """
        Cholesky-factorizes the LazyTensor

        Parameters:
            upper (bool) - upper triangular or lower triangular factor (default: False)

        Returns:
            (LazyTensor) Cholesky factor (lower triangular)
        """
        res = self._cholesky()
        if upper:
            res = res.transpose(-1, -2)
        return res

    def clone(self):
        """
        Clones the LazyTensor (creates clones of all underlying tensors)
        """
        args = [arg.clone() if hasattr(arg, "clone") else arg for arg in self._args]
        kwargs = {key: val.clone() if hasattr(val, "clone") else val for key, val in self._kwargs.items()}
        return self.__class__(*args, **kwargs)

    def cpu(self):
        """
        Returns:
            :obj:`~gpytorch.lazy.LazyTensor`: a new LazyTensor identical to ``self``, but on the CPU.
        """
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "cpu"):
                new_args.append(arg.cpu())
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "cpu"):
                new_kwargs[name] = val.cpu()
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    def cuda(self, device_id=None):
        """
        This method operates identically to :func:`torch.nn.Module.cuda`.

        Args:
            device_id (:obj:`str`, optional):
                Device ID of GPU to use.
        Returns:
            :obj:`~gpytorch.lazy.LazyTensor`:
                a new LazyTensor identical to ``self``, but on the GPU.
        """
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "cuda"):
                new_args.append(arg.cuda(device_id))
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "cuda"):
                new_kwargs[name] = val.cuda(device_id)
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    @property
    def device(self):
        return self._args[0].device

    def detach(self):
        """
        Removes the LazyTensor from the current computation graph.
        (In practice, this function removes all Tensors that make up the
        LazyTensor from the computation graph.)
        """
        return self.clone().detach_()

    def detach_(self):
        """
        An in-place version of `detach`.
        """
        for arg in self._args:
            if hasattr(arg, "detach"):
                arg.detach_()
        for val in self._kwargs.values():
            if hasattr(val, "detach"):
                val.detach_()
        return self

    def diag(self):
        """
        As :func:`torch.diag`, returns the diagonal of the matrix :math:`K` this LazyTensor represents as a vector.

        Returns:
            :obj:`torch.tensor`: The diagonal of :math:`K`. If :math:`K` is :math:`n \times n`, this will be a length
            n vector. If this LazyTensor represents a batch (e.g., is :math:`b \times n \times n`), this will be a
            :math:`b \times n` matrix of diagonals, one for each matrix in the batch.
        """
        if settings.debug.on():
            if not self.is_square:
                raise RuntimeError("Diag works on square matrices (or batches)")

        row_col_iter = torch.arange(0, self.matrix_shape[-1], dtype=torch.long, device=self.device)
        return self[..., row_col_iter, row_col_iter]

    def dim(self):
        """
        Alias of :meth:`~gpytorch.lazy.LazyTensor.ndimension`
        """
        return self.ndimension()

    @property
    def dtype(self):
        return self._args[0].dtype

    def expand(self, *sizes):
        if len(sizes) == 1 and hasattr(sizes, "__iter__"):
            sizes = sizes[0]
        if len(sizes) < 2 or tuple(sizes[-2:]) != self.matrix_shape:
            raise RuntimeError(
                "Invalid expand arguments {}. Currently, repeat only works to create repeated "
                "batches of a 2D LazyTensor.".format(tuple(sizes))
            )
        elif all(isinstance(size, int) for size in sizes):
            shape = torch.Size(sizes)
        else:
            raise RuntimeError("Invalid arguments {} to expand.".format(sizes))

        res = self._expand_batch(batch_shape=shape[:-2])
        return res

    @cached
    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyTensor represents. This function
        should return a Tensor storing an exact representation of this LazyTensor.
        """
        num_rows, num_cols = self.matrix_shape

        if num_rows < num_cols:
            eye = torch.eye(num_rows, dtype=self.dtype, device=self.device)
            eye = eye.expand(*self.batch_shape, num_rows, num_rows)
            res = self.transpose(-1, -2).matmul(eye).transpose(-1, -2).contiguous()
        else:
            eye = torch.eye(num_cols, dtype=self.dtype, device=self.device)
            eye = eye.expand(*self.batch_shape, num_cols, num_cols)
            res = self.matmul(eye)
        return res

    def evaluate_kernel(self):
        """
        Return a new LazyTensor representing the same one as this one, but with
        all lazily evaluated kernels actually evaluated.
        """
        return self.representation_tree()(*self.representation())

    def inv_matmul(self, right_tensor, left_tensor=None):
        """
        Computes a linear solve (w.r.t self = :math:`A`) with several right hand sides :math:`R`.
        I.e. computes

        ... math::

            \begin{equation}
                A^{-1} R,
            \end{equation}

        where :math:`R` is :attr:`right_tensor` and :math:`A` is the LazyTensor.

        If :attr:`left_tensor` is supplied, computes

        ... math::

            \begin{equation}
                L A^{-1} R,
            \end{equation}

        where :math:`L` is :attr:`left_tensor`. Supplying this can reduce the number of
        CG calls required.

        Args:
            - :obj:`torch.tensor` (n x k) - Matrix :math:`R` right hand sides
            - :obj:`torch.tensor` (m x n) - Optional matrix :math:`L` to perform left multiplication with

        Returns:
            - :obj:`torch.tensor` - :math:`A^{-1}R` or :math:`LA^{-1}R`.
        """
        if not self.is_square:
            raise RuntimeError(
                "inv_matmul only operates on (batches of) square (positive semi-definite) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if self.dim() == 2 and right_tensor.dim() == 1:
            if self.shape[-1] != right_tensor.numel():
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, right_tensor.shape
                    )
                )

        func = InvMatmul
        if left_tensor is None:
            return func.apply(
                self.representation_tree(),
                False,
                right_tensor,
                *self.representation(),
            )
        else:
            return func.apply(
                self.representation_tree(),
                True,
                left_tensor,
                right_tensor,
                *self.representation(),
            )

    def inv_quad(self, tensor, reduce_inv_quad=True):
        """
        Computes an inverse quadratic form (w.r.t self) with several right hand sides.
        I.e. computes tr( tensor^T self^{-1} tensor )

        NOTE: Don't overwrite this function!
        Instead, overwrite inv_quad_logdet

        Args:
            - tensor (tensor nxk) - Vector (or matrix) for inverse quad

        Returns:
            - tensor - tr( tensor^T (self)^{-1} tensor )
        """
        if not self.is_square:
            raise RuntimeError(
                "inv_quad only operates on (batches of) square (positive semi-definite) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if self.dim() == 2 and tensor.dim() == 1:
            if self.shape[-1] != tensor.numel():
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, tensor.shape
                    )
                )
        elif self.dim() != tensor.dim():
            raise RuntimeError(
                "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                "of dimensions.".format(self.shape, tensor.shape)
            )
        elif self.shape[-1] != tensor.shape[-2]:
            raise RuntimeError(
                "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                    self.shape, tensor.shape
                )
            )

        args = (tensor,) + self.representation()
        func = InvQuad.apply
        inv_quad_term = func(self.representation_tree(), *args)

        if reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        """
        Computes an inverse quadratic form (w.r.t self) with several right hand sides.
        I.e. computes tr( tensor^T self^{-1} tensor )
        In addition, computes an (approximate) log determinant of the the matrix

        Args:
            - tensor (tensor nxk) - Vector (or matrix) for inverse quad

        Returns:
            - scalar - tr( tensor^T (self)^{-1} tensor )
            - scalar - log determinant
        """
        # Special case: use Cholesky to compute these terms
        if settings.fast_computations.log_prob.off() or (self.size(-1) <= settings.max_cholesky_size.value()):
            from .chol_lazy_tensor import CholLazyTensor

            cholesky = CholLazyTensor(self.cholesky())
            return cholesky.inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad)

        # Default: use modified batch conjugate gradients to compute these terms
        # See NeurIPS 2018 paper: https://arxiv.org/abs/1809.11165
        if not self.is_square:
            raise RuntimeError(
                "inv_quad_logdet only operates on (batches of) square (positive semi-definite) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if inv_quad_rhs is not None:
            if self.dim() == 2 and inv_quad_rhs.dim() == 1:
                if self.shape[-1] != inv_quad_rhs.numel():
                    raise RuntimeError(
                        "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                            self.shape, inv_quad_rhs.shape
                        )
                    )
            elif self.dim() != inv_quad_rhs.dim():
                raise RuntimeError(
                    "LazyTensor (size={}) and right-hand-side Tensor (size={}) should have the same number "
                    "of dimensions.".format(self.shape, inv_quad_rhs.shape)
                )
            elif self.batch_shape != inv_quad_rhs.shape[:-2] or self.shape[-1] != inv_quad_rhs.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, inv_quad_rhs.shape
                    )
                )

        args = self.representation()
        if inv_quad_rhs is not None:
            args = [inv_quad_rhs] + list(args)

        probe_vectors, probe_vector_norms = self._probe_vectors_and_norms()

        func = InvQuadLogDet.apply

        inv_quad_term, logdet_term = func(
            self.representation_tree(),
            self.dtype,
            self.device,
            self.matrix_shape,
            self.batch_shape,
            (inv_quad_rhs is not None),
            logdet,
            probe_vectors,
            probe_vector_norms,
            *args,
        )

        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term, logdet_term

    @property
    def is_square(self):
        return self.matrix_shape[0] == self.matrix_shape[1]

    def logdet(self):
        """
        Computes an (approximate) log determinant of the matrix

        NOTE: Don't overwrite this function!
        Instead, overwrite inv_quad_logdet

        Returns:
            - scalar: log determinant
        """
        _, res = self.inv_quad_logdet(inv_quad_rhs=None, logdet=True)
        return res

    def matmul(self, other):
        """
        Multiplies self by a matrix

        Args:
            other (:obj:`torch.tensor`): Matrix or vector to multiply with. Can be either a :obj:`torch.tensor`
                or a :obj:`gpytorch.lazy.LazyTensor`.

        Returns:
            :obj:`torch.tensor`: Tensor or LazyTensor containing the result of the matrix multiplication :math:`KM`,
            where :math:`K` is the (batched) matrix that this :obj:`gpytorch.lazy.LazyTensor` represents, and :math:`M`
            is the (batched) matrix input to this method.
        """
        # TODO: Move this check to MatmulLazyTensor and Matmul (so we can pass the shapes through from there)
        _matmul_broadcast_shape(self.shape, other.shape)

        if isinstance(other, LazyTensor):
            from .matmul_lazy_tensor import MatmulLazyTensor

            return MatmulLazyTensor(self, other)

        func = Matmul()
        return func.apply(self.representation_tree(), other, *self.representation())

    @property
    def matrix_shape(self):
        """
        Returns the shape of the matrix being represented (without batching).
        """
        return torch.Size(self.shape[-2:])

    def mul(self, other):
        """
        Multiplies the matrix by a constant, or elementwise the matrix by another matrix

        Args:
            other (:obj:`torch.tensor` or :obj:`~gpytorch.lazy.LazyTensor`): constant or matrix to elementwise
            multiply by.

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`: Another lazy tensor representing the result of the multiplication. if
            other was a constant (or batch of constants), this will likely be a
            :obj:`gpytorch.lazy.ConstantMulLazyTensor`. If other was
            another matrix, this will likely be a :obj:`gpytorch.lazy.MulLazyTensor`.
        """
        from .zero_lazy_tensor import ZeroLazyTensor
        from .non_lazy_tensor import lazify

        if isinstance(other, ZeroLazyTensor):
            return other

        if not (torch.is_tensor(other) or isinstance(other, LazyTensor)):
            other = torch.tensor(other, dtype=self.dtype, device=self.device)

        try:
            _mul_broadcast_shape(self.shape, other.shape)
        except RuntimeError:
            raise RuntimeError(
                "Cannot multiply LazyTensor of size {} by an object of size {}".format(self.shape, other.shape)
            )

        if torch.is_tensor(other):
            if other.numel() == 1:
                return self._mul_constant(other.squeeze())
            elif other.shape[-2:] == torch.Size((1, 1)):
                return self._mul_constant(other.view(*other.shape[:-2]))

        return self._mul_matrix(lazify(other))

    def ndimension(self):
        """
        Returns the number of dimensions
        """
        return len(self.size())

    def numel(self):
        """
        Returns the number of elements
        """
        return self.shape.numel()

    def numpy(self):
        """
        Return self as an evaluated numpy array
        """
        return self.evaluate().detach().cpu().numpy()

    def permute(self, *dims):
        num_dims = self.dim()
        orig_dims = dims
        dims = tuple(dim if dim >= 0 else dim + num_dims for dim in dims)

        if settings.debug.on():
            if len(dims) != num_dims:
                raise RuntimeError("number of dims don't match in permute")
            if sorted(set(dims)) != sorted(dims):
                raise RuntimeError("repeated dim in permute")

            for dim, orig_dim in zip(dims, orig_dims):
                if dim >= num_dims:
                    raise RuntimeError(
                        "Dimension out of range (expected to be in range of [{}, {}], but got "
                        "{}.".format(-num_dims, num_dims - 1, orig_dim)
                    )

        if dims[-2:] != (num_dims - 2, num_dims - 1):
            raise ValueError("At the moment, cannot permute the non-batch dimensions of LazyTensors.")

        return self._permute_batch(*dims[:-2])

    def prod(self, dim=None):
        """
        For a `b x n x m` LazyTensor, compute the product over the batch dimension.

        The `mul_batch_size` controls whether or not the batch dimension is grouped when multiplying.
            * `mul_batch_size=None` (default): The entire batch dimension is multiplied. Returns a `n x n` LazyTensor.
            * `mul_batch_size=k`: Creates `b/k` groups, and muls the `k` entries of this group.
                (The LazyTensor is reshaped as a `b/k x k x n x m` LazyTensor and the `k` dimension is multiplied over.
                Returns a `b/k x n x m` LazyTensor.

        Args:
            :attr:`mul_batch_size` (int or None):
                Controls the number of groups that are multiplied over (default: None).

        Returns:
            :obj:`~gpytorch.lazy.LazyTensor`

        Example:
            >>> lazy_tensor = gpytorch.lazy.NonLazyTensor(torch.tensor([
                    [[2, 4], [1, 2]],
                    [[1, 1], [0, -1]],
                    [[2, 1], [1, 0]],
                    [[3, 2], [2, -1]],
                ]))
            >>> lazy_tensor.mul_batch().evaluate()
            >>> # Returns: torch.Tensor([[12, 8], [0, 0]])
            >>> lazy_tensor.mul_batch(mul_batch_size=2)
            >>> # Returns: torch.Tensor([[[2, 4], [0, -2]], [[6, 2], [2, 0]]])
        """
        if dim is None:
            raise ValueError("At the moment, LazyTensor.prod requires a dim argument (got None)")

        orig_dim = dim
        if dim < 0:
            dim = self.dim() + dim
        if dim >= len(self.batch_shape):
            raise ValueError(
                "At the moment, LazyTensor.prod only works on batch dimensions. "
                "Got dim={} for LazyTensor of shape {}".format(orig_dim, self.shape)
            )

        return self._prod_batch(dim)

    def repeat(self, *sizes):
        """
        Repeats this tensor along the specified dimensions.

        Currently, this only works to create repeated batches of a 2D LazyTensor.
        I.e. all calls should be `lazy_tensor.repeat(<size>, 1, 1)`.

        Example:
            >>> lazy_tensor = gpytorch.lazy.ToeplitzLazyTensor(torch.tensor([4. 1., 0.5]))
            >>> lazy_tensor.repeat(2, 1, 1).evaluate()
            tensor([[[4.0000, 1.0000, 0.5000],
                     [1.0000, 4.0000, 1.0000],
                     [0.5000, 1.0000, 4.0000]],
                    [[4.0000, 1.0000, 0.5000],
                     [1.0000, 4.0000, 1.0000],
                     [0.5000, 1.0000, 4.0000]]])
        """
        from .batch_repeat_lazy_tensor import BatchRepeatLazyTensor

        if len(sizes) < 3 or tuple(sizes[-2:]) != (1, 1):
            raise RuntimeError(
                "Invalid repeat arguments {}. Currently, repeat only works to create repeated "
                "batches of a 2D LazyTensor.".format(tuple(sizes))
            )
        return BatchRepeatLazyTensor(self, batch_repeat=torch.Size(sizes[:-2]))

    def representation(self):
        """
        Returns the Tensors that are used to define the LazyTensor
        """
        representation = []
        for arg in self._args:
            if torch.is_tensor(arg):
                representation.append(arg)
            elif hasattr(arg, "representation") and callable(arg.representation):  # Is it a LazyTensor?
                representation += list(arg.representation())
            else:
                raise RuntimeError("Representation of a LazyTensor should consist only of Tensors")
        return tuple(representation)

    def representation_tree(self):
        """
        Returns a :obj:`gpytorch.lazy.LazyTensorRepresentationTree` tree object that recursively encodes the
        representation of this lazy tensor. In particular, if the definition of this lazy tensor depends on other
        lazy tensors, the tree is an object that can be used to reconstruct the full structure of this lazy tensor,
        including all subobjects. This is used internally.
        """
        return LazyTensorRepresentationTree(self)

    @property
    def requires_grad(self):
        return any(
            arg.requires_grad
            for arg in tuple(self._args) + tuple(self._kwargs.values())
            if hasattr(arg, "requires_grad")
        )

    @requires_grad.setter
    def requires_grad(self, val):
        for arg in self._args:
            if hasattr(arg, "requires_grad"):
                if arg.dtype in (torch.float, torch.double, torch.half):
                    arg.requires_grad = val
        for arg in self._kwargs.values():
            if hasattr(arg, "requires_grad"):
                arg.requires_grad = val

    def requires_grad_(self, val):
        """
        Sets `requires_grad=val` on all the Tensors that make up the LazyTensor
        This is an inplace operation.
        """
        self.requires_grad = val
        return self

    @cached(name="root_decomposition")
    def root_decomposition(self):
        """
        Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
        This can be used for sampling from a Gaussian distribution, or for obtaining a
        low-rank version of a matrix
        """
        from .chol_lazy_tensor import CholLazyTensor
        from .root_lazy_tensor import RootLazyTensor

        if not self.is_square:
            raise RuntimeError(
                "root_decomposition only operates on (batches of) square (symmetric) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if (
            self.size(-1) <= settings.max_cholesky_size.value()
            or settings.fast_computations.covar_root_decomposition.off()
        ):
            try:
                res = self.cholesky()
                return CholLazyTensor(res)

            except RuntimeError as e:
                warnings.warn(
                    "Runtime Error when computing Cholesky decomposition: {}. Using RootDecomposition.".format(e)
                )

        res = self._root_decomposition()
        return RootLazyTensor(res)

    @cached(name="root_inv_decomposition")
    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        """
        Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
        This can be used for sampling from a Gaussian distribution, or for obtaining a
        low-rank version of a matrix
        """
        from .root_lazy_tensor import RootLazyTensor
        from .non_lazy_tensor import lazify

        if self.shape[-2:].numel() == 1:
            return RootLazyTensor(1 / self.evaluate().sqrt())

        if (
            self.size(-1) <= settings.max_cholesky_size.value()
            or settings.fast_computations.covar_root_decomposition.off()
        ):
            try:
                L = delazify(self.cholesky())
                # we know L is triangular, so inverting is a simple triangular solve agaist the identity
                # we don't need the batch shape here, thanks to broadcasting
                Eye = torch.eye(L.shape[-2], device=L.device, dtype=L.dtype)
                Linv = torch.triangular_solve(Eye, L, upper=False)[0]
                res = lazify(Linv.transpose(-1, -2))
                return RootLazyTensor(res)
            except RuntimeError as e:
                warnings.warn(
                    "Runtime Error when computing Cholesky decomposition: {}. Using RootDecomposition.".format(e)
                )

        if not self.is_square:
            raise RuntimeError(
                "root_inv_decomposition only operates on (batches of) square (symmetric) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        if initial_vectors is not None:
            if self.dim() == 2 and initial_vectors.dim() == 1:
                if self.shape[-1] != initial_vectors.numel():
                    raise RuntimeError(
                        "LazyTensor (size={}) cannot be multiplied with initial_vectors (size={}).".format(
                            self.shape, initial_vectors.shape
                        )
                    )
            elif self.dim() != initial_vectors.dim():
                raise RuntimeError(
                    "LazyTensor (size={}) and initial_vectors (size={}) should have the same number "
                    "of dimensions.".format(self.shape, initial_vectors.shape)
                )
            elif self.batch_shape != initial_vectors.shape[:-2] or self.shape[-1] != initial_vectors.shape[-2]:
                raise RuntimeError(
                    "LazyTensor (size={}) cannot be multiplied with initial_vectors (size={}).".format(
                        self.shape, initial_vectors.shape
                    )
                )

        inv_roots = self._root_inv_decomposition(initial_vectors)

        # Choose the best of the inv_roots, if there were more than one initial vectors
        if initial_vectors is not None and initial_vectors.size(-1) > 1:
            num_probes = initial_vectors.size(-1)
            test_vectors = test_vectors.unsqueeze(0)

            # Compute solves
            solves = inv_roots.matmul(inv_roots.transpose(-1, -2).matmul(test_vectors))

            # Compute self * solves
            solves = (
                solves.permute(*range(1, self.dim() + 1), 0)
                .contiguous()
                .view(*self.batch_shape, self.matrix_shape[-1], -1)
            )
            mat_times_solves = self.matmul(solves)
            mat_times_solves = mat_times_solves.view(*self.batch_shape, self.matrix_shape[-1], -1, num_probes).permute(
                -1, *range(0, self.dim())
            )

            # Compute residuals
            residuals = (mat_times_solves - test_vectors).norm(2, dim=-2)
            residuals = residuals.view(residuals.size(0), -1).sum(-1)

            # Choose solve that best fits
            _, best_solve_index = residuals.min(0)
            inv_root = inv_roots[best_solve_index].squeeze(0)

        else:
            inv_root = inv_roots

        return RootLazyTensor(inv_root)

    def size(self, val=None):
        """
        Returns the size of the resulting Tensor that the lazy tensor represents
        """
        size = self._size()
        if val is not None:
            return size[val]
        return size

    def squeeze(self, dim):
        if self.size(dim) != 1:
            return self
        else:
            index = [_noop_index] * self.dim()
            index[dim] = 0
            index = tuple(index)
            return self[index]

    @property
    def shape(self):
        return self.size()

    def sum(self, dim=None):
        """
        Sum the LazyTensor across a dimension.
        The `dim` controls which batch dimension is summed over.
        If set to None, then sums all dimensions

        Args:
            :attr:`dim` (int):
                Which dimension is being summed over (default=None)

        Returns:
            :obj:`~gpytorch.lazy.LazyTensor` or Tensor.

        Example:
            >>> lazy_tensor = gpytorch.lazy.NonLazyTensor(torch.tensor([
                    [[2, 4], [1, 2]],
                    [[1, 1], [0, -1]],
                    [[2, 1], [1, 0]],
                    [[3, 2], [2, -1]],
                ]))
            >>> lazy_tensor.sum(0).evaluate()
        """
        # Case: summing everything
        if dim is None:
            ones = torch.ones(self.size(-2), 1, dtype=self.dtype, device=self.device)
            return (self @ ones).sum()

        # Otherwise: make dim positive
        orig_dim = dim
        if dim < 0:
            dim = self.dim() + dim

        # Case: summing across columns
        if dim == (self.dim() - 1):
            ones = torch.ones(self.size(-1), 1, dtype=self.dtype, device=self.device)
            return (self @ ones).squeeze(-1)
        # Case: summing across rows
        elif dim == (self.dim() - 2):
            ones = torch.ones(self.size(-2), 1, dtype=self.dtype, device=self.device)
            return (self.transpose(-1, -2) @ ones).squeeze(-1)
        # Otherwise: it's a batch dimension
        elif dim < self.dim():
            return self._sum_batch(dim)
        else:
            raise ValueError("Invalid dim ({}) for LazyTensor of size {}".format(orig_dim, self.shape))

    def to(self, device_id):
        """
        A device-agnostic method of moving the lazy_tensor to the specified device.

        Args:
            device_id (:obj: `torch.device`): Which device to use (GPU or CPU).
        Returns:
            :obj:`~gpytorch.lazy.LazyTensor`: New LazyTensor identical to self on specified device
        """
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, "to"):
                new_args.append(arg.to(device_id))
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, "to"):
                new_kwargs[name] = val.to(device_id)
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    def t(self):
        """
        Alias of :meth:`~gpytorch.lazy.LazyTensor.transpose` for 2D LazyTensor.
        (Tranposes the two dimensions.)
        """
        if self.ndimension() != 2:
            raise RuntimeError("Cannot call t for more than 2 dimensions")
        return self.transpose(0, 1)

    def transpose(self, dim1, dim2):
        """
        Transpose the dimensions `dim1` and `dim2` of the LazyTensor.

        Example:
            >>> lazy_tensor = gpytorch.lazy.NonLazyTensor(torch.randn(3, 5))
            >>> lazy_tensor.transpose(0, 1)
        """
        ndimension = self.ndimension()
        if dim1 < 0:
            dim1 = ndimension + dim1
        if dim2 < 0:
            dim2 = ndimension + dim2
        if dim1 >= ndimension or dim2 >= ndimension or not isinstance(dim1, int) or not isinstance(dim2, int):
            raise RuntimeError("Invalid dimension")

        # Batch case
        if dim1 < ndimension - 2 and dim2 < ndimension - 2:
            small_dim = dim1 if dim1 < dim2 else dim2
            large_dim = dim2 if dim1 < dim2 else dim1
            res = self._permute_batch(
                *range(small_dim), large_dim, *range(small_dim + 1, large_dim), small_dim,
                *range(large_dim + 1, ndimension - 2)
            )

        elif dim1 >= ndimension - 2 and dim2 >= ndimension - 2:
            res = self._transpose_nonbatch()

        else:
            raise RuntimeError("Cannot transpose batch dimension with non-batch dimension")

        return res

    def unsqueeze(self, dim):
        positive_dim = (self.dim() + dim + 1) if dim < 0 else dim
        if positive_dim > len(self.batch_shape):
            raise ValueError(
                "Can only unsqueeze batch dimensions of {} (size {}). Got "
                "dim={}.".format(self.__class__.__name__, self.shape, dim)
            )
        res = self._unsqueeze_batch(positive_dim)
        return res

    def zero_mean_mvn_samples(self, num_samples):
        """
        Assumes that self is a covariance matrix, or a batch of covariance matrices.
        Returns samples from a zero-mean MVN, defined by self (as covariance matrix)

        Self should be symmetric, either (batch_size x num_dim x num_dim) or (num_dim x num_dim)

        Args:
            :attr:`num_samples` (int):
                Number of samples to draw.

        Returns:
            :obj:`torch.tensor`:
                Samples from MVN (num_samples x batch_size x num_dim) or (num_samples x num_dim)
        """
        if self.size()[-2:] == torch.Size([1, 1]):
            covar_root = self.evaluate().sqrt()
        else:
            covar_root = self.root_decomposition().root

        base_samples = torch.randn(
            *self.batch_shape, covar_root.size(-1), num_samples, dtype=self.dtype, device=self.device
        )
        samples = covar_root.matmul(base_samples).permute(-1, *range(self.dim() - 1)).contiguous()

        return samples

    def __add__(self, other):
        """
        Return a :obj:`gpytorch.lazy.LazyTensor` that represents the sum of this lazy tensor and another matrix
        or lazy tensor.

        Args:
            :attr:`other` (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyTensor`):
                Matrix to add to this one.

        Returns:
            :obj:`gpytorch.lazy.SumLazyTensor`:
                A sum lazy tensor representing the sum of this lazy tensor and other.
        """
        from .sum_lazy_tensor import SumLazyTensor
        from .zero_lazy_tensor import ZeroLazyTensor
        from .diag_lazy_tensor import DiagLazyTensor
        from .added_diag_lazy_tensor import AddedDiagLazyTensor

        if isinstance(other, ZeroLazyTensor):
            return self
        elif isinstance(other, DiagLazyTensor):
            return AddedDiagLazyTensor(self, other)
        else:
            return SumLazyTensor(self, other)

    def __div__(self, other):
        """
        Return a :obj:`gpytorch.lazy.LazyTensor` that represents the product of this lazy tensor and
        the elementwise reciprocal of another matrix or lazy tensor.

        Args:
            :attr:`other` (:obj:`torch.tensor` or :obj:`gpytorch.lazy.LazyTensor`):
                Matrix to divide this one by.

        Returns:
            :obj:`gpytorch.lazy.MulLazyTensor`:
                Result of division.
        """
        from .zero_lazy_tensor import ZeroLazyTensor

        if isinstance(other, ZeroLazyTensor):
            raise RuntimeError("Attempted to divide by a ZeroLazyTensor (divison by zero)")

        return self.mul(1.0 / other)

    def __getitem__(self, index):
        """
        Supports subindexing of the matrix this LazyTensor represents. This may return either another
        :obj:`gpytorch.lazy.LazyTensor` or a :obj:`torch.tensor` depending on the exact implementation.
        """
        ndimension = self.ndimension()

        # Process the index
        index = index if isinstance(index, tuple) else (index,)
        index = tuple(torch.tensor(idx) if isinstance(idx, list) else idx for idx in index)
        index = tuple(idx.item() if torch.is_tensor(idx) and not len(idx.shape) else idx for idx in index)

        # Handle the ellipsis
        # Find the index of the ellipsis
        ellipsis_locs = tuple(index for index, item in enumerate(index) if item is Ellipsis)
        if settings.debug.on():
            if len(ellipsis_locs) > 1:
                raise RuntimeError(
                    "Cannot have multiple ellipsis in a __getitem__ call. LazyTensor {} "
                    " received index {}.".format(self, index)
                )
        if len(ellipsis_locs) == 1:
            ellipsis_loc = ellipsis_locs[0]
            num_to_fill_in = ndimension - (len(index) - 1)
            index = (
                index[:ellipsis_loc]
                + tuple(_noop_index for _ in range(num_to_fill_in))
                + index[ellipsis_loc + 1 :]
            )

        # Pad the index with empty indices
        index = index + tuple(_noop_index for _ in range(ndimension - len(index)))

        # Make the index a tuple again
        *batch_indices, row_index, col_index = index

        # Helpers to determine what the final shape will be if we're tensor indexed
        batch_has_tensor_index = bool(len(batch_indices)) and any(torch.is_tensor(index) for index in batch_indices)
        row_has_tensor_index = torch.is_tensor(row_index)
        col_has_tensor_index = torch.is_tensor(col_index)
        # These are the cases where the row and/or column indices will be "absorbed" into other indices
        row_col_are_absorbed = any((
            batch_has_tensor_index and (row_has_tensor_index or col_has_tensor_index),
            not batch_has_tensor_index and (row_has_tensor_index and col_has_tensor_index),
        ))

        # If we're indexing the LT with ints or slices
        # Replace the ints with slices, and we'll just squeeze the dimensions later
        squeeze_row = False
        squeeze_col = False
        if isinstance(row_index, int):
            row_index = slice(row_index, row_index + 1, None)
            squeeze_row = True
        if isinstance(col_index, int):
            col_index = slice(col_index, col_index + 1, None)
            squeeze_col = True

        # Call self._getitem - now that the index has been processed
        # Alternatively, if we're using tensor indices and losing dimensions, use self._get_indices
        if row_col_are_absorbed:
            # Convert all indices into tensor indices
            *batch_indices, row_index, col_index, = _convert_indices_to_tensors(
                self, (*batch_indices, row_index, col_index)
            )
            res = self._get_indices(row_index, col_index, *batch_indices)
        else:
            res = self._getitem(row_index, col_index, *batch_indices)

        # If we selected a single row and/or column (or did tensor indexing), we'll be retuning a tensor
        # with the appropriate shape
        if (squeeze_row or squeeze_col or row_col_are_absorbed):
            res = delazify(res)
        if squeeze_row:
            res = res.squeeze(-2)
        if squeeze_col:
            res = res.squeeze(-1)

        # Make sure we're getting the expected shape
        if settings.debug.on() and self.__class__._check_size:
            expected_shape = _compute_getitem_size(self, index)
            if expected_shape != res.shape:
                raise RuntimeError(
                    "{}.__getitem__ failed! Expected a final shape of size {}, got {}. This is a bug with GPyTorch, "
                    "or your custom LazyTensor.".format(self.__class__.__name__, expected_shape, res.shape)
                )

        # We're done!
        return res

    def __matmul__(self, other):
        return self.matmul(other)

    def __mul__(self, other):
        return self.mul(other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self.mul(other)


def _import_dotted_name(name):
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


def delazify(obj):
    """
    A function which ensures that `obj` is a (normal) Tensor.

    If `obj` is a Tensor, this function does nothing.
    If `obj` is a LazyTensor, this function evaluates it.
    """

    if torch.is_tensor(obj):
        return obj
    elif isinstance(obj, LazyTensor):
        return obj.evaluate()
    else:
        raise TypeError("object of class {} cannot be made into a Tensor".format(obj.__class__.__name__))


_deprecate_renamed_methods(LazyTensor, inv_quad_log_det="inv_quad_logdet", log_det="logdet")

__all__ = ["LazyTensor", "delazify"]
