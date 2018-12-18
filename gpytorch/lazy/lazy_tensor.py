#!/usr/bin/env python3

import math

import gpytorch
import torch

from .. import settings
from ..functions._inv_matmul import InvMatmul
from ..functions._inv_quad_log_det import InvQuadLogDet
from ..functions._matmul import Matmul
from ..functions._root_decomposition import RootDecomposition
from ..utils.broadcasting import _matmul_broadcast_shape
from ..utils.memoize import cached
from ..utils.qr import batch_qr
from ..utils.svd import batch_svd
from .lazy_tensor_representation_tree import LazyTensorRepresentationTree


class LazyTensor(object):
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

    * :func:`~gpytorch.lazy.LazyTensor._get_indices`, which returns a Tensor where the entries are determined by
      LongTensors of indices.
    * :func:`~gpytorch.lazy.LazyTensor._matmul`, which performs a matrix multiplication :math:`KM`
    * :func:`~gpytorch.lazy.LazyTensor._quad_form_derivative`, which computes a quadratic form with the derivative,
      :math:`\mathbf{v}^{\\top}\\frac{dK}{dR}\mathbf{v}`, where :math:`R` denotes the actual tensors used to represent
      :math:`K`. In the linear kernel example, :math:`K=XX^{\\top}`, this would be :math:`\\frac{dK}{dX}`. If :math:`K`
      is a Toeplitz matrix (see :class:`gpytorch.lazy.ToeplitzLazyTensor`) represented by its first column
      :math:`\mathbf{c}`, this would return :math:`\mathbf{v}^{\\top}\\frac{dK}{d\mathbf{c}}\mathbf{v}`.
    * :func:`~gpytorch.lazy.LazyTensor._size`, which returns a :class:`torch.Size` containing the dimensions of
      :math:`K`.
    * :func:`~gpytorch.lazy.LazyTensor._transpose_nonbatch`, which returns a transposed version of the LazyTensor

    In addition to these, a LazyTensor may need to define the :func:`~gpytorch.lazy.LazyTensor._transpose_nonbatch`,
    :func:`~gpytorch.lazy.LazyTensor._get_indices`, and :func:`~gpytorch.lazy.LazyTensor._get_indices`
    functions in special cases. See the documentation for these methods for details.

    .. note::
        The base LazyTensor class provides default implementations of many other operations in order to mimic the
        behavior of a standard tensor as closely as possible. For example, we provide default implementations of
        :func:`~gpytorch.lazy.LazyTensor.__getitem__`, :func:`~gpytorch.lazy.LazyTensor.__add__`, etc that either
        make use of other lazy tensors or exploit the functions that **must** be defined above.

        While these implementations are provided for convenience, it is advisable in many cases to override them for the
        sake of efficiency.

    .. note::
        LazyTensors are designed by default to optionally represent batches of matrices. Thus, the size of a
        LazyTensor may be (for example) :math:`b \times n \times n`. Many of the methods are designed to efficiently
        operate on these batches if present.
    """

    def _get_indices(self, *batch_indices, left_indices, right_indices):
        """
        Returns entries of the matrix, indexed by batch, row, and column indices
        """
        raise NotImplementedError("The class {} requires a _get_indices function!".format(self.__class__.__name__))

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

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

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

    def _getitem(self, *indices):
        """
        Supports subindexing of the matrix this LazyTensor represents. This may return either another
        :obj:`gpytorch.lazy.LazyTensor` or a :obj:`torch.tensor` depending on the exact implementation.

        ..note::
            LazyTensor.__getitem__ uses this as a helper method. If you are writing your own custom LazyTensor,
            override this method rather than __getitem__ (so that you don't have to repeat the extra work)

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.__getitem__`,
            which does some additional work. Calling this method directly is discouraged.

        Args:
            :attr:`indices` (tuple of int, slice, or LongTensor):
                A collection of indices for each of the dimensions. There will be exactly one index per dimension.
        """
        if settings.debug.on():
            if len(indices) != self.dim():
                raise RuntimeError(
                    "{}._getitem() called with {} indices - expected {}. "
                    "This is potentially a bug in GPyTorch.".format(self.__class__.__name__, len(indices), self.dim())
                )

        components = list(self._args)
        indices = list(indices)

        # Normal case if we're indexing the LT with ints or slices
        # Also squeeze dimensions if we're indexing with tensors
        squeeze_left = False
        squeeze_right = False
        if isinstance(indices[-2], int):
            indices[-2] = slice(indices[-2], indices[-2] + 1, None)
            squeeze_left = True
        elif torch.is_tensor(indices[-2]):
            squeeze_left = True
        if isinstance(indices[-1], int):
            indices[-1] = slice(indices[-1], indices[-1] + 1, None)
            squeeze_right = True
        elif torch.is_tensor(indices[-1]):
            squeeze_right = True

        # Handle batch dimensions
        isbatch = self.dim() >= 3
        first_tensor_index_dim = None
        if isbatch:
            batch_index = tuple(indices[:-2])
            for i, item in enumerate(components):
                components[i] = item[batch_index]

            for i, idx in enumerate(batch_index):
                if torch.is_tensor(idx):
                    first_tensor_index_dim = i
                    break

        new_lazy_tensor = self.__class__(*components, **self._kwargs)

        # Handle index
        left_index = indices[-2]
        right_index = indices[-1]

        # Special case: if both row and col are not indexed, then we are done
        if (
            not torch.is_tensor(left_index)
            and left_index == slice(None, None, None)
            and not torch.is_tensor(right_index)
            and right_index == slice(None, None, None)
        ):
            return new_lazy_tensor

        # Special case: if both row and col are tensor indexed, then we use _get_indices
        if torch.is_tensor(left_index) and torch.is_tensor(right_index):
            if left_index.numel() != right_index.numel():
                raise RuntimeError(
                    "Expected the tensor indices to be the same size: got {} and {}".format(
                        left_index.numel(), right_index.numel()
                    )
                )

            if new_lazy_tensor.ndimension() == 2:
                return new_lazy_tensor._get_indices(left_index, right_index)

            else:
                batch_index = torch.arange(0, new_lazy_tensor.size(0), dtype=torch.long, device=self.device)
                if first_tensor_index_dim is not None:
                    if batch_index.numel() != left_index.numel():
                        raise RuntimeError(
                            "Expected the tensor indices to be the same size: got {}, {} and {}".format(
                                batch_index.numel(), left_index.numel(), right_index.numel()
                            )
                        )
                    return new_lazy_tensor._get_indices(left_index, right_index, batch_index)
                else:
                    batch_size = batch_index.numel()
                    row_col_size = left_index.numel()
                    batch_index = batch_index.unsqueeze(1).repeat(1, row_col_size).view(-1)
                    left_index = left_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
                    right_index = right_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
                    res = new_lazy_tensor._get_indices(left_index, right_index, batch_index)
                    return res.view(batch_size, row_col_size)

        # Normal case: we have to do some processing on eithe rthe rows or columns
        res = new_lazy_tensor._getitem_nonbatch(left_index, right_index, first_tensor_index_dim)
        if (squeeze_left or squeeze_right) and isinstance(res, LazyTensor):
            res = res.evaluate()
        if squeeze_left:
            res = res.squeeze(-2)
        if squeeze_right:
            res = res.squeeze(-1)

        return res

    def _getitem_nonbatch(self, row_index, col_index, first_tensor_index_dim=None):
        """
        Given an index over rows and columns, gets those items from the LazyTensor.
        Implementing this is not necessary, but it improves performance

        Args:
            row_index (slice or LongTensor): index over rows
            col_index (slice or LongTensor): index over columns
            first_tensor_index_dim (int or None): first batch dim to have a tensor index (default: None)

        Returns:
            LazyTensor
        """
        from .interpolated_lazy_tensor import InterpolatedLazyTensor

        ndimension = self.ndimension()
        batch_sizes = list(self.size()[:-2])

        left_row_iter = torch.arange(0, self.size()[-2], dtype=torch.long, device=self.device)
        right_row_iter = torch.arange(0, self.size()[-1], dtype=torch.long, device=self.device)
        left_interp_indices = left_row_iter[row_index].unsqueeze(-1)
        right_interp_indices = right_row_iter[col_index].unsqueeze(-1)

        left_interp_len = len(left_interp_indices)
        right_interp_len = len(right_interp_indices)
        for _ in range(ndimension - 2):
            left_interp_indices.unsqueeze_(0)
            right_interp_indices.unsqueeze_(0)

        if first_tensor_index_dim is not None and torch.is_tensor(row_index):
            view_size = [1] * ndimension
            view_size[first_tensor_index_dim] = left_interp_indices.numel()
            left_interp_indices = left_interp_indices.view(*view_size).expand(*(batch_sizes + [1, 1]))
        else:
            left_interp_indices = left_interp_indices.expand(*(batch_sizes + [left_interp_len, 1]))
        left_interp_values = torch.ones(left_interp_indices.size(), dtype=self.dtype, device=self.device)
        if first_tensor_index_dim is not None and torch.is_tensor(col_index):
            view_size = [1] * ndimension
            view_size[first_tensor_index_dim] = right_interp_indices.numel()
            right_interp_indices = right_interp_indices.view(*view_size).expand(*(batch_sizes + [1, 1]))
        else:
            right_interp_indices = right_interp_indices.expand(*(batch_sizes + [right_interp_len, 1]))
        right_interp_values = torch.ones(right_interp_indices.size(), dtype=self.dtype, device=self.device)

        res = InterpolatedLazyTensor(
            self, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        return res

    def _inv_matmul_preconditioner(self):
        """
        (Optional) define a preconditioner that can be used for linear systems, but not necessarily
        for log determinants. By default, this can call :meth:`~gpytorch.lazy.LazyTensor._preconditioner`.

        Returns:
            function: a function on x which performs P^{-1}(x)
        """
        base_precond, _ = self._preconditioner()

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
                proj_q = batch_qr(projected_mat)
                orthog_projected_mat = self._matmul(proj_q).transpose(-2, -1)
                U, S, V = batch_svd(orthog_projected_mat)
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

    def _quad_form_derivative(self, left_vecs, right_vecs):
        """
        Given u (left_vecs) and v (right_vecs),
        Computes the derivatives of (u^t K v) w.r.t. K

        ..note::
            This method is intended to be used only internally by various Functions that support backpropagation.
            For example, this method is used internally by :func:`~gpytorch.lazy.LazyTensor.inv_quad_log_det`. It is
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

        return grads

    def _preconditioner(self):
        """
        (Optional) define a preconditioner (P) for linear conjugate gradients

        Returns:
            function: a function on x which performs P^{-1}(x)
            scalar: the log determinant of P
        """
        return None, None

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

        if self.size(-1) != self.size(-2):
            raise RuntimeError("add_diag only defined for square matrices")

        # Expand things the correct way
        if self.ndimension() == 3:
            if diag.dim() == 0:
                diag = diag.view(1, 1).expand(self.size(0), self.size(1))
            elif diag.dim() == 1:
                diag = diag.unsqueeze(0).expand(self.size(0), self.size(1))
            elif diag.ndimension() == 2:
                diag = diag.expand(self.size(0), self.size(1))
            else:
                raise RuntimeError(
                    "For a 3D tensor ({}), add_diag expects a 1D or 2D diag. "
                    "Got size ({})".format(self.size(), diag.size())
                )
        else:
            if diag.dim() == 0:
                diag = diag.view(1).expand(self.size(0))
            elif diag.dim() == 1:
                diag = diag.expand(self.size(0))
            else:
                raise RuntimeError(
                    "For a 2D tensor ({}), add_diag expects a 0D or 1D diag. "
                    "Got size ({})".format(self.size(), diag.size())
                )

        diag_lazy_tsr = DiagLazyTensor(diag)
        return AddedDiagLazyTensor(self, diag_lazy_tsr)

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
        size = self.size()
        if size[-1] != size[-2]:
            raise RuntimeError("Diag works on square matrices (or batches)")

        row_col_iter = torch.arange(0, size[-1], dtype=torch.long, device=self.device)
        if self.ndimension() == 3:
            batch_iter = torch.arange(0, size[0], dtype=torch.long, device=self.device)
            batch_iter = batch_iter.unsqueeze(1).repeat(1, size[1]).view(-1)
            row_col_iter = row_col_iter.unsqueeze(1).repeat(size[0], 1).view(-1)
            return self._get_indices(row_col_iter, row_col_iter, batch_iter).view(size[0], size[1])
        else:
            return self._get_indices(row_col_iter, row_col_iter)

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
            shape = sizes[0]
        elif all(isinstance(size, int) for size in sizes):
            shape = torch.Size(sizes)
        else:
            raise RuntimeError("Invalid arguments {} to expand.".format(sizes))

        current_shape = torch.Size([1 for _ in range(len(shape) - self.dim())] + list(self.shape))
        repeat_shape = torch.Size(
            [expand_size // current_size for expand_size, current_size in zip(shape, current_shape)]
        )
        return self.repeat(*repeat_shape)

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
            return self.transpose(-1, -2).matmul(eye).transpose(-1, -2).contiguous()
        else:
            eye = torch.eye(num_cols, dtype=self.dtype, device=self.device)
            eye = eye.expand(*self.batch_shape, num_cols, num_cols)
            return self.matmul(eye)

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

        func = InvMatmul(
            self.representation_tree(), preconditioner=self._inv_matmul_preconditioner(),
            has_left=(left_tensor is not None)
        )
        if left_tensor is None:
            return func(right_tensor, *self.representation())
        else:
            return func(left_tensor, right_tensor, *self.representation())

    def inv_quad(self, tensor):
        """
        Computes an inverse quadratic form (w.r.t self) with several right hand sides.
        I.e. computes tr( tensor^T self^{-1} tensor )

        NOTE: Don't overwrite this function!
        Instead, overwrite inv_quad_log_det

        Args:
            - tensor (tensor nxk) - Vector (or matrix) for inverse quad

        Returns:
            - tensor - tr( tensor^T (self)^{-1} tensor )
        """
        res, _ = self.inv_quad_log_det(inv_quad_rhs=tensor, log_det=False)
        return res

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False, reduce_inv_quad=True):
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
        if not self.is_square:
            raise RuntimeError(
                "inv_quad_log_det only operates on (batches of) square (positive semi-definite) LazyTensors. "
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

        inv_quad_term, log_det_term = InvQuadLogDet(
            representation_tree=self.representation_tree(),
            matrix_shape=self.matrix_shape,
            batch_shape=self.batch_shape,
            dtype=self.dtype,
            device=self.device,
            inv_quad=(inv_quad_rhs is not None),
            log_det=log_det,
            preconditioner=self._preconditioner()[0],
            log_det_correction=self._preconditioner()[1],
        )(*args)

        if inv_quad_term.numel() and reduce_inv_quad:
            inv_quad_term = inv_quad_term.sum(-1)
        return inv_quad_term, log_det_term

    @property
    def is_square(self):
        return self.matrix_shape[0] == self.matrix_shape[1]

    def log_det(self):
        """
        Computes an (approximate) log determinant of the matrix

        NOTE: Don't overwrite this function!
        Instead, overwrite inv_quad_log_det

        Returns:
            - scalar: log determinant
        """
        _, res = self.inv_quad_log_det(inv_quad_rhs=None, log_det=True)
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

        func = Matmul(self.representation_tree())
        return func(other, *self.representation())

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
        if not (torch.is_tensor(other) or isinstance(other, LazyTensor)) or (
            torch.is_tensor(other) and (other.numel() == 1 or (self.dim() == 3 and other.numel() == self.size(0)))
        ):
            from .constant_mul_lazy_tensor import ConstantMulLazyTensor

            return ConstantMulLazyTensor(self, other)

        elif other.size() == self.size():
            from .mul_lazy_tensor import MulLazyTensor

            return MulLazyTensor(self, other).evaluate_kernel()

        else:
            raise RuntimeError(
                '"other" must be a constant (or batch of constants), or the same size as self.\n'
                "Expected: size of [1] or [%d] or %s.\n"
                "Got: size of %s"
                % (self.size(0) if self.ndimension() == 3 else 1, repr(self.size()), repr(other.size()))
            )

    def mul_batch(self, mul_batch_size=None):
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
        from .mul_lazy_tensor import MulLazyTensor
        from .root_lazy_tensor import RootLazyTensor

        if self.ndimension() < 3:
            raise RuntimeError("mul_batch only works with batched lazy tensors")
        if self.size(0) == 1:
            return self.sum_batch()

        roots = self.root_decomposition().root.evaluate()
        n_batch = roots.size(0) if mul_batch_size is None else mul_batch_size
        true_batch_size = roots.size(0) // mul_batch_size if mul_batch_size is not None else 1

        while True:
            roots = roots.view(true_batch_size, n_batch, roots.size(1), roots.size(2))

            # Take care of extra roots (odd roots), if they exist
            if n_batch % 2:
                extra_root = (
                    torch.randn(roots.size(0), 1, roots.size(2), roots.size(3), dtype=roots.dtype, device=roots.device)
                    .mul_(1e-6 / math.sqrt(roots.size(3)))
                    .add_(1.0 / math.sqrt(roots.size(3)))
                )
                roots = torch.cat([roots, extra_root], 1)
                n_batch += 1

            # Divide and conqour
            # Assumes that there's an even number of roots
            part1 = roots[:, : n_batch // 2]
            part1 = part1.contiguous().view(-1, roots.size(2), roots.size(3))
            part2 = roots[:, n_batch // 2 : 2 * (n_batch // 2)]
            part2 = part2.contiguous().view(-1, roots.size(2), roots.size(3))

            if n_batch // 2 == 1:
                if mul_batch_size is None:
                    part1 = part1.squeeze(0)
                    part2 = part2.squeeze(0)
                res = MulLazyTensor(RootLazyTensor(part1), RootLazyTensor(part2)).evaluate_kernel()
                break
            else:
                res = MulLazyTensor(RootLazyTensor(part1), RootLazyTensor(part2)).evaluate_kernel()
                roots = res.root_decomposition().root.evaluate()
                n_batch = n_batch // 2

        return res

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
        if len(sizes) < 3 or tuple(sizes[-2:]) != (1, 1):
            raise RuntimeError(
                "Invalid repeat arguments {}. Currently, repeat only works to create repeated "
                "batches of a 2D LazyTensor.".format(tuple(sizes))
            )

        from .batch_repeat_lazy_tensor import BatchRepeatLazyTensor

        return BatchRepeatLazyTensor(self, batch_repeat=torch.Size(sizes[:-2]))

    def representation(self):
        """
        Returns the Tensors that are used to define the LazyTensor
        """
        representation = []
        for arg in self._args:
            if torch.is_tensor(arg):
                representation.append(arg)
            elif isinstance(arg, LazyTensor):
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
        return any(arg.requires_grad for arg in tuple(self._args) + tuple(self._kwargs.values()))

    @requires_grad.setter
    def requires_grad(self, val):
        for arg in self._args:
            if hasattr(arg, "requires_grad"):
                if arg.dtype in (torch.float, torch.double, torch.half):
                    arg.requires_grad = val
        for val in self._kwargs.values():
            if hasattr(val, "requires_grad"):
                val.requires_grad = val

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
        from .root_lazy_tensor import RootLazyTensor

        if not self.is_square:
            raise RuntimeError(
                "root_decomposition only operates on (batches of) square (symmetric) LazyTensors. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        # when dealing with small matrices, it's usually faster to use Choleksy decomposition
        if self.matrix_shape.numel() <= (settings.max_cholesky_numel.value()):
            try:
                res = torch.cholesky(self.evaluate())
                return RootLazyTensor(res)
            except RuntimeError:
                pass

        res, _ = RootDecomposition(
            self.representation_tree(),
            max_iter=self.root_decomposition_size(),
            dtype=self.dtype,
            device=self.device,
            batch_shape=self.batch_shape,
            matrix_shape=self.matrix_shape,
        )(*self.representation())
        return RootLazyTensor(res)

    @cached
    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        """
        Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
        This can be used for sampling from a Gaussian distribution, or for obtaining a
        low-rank version of a matrix
        """
        from .root_lazy_tensor import RootLazyTensor

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

        roots, inv_roots = RootDecomposition(
            self.representation_tree(),
            max_iter=self.root_decomposition_size(),
            dtype=self.dtype,
            device=self.device,
            batch_shape=self.batch_shape,
            matrix_shape=self.matrix_shape,
            root=True,
            inverse=True,
            initial_vectors=initial_vectors,
        )(*self.representation())

        if initial_vectors is not None and initial_vectors.size(-1) > 1:
            getattr(self, '__cache')["root_decomposition"] = RootLazyTensor(roots[0])
        else:
            getattr(self, '__cache')["root_decomposition"] = RootLazyTensor(roots)

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

    def root_decomposition_size(self):
        """
        This is the inner size of the root decomposition.
        This is primarily used to determine if it will be cheaper to compute a
        different root or not
        """
        return settings.max_root_decomposition_size.value()

    def size(self, val=None):
        """
        Returns the size of the resulting Tensor that the lazy tensor represents
        """
        size = self._size()
        if val is not None:
            return size[val]
        return size

    @property
    def shape(self):
        return self.size()

    def sum_batch(self, sum_batch_size=None):
        """
        Sum the `b x n x m` LazyTensor over the batch dimension.

        The `sum_batch_size` controls whether or not the batch dimension is grouped when summing.
            * `sum_batch_size=None` (default): The entire batch dimension is summed. Returns a `n x n` LazyTensor.
            * `sum_batch_size=k`: Creates `b/k` groups, and sums the `k` entries of this group.
                (The LazyTensor is reshaped as a `b/k x k x n x m` LazyTensor and the `k` dimension is summed over.
                Returns a `b/k x n x m` LazyTensor.

        Args:
            :attr:`sum_batch_size` (int or None):
                Controls the number of groups that are summed over (default: None).

        Returns:
            :obj:`~gpytorch.lazy.LazyTensor`

        Example:
            >>> lazy_tensor = gpytorch.lazy.NonLazyTensor(torch.tensor([
                    [[2, 4], [1, 2]],
                    [[1, 1], [0, -1]],
                    [[2, 1], [1, 0]],
                    [[3, 2], [2, -1]],
                ]))
            >>> lazy_tensor.sum_batch().evaluate()
            >>> # Returns: torch.Tensor([[8, 8], [4, 0]])
            >>> lazy_tensor.sum_batch(sum_batch_size=2)
            >>> # Returns: torch.Tensor([[[3, 5], [1, 1]], [[5, 3], [3, -1]]])
        """
        from .sum_batch_lazy_tensor import SumBatchLazyTensor

        return SumBatchLazyTensor(self, num_blocks=sum_batch_size)

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
            res = self.__class__(*(arg.transpose(dim1, dim2) for arg in self._args), **self._kwargs)

        elif dim1 >= ndimension - 2 and dim2 >= ndimension - 2:
            res = self._transpose_nonbatch()

        else:
            raise RuntimeError("Cannot transpose batch dimension with non-batch dimension")

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

        if self.ndimension() == 3:
            base_samples = torch.randn(
                self.size(0), covar_root.size(-1), num_samples, dtype=self.dtype, device=self.device
            )
            samples = covar_root.matmul(base_samples).permute(2, 0, 1).contiguous()
        else:
            base_samples = torch.randn(covar_root.size(-1), num_samples, dtype=self.dtype, device=self.device)
            samples = covar_root.matmul(base_samples).permute(1, 0).contiguous()

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
                + tuple(slice(None, None, None) for _ in range(num_to_fill_in))
                + index[ellipsis_loc + 1 :]
            )

        # Pad the index with empty slices
        index = index + tuple(slice(None, None, None) for _ in range(ndimension - len(index)))

        # Make the index a tuple again
        index = tuple(index)

        # Call self._getitem - now that the index has been processed
        return self._getitem(*index)

    def __mul__(self, other):
        """
        Convenience alias of :meth:`~gpytorch.lazy.LazyTensor.mul` that allows the standard product operator to be
        used.
        """
        from .zero_lazy_tensor import ZeroLazyTensor
        from .diag_lazy_tensor import DiagLazyTensor

        if isinstance(other, ZeroLazyTensor):
            return other
        elif isinstance(other, DiagLazyTensor):
            return other * self

        return self.mul(other)

    def __setattr__(self, name, val):
        if torch.is_tensor(val) or isinstance(val, LazyTensor):
            if not hasattr(self, "_args"):
                raise RuntimeError(
                    "Cannot assign {name} to LazyTensor before calling LazyTensor.__init__()".format(name=name)
                )
        object.__setattr__(self, name, val)


def _import_dotted_name(name):
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj
