from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from ..functions._inv_matmul import InvMatmul
from ..functions._inv_quad_log_det import InvQuadLogDet
from ..functions._root_decomposition import RootDecomposition
from ..functions._matmul import Matmul
from .. import beta_features, settings
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

    * :func:`~gpytorch.lazy.LazyTensor._matmul`, which performs a matrix multiplication :math:`KM`
    * :func:`~gpytorch.lazy.LazyTensor._quad_form_derivative`, which computes a quadratic form with the derivative,
      :math:`\mathbf{v}^{\\top}\\frac{dK}{dR}\mathbf{v}`, where :math:`R` denotes the actual tensors used to represent
      :math:`K`. In the linear kernel example, :math:`K=XX^{\\top}`, this would be :math:`\\frac{dK}{dX}`. If :math:`K`
      is a Toeplitz matrix (see :class:`gpytorch.lazy.ToeplitzLazyTensor`) represented by its first column
      :math:`\mathbf{c}`, this would return :math:`\mathbf{v}^{\\top}\\frac{dK}{d\mathbf{c}}\mathbf{v}`.
    * :func:`~gpytorch.lazy.LazyTensor._size`, which returns a :class:`torch.Size` containing the dimensions of
      :math:`K`.

    In addition to these, a LazyTensor may need to define the :func:`~gpytorch.lazy.LazyTensor._transpose_nonbatch`,
    :func:`~gpytorch.lazy.LazyTensor._get_indices`, and :func:`~gpytorch.lazy.LazyTensor._batch_get_indices`
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

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def _args(self):
        return self._args_memo

    @_args.setter
    def _args(self, args):
        self._args_memo = args

    def _preconditioner(self):
        """
        (Optional) define a preconditioner (P) for linear conjugate gradients

        Returns:
            function: a function on x which performs P^{-1}(x)
            scalar: the log determinant of P
        """
        return None, None

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

    def _getitem_nonbatch(self, row_index, col_index):
        """
        Given an index over rows and columns,
        Returns a LazyTensor with those rows and columns selected

        Implementing this is not necessary, but it improves performance

        Args:
            row_index (slice or LongTensor): index over rows
            col_index (slice or LongTensor): index over columns
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

        left_interp_indices = left_interp_indices.expand(*(batch_sizes + [left_interp_len, 1]))
        left_interp_values = torch.ones(left_interp_indices.size(), dtype=self.dtype, device=self.device)
        right_interp_indices = right_interp_indices.expand(*(batch_sizes + [right_interp_len, 1]))
        right_interp_values = torch.ones(right_interp_indices.size(), dtype=self.dtype, device=self.device)

        res = InterpolatedLazyTensor(
            self, left_interp_indices, left_interp_values, right_interp_indices, right_interp_values
        )
        return res

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
        raise NotImplementedError("The class %s requires a _matmul function!" % self.__class__.__name__)

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
        raise NotImplementedError("The class %s requires a _quad_form_derivative function!" % self.__class__.__name__)

    def _size(self):
        """
        Returns the size of the resulting Tensor that the lazy tensor represents.

        ..note::
            This method is used internally by the related function :func:`~gpytorch.lazy.LazyTensor.size`, which does
            some additional work. Calling this method directly is discouraged.

        Returns:
            :obj:`torch.Size`: The size of the matrix :math:`K` represented by this LazyTensor
        """
        raise NotImplementedError("The class %s requires a _size function!" % self.__class__.__name__)

    def _transpose_nonbatch(self):
        """
        Transposes non-batch dimensions (e.g. last two)

        Implement this method, rather than transpose() or t().
        This is because size does some additional work
        """
        raise NotImplementedError("The class %s requires a _transpose_nonbatch function!" % self.__class__.__name__)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        """
        This function is necessary if you're using a LazyTensor as part of
        and InterpolatedLazyTensor

        Batch version of _get_indices
        For each matrix in batch_indices, returns entries of the matrix,
        indexed by left and right indices
        Only works for batch lazy tensors
        """
        raise NotImplementedError("The class %s requires a _batch_get_indices function!" % self.__class__.__name__)

    def _get_indices(self, left_indices, right_indices):
        """
        This function is necessary if you're using a LazyTensor as part of
        and InterpolatedLazyTensor

        Returns entries of the matrix, indexed by left and right indices
        Only works for non-batch lazy tensors
        """
        raise NotImplementedError("The class %s requires a _get_indices function!" % self.__class__.__name__)

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
        if self.ndimension() == 3:
            diag_lazy_tsr = DiagLazyTensor(diag.unsqueeze(0).expand(self.size(0), self.size(1)))
            return AddedDiagLazyTensor(self, diag_lazy_tsr)
        else:
            diag_lazy_tsr = DiagLazyTensor(diag.expand(self.size(0)))
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

    @property
    def device(self):
        return self._args[0].device

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
            return self._batch_get_indices(batch_iter, row_col_iter, row_col_iter).view(size[0], size[1])
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

    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyTensor represents. This function
        should return a Tensor storing an exact representation of this LazyTensor.
        """
        size = self.size()
        if len(size) == 2:
            batch_mode = False
            n_rows, n_cols = size
        else:
            batch_mode = True
            batch_size, n_rows, n_cols = size

        if n_rows < n_cols:
            eye = torch.eye(n_rows, dtype=self.dtype, device=self.device)
            if batch_mode:
                eye = eye.unsqueeze(0).expand(batch_size, n_rows, n_rows)
                return self.transpose(1, 2).matmul(eye).transpose(1, 2).contiguous()
            else:
                return self.t().matmul(eye).t().contiguous()
        else:
            eye = torch.eye(n_cols, dtype=self.dtype, device=self.device)
            if batch_mode:
                eye = eye.unsqueeze(0).expand(batch_size, n_cols, n_cols)
            return self.matmul(eye)

    def evaluate_kernel(self):
        """
        Return a new LazyTensor representing the same one as this one, but with
        all lazily evaluated kernels actually evaluated.
        """
        return self.representation_tree()(*self.representation())

    def exact_predictive_mean(self, full_mean, train_labels, n_train, likelihood, precomputed_cache=None):
        """
        Computes the posterior predictive covariance of a GP
        Assumes that self is the block prior covariance matrix of training and testing points
        [ K_XX, K_XX*; K_X*X, K_X*X* ]

        Args:
            full_mean (:obj:`torch.tensor`): the training and test prior means, stacked on top of each other
            train_labels (:obj:`torch.tensor`): the training labels minus the training prior mean
            noise (:obj:`torch.tensor`): the observed noise (from the likelihood)
            precomputed_cache (optional): speeds up subsequent computations (default: None)

        Returns:
            :obj:`torch.tensor`: The predictive posterior mean of the test points
        """
        from ..random_variables import GaussianRandomVariable

        if precomputed_cache is None:
            train_mean = full_mean.narrow(-1, 0, n_train)
            if self.ndimension() == 3:
                train_train_covar = self[:, :n_train, :n_train]
            else:
                train_train_covar = self[:n_train, :n_train]

            train_mean = full_mean.narrow(-1, 0, train_train_covar.size(-1))
            grv = GaussianRandomVariable(train_mean, train_train_covar)
            train_mean, train_train_covar = likelihood(grv).representation()

            train_labels_offset = train_labels - train_mean
            if self.ndimension() == 3:
                train_labels_offset = train_labels_offset.unsqueeze(-1)
            precomputed_cache = train_train_covar.inv_matmul(train_labels_offset)

        test_mean = full_mean.narrow(-1, train_labels.size(-1), full_mean.size(-1) - train_labels.size(-1))
        if self.ndimension() == 3:
            test_train_covar = self[:, n_train:, :n_train]
        else:
            test_train_covar = self[n_train:, :n_train]
        res = test_train_covar.matmul(precomputed_cache)
        if res.ndimension() == 3:
            res = res.squeeze(-1)
        res = res + test_mean
        return res, precomputed_cache.detach()

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        """
        Computes a cache for K_X*X (K_XX + sigma^2 I)^-1 K_X*X if possible. By default, this does no work and returns
        the first argument.

        Args:
            train_train_covar_inv_root (:obj:`torch.tensor`): a root of (K_XX + sigma^2 I)^-1
            test_train_covar (:obj:`torch.tensor`): the observed noise (from the likelihood)

        Returns
            - A precomputed cache
        """
        return train_train_covar_inv_root.detach()

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        """
        Computes :math:`K_{X^{*}X} S` given a precomputed cache
        Where :math:`S` is a tensor such that :math:`SS^{\\top} = (K_{XX} + \sigma^2 I)^{-1}`

        Args:
            precomputed_cache (:obj:`torch.tensor`): What was computed in _exact_predictive_covar_inv_quad_form_cache
            test_train_covar (:obj:`torch.tensor`): The observed noise (from the likelihood)

        Returns
            :obj:`~gpytorch.lazy.LazyTensor`: :math:`K_{X^{*}X} S`
        """
        # Here the precomputed cache represents S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        return test_train_covar.matmul(precomputed_cache)

    def exact_predictive_covar(self, n_train, likelihood, precomputed_cache=None):
        """
        Computes the posterior predictive covariance of a GP
        Assumes that self is the block prior covariance matrix of training and testing points
        [ K_XX, K_XX*; K_X*X, K_X*X* ]

        Args:
            n_train (int): The number of training points in the full covariance matrix
            noise (scalar): The observed noise (from the likelihood)
            precomputed_cache (optional): speeds up subsequent computations (default: None)

        Returns:
            :obj:`gpytorch.lazy.LazyTensor`: A LazyTensor representing the predictive posterior covariance of the
                                               test points
        """
        from ..random_variables import GaussianRandomVariable

        if self.ndimension() == 3:
            train_train_covar = self[:, :n_train, :n_train]
            test_train_covar = self[:, n_train:, :n_train]
            test_test_covar = self[:, n_train:, n_train:]
        else:
            train_train_covar = self[:n_train, :n_train]
            test_train_covar = self[n_train:, :n_train]
            test_test_covar = self[n_train:, n_train:]

        train_train_covar = likelihood(GaussianRandomVariable(torch.zeros(1), train_train_covar)).covar()
        if not beta_features.fast_pred_var.on():
            from .matmul_lazy_tensor import MatmulLazyTensor

            test_train_covar = test_train_covar.evaluate()
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar).mul(-1)
            res = test_test_covar + MatmulLazyTensor(test_train_covar, covar_correction_rhs)
            return res, None

        if precomputed_cache is None:
            train_train_covar_inv_root = train_train_covar.root_inv_decomposition()
            precomputed_cache = self._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar
            )

        from .root_lazy_tensor import RootLazyTensor

        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache, test_train_covar)
        res = test_test_covar + RootLazyTensor(covar_inv_quad_form_root).mul(-1)
        return res, precomputed_cache

    def inv_matmul(self, tensor):
        """
        Computes a linear solve (w.r.t self = :math:`K`) with several right hand sides :math:`M`.

        Args:
            - :obj:`torch.tensor` (n x k) - Matrix :math:`M` right hand sides

        Returns:
            - :obj:`torch.tensor` - :math:`K^{-1}M`
        """
        # Work out batch dimension, if necessary
        lazy_tsr = self
        if lazy_tsr.ndimension() == 3 and tensor.ndimension() == 3:
            if lazy_tsr.size(0) == 1 and tensor.size(0) > 1:
                lazy_tsr = lazy_tsr.repeat(tensor.size(0), 1, 1)
            elif tensor.size(0) == 1:
                tensor = tensor.expand(lazy_tsr.size(0), tensor.size(1), tensor.size(2))
        elif self.ndimension() > 3 or tensor.ndimension() > 3:
            raise RuntimeError

        func = InvMatmul(self.representation_tree(), preconditioner=self._preconditioner()[0])
        return func(tensor, *self.representation())

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

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False):
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
        # Work out batch dimension, if necessary
        lazy_tsr = self
        if inv_quad_rhs is not None:
            if lazy_tsr.ndimension() == 3 and inv_quad_rhs.ndimension() == 3:
                if lazy_tsr.size(0) == 1 and inv_quad_rhs.size(0) > 1:
                    lazy_tsr = lazy_tsr.repeat(inv_quad_rhs.size(0), 1, 1)
                elif inv_quad_rhs.size(0) == 1:
                    inv_quad_rhs = inv_quad_rhs.expand(lazy_tsr.size(0), inv_quad_rhs.size(1), inv_quad_rhs.size(2))
            elif self.ndimension() > 3 or inv_quad_rhs.ndimension() > 3:
                raise RuntimeError

        matrix_size = self.size(-1)
        batch_size = self.size(0) if self.ndimension() == 3 else None

        args = lazy_tsr.representation()
        if inv_quad_rhs is not None:
            args = [inv_quad_rhs] + list(args)

        res = InvQuadLogDet(
            representation_tree=self.representation_tree(),
            matrix_size=matrix_size,
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device,
            inv_quad=(inv_quad_rhs is not None),
            log_det=log_det,
            preconditioner=self._preconditioner()[0],
            log_det_correction=self._preconditioner()[1],
        )(*args)
        return res

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

    def matmul(self, tensor):
        """
        Multiplies self by a matrix

        Args:
            tensor (:obj:`torch.tensor`): Matrix or vector to multiply with. Must be a proper `:obj:`torch.tensor`,
            and not another :obj:`gpytorch.lazy.LazyTensor`.

        Returns:
            :obj:`torch.tensor`: Tensor containing the result of the matrix multiplication :math:`KM`, where :math:`K`
            is the matrix that this :obj:`gpytorch.lazy.LazyTensor` represents, and :math:`M` is the matrix input
            to this method.
        """

        # Work out batch dimension, if necessary
        lazy_tsr = self
        if lazy_tsr.ndimension() == 3 and tensor.ndimension() == 3:
            if lazy_tsr.size(0) == 1 and tensor.size(0) > 1:
                lazy_tsr = lazy_tsr.repeat(tensor.size(0), 1, 1)
            elif tensor.size(0) == 1:
                tensor = tensor.expand(lazy_tsr.size(0), tensor.size(1), tensor.size(2))
        elif self.ndimension() > 3 or tensor.ndimension() > 3:
            raise RuntimeError

        func = Matmul(self.representation_tree())
        return func(tensor, *self.representation())

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

        roots = self.root_decomposition()
        n_batch = roots.size(0) if mul_batch_size is None else mul_batch_size
        true_batch_size = roots.size(0) // mul_batch_size if mul_batch_size is not None else 1

        while True:
            roots = roots.view(true_batch_size, n_batch, roots.size(1), roots.size(2))

            # Take care of extra roots (odd roots), if they exist
            if n_batch % 2:
                extra_root = (
                    torch.randn(roots.size(0), 1, roots.size(2), roots.size(3), dtype=roots.dtype, device=roots.device)
                    .mul_(1e-6 / math.sqrt(roots.size(3)))
                    .add_(1. / math.sqrt(roots.size(3)))
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
                roots = res.root_decomposition()
                n_batch = n_batch // 2

        return res

    def ndimension(self):
        """
        Returns the number of dimensions
        """
        return len(self.size())

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

    def root_decomposition(self):
        """
        Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
        This can be used for sampling from a Gaussian distribution, or for obtaining a
        low-rank version of a matrix
        """
        batch_size = self.size(0) if self.ndimension() == 3 else None
        res, _ = RootDecomposition(
            self.representation_tree(),
            dtype=self.dtype,
            device=self.device,
            size=self.size(-1),
            max_iter=self.root_decomposition_size(),
            batch_size=batch_size,
        )(*self.representation())
        return res

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        """
        Returns a (usually low-rank) root decomposotion lazy tensor of a PSD matrix.
        This can be used for sampling from a Gaussian distribution, or for obtaining a
        low-rank version of a matrix
        """
        if initial_vectors is not None:
            if self.ndimension() == 3:
                if initial_vectors.ndimension() == 2:
                    initial_vectors = initial_vectors.unsqueeze(-1)
            else:
                if initial_vectors.ndimension() == 1:
                    initial_vectors = initial_vectors.unsqueeze(-1)
            if initial_vectors.size(-1) > 1 and test_vectors is None:
                raise RuntimeError("You must supply test vectors if you supply more than one " "initial vector")

        batch_size = self.size(0) if self.ndimension() == 3 else None
        roots, inv_roots = RootDecomposition(
            self.representation_tree(),
            dtype=self.dtype,
            device=self.device,
            size=self.size(-1),
            max_iter=self.root_decomposition_size(),
            root=True,
            inverse=True,
            batch_size=batch_size,
        )(*self.representation())

        # Choose the best of the inv_roots, if there were more than one initial vectors
        if initial_vectors is not None and initial_vectors.size(-1) > 1:
            n_probes = initial_vectors.size(-1)
            n_dim = self.size(-1)
            test_vectors = test_vectors.unsqueeze(0)

            # Compute solves
            solves = inv_roots.matmul(inv_roots.transpose(-1, -2).matmul(test_vectors))

            # Compute self * solves
            if batch_size is None:
                solves = solves.permute(1, 2, 0).contiguous().view(n_dim, -1)
                mat_times_solves = self.matmul(solves)
                mat_times_solves = mat_times_solves.view(n_dim, -1, n_probes).permute(2, 0, 1)
            else:
                solves = solves.permute(1, 2, 3, 0).contiguous().view(batch_size, n_dim, -1)
                mat_times_solves = self.matmul(solves)
                mat_times_solves = mat_times_solves.view(batch_size, n_dim, -1, n_probes).permute(3, 0, 1, 2)

            # Compute residuals
            residuals = (mat_times_solves - test_vectors).norm(2, dim=-2)
            if batch_size is None:
                residuals = residuals.sum(-1)
            else:
                residuals = residuals.sum(-1).sum(-1)

            # Choose solve that best fits
            _, best_solve_index = residuals.min(0)
            inv_root = inv_roots[best_solve_index].squeeze(0)

        else:
            inv_root = inv_roots

        return inv_root

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

    def t(self):
        """
        Alias of :meth:`~gpytorch.lazy.LazyTensor.transpose` for 2D LazyTensor.
        (Tranposes the two dimensions.)
        """
        if self.ndimension() != 2:
            raise RuntimeError("Cannot call t for more than 2 dimensions")
        return self.transpose(0, 1)

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
            covar_root = self.root_decomposition()

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

        if isinstance(other, ZeroLazyTensor):
            return self

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

        return self.mul(1. / other)

    def __mul__(self, other):
        """
        Convenience alias of :meth:`~gpytorch.lazy.LazyTensor.mul` that allows the standard product operator to be
        used.
        """
        from .zero_lazy_tensor import ZeroLazyTensor

        if isinstance(other, ZeroLazyTensor):
            return other

        return self.mul(other)

    def __getitem__(self, index):
        """
        Supports subindexing of the matrix this LazyTensor represents. This may return either another
        :obj:`gpytorch.lazy.LazyTensor` or a :obj:`torch.tensor` depending on the exact implementation.
        """
        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))
        components = list(self._args)

        squeeze_left = False
        squeeze_right = False
        if isinstance(index[-2], int):
            index[-2] = slice(index[-2], index[-2] + 1, None)
            squeeze_left = True
        if isinstance(index[-1], int):
            index[-1] = slice(index[-1], index[-1] + 1, None)
            squeeze_right = True

        # Handle batch dimensions
        isbatch = ndimension >= 3
        if isbatch:
            batch_index = tuple(index[:-2])
            for i, item in enumerate(components):
                components[i] = item[batch_index]

        new_lazy_tensor = self.__class__(*components, **self._kwargs)

        # Handle index
        left_index = index[-2]
        right_index = index[-1]

        if (
            not torch.is_tensor(left_index)
            and left_index == slice(None, None, None)
            and not torch.is_tensor(right_index)
            and right_index == slice(None, None, None)
        ):
            return new_lazy_tensor

        res = new_lazy_tensor._getitem_nonbatch(left_index, right_index)
        if squeeze_left or squeeze_right:
            res = res.evaluate()
            if squeeze_left:
                res = res.squeeze(-2)
            if squeeze_right:
                res = res.squeeze(-1)

        return res

    def __setattr__(self, name, val):
        if torch.is_tensor(val) or isinstance(val, LazyTensor):
            if not hasattr(self, "_args"):
                raise RuntimeError(
                    "Cannot assign {name} to LazyTensor before calling " "LazyTensor.__init__()".format(name=name)
                )
        object.__setattr__(self, name, val)


def _import_dotted_name(name):
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj
