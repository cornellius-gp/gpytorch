from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from torch.autograd import Variable
from ..functions._inv_matmul import InvMatmul
from ..functions._inv_quad_log_det import InvQuadLogDet
from ..functions._root_decomposition import RootDecomposition
from ..functions._matmul import Matmul
from .. import beta_features, settings
from .lazy_variable_representation_tree import LazyVariableRepresentationTree


class LazyVariable(object):

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
        - precond_fn (function) - a function on x which performs P^{-1}(x)
        - predond_log_det (scalar) - the log determinant of P
        """
        return None, None

    def _approx_diag(self):
        """
        (Optional) returns an (approximate) diagonal of the matrix

        Sometimes computing an exact diagonal is a bit computationally slow
        When we don't need an exact diagonal (e.g. for the pivoted cholesky decomposition,
        this function is called

        Defaults to calling the exact diagonal function

        Returns:
        - diag (tensor) - the diagonal (or batch of diagonals)
        """
        return self.diag()

    def _getitem_nonbatch(self, row_index, col_index):
        """
        Given an index over rows and columns,
        Returns a LazyVariable with those rows and columns selected

        Implementing this is not necessary, but it improves performance

        Args:
            - row_index (slice or LongTensor) - index over rows
            - col_index (slice or LongTensor) - index over columns
        """
        from .interpolated_lazy_variable import InterpolatedLazyVariable

        representation = self.representation()
        ndimension = self.ndimension()

        batch_sizes = list(self.size()[:-2])
        left_row_iter = representation[0].data.new(self.size()[-2]).long()
        right_row_iter = representation[0].data.new(self.size()[-1]).long()
        torch.arange(0, self.size()[-2], out=left_row_iter)
        torch.arange(0, self.size()[-1], out=right_row_iter)

        left_interp_indices = left_row_iter[row_index].unsqueeze(-1)
        right_interp_indices = right_row_iter[col_index].unsqueeze(-1)

        left_interp_len = len(left_interp_indices)
        right_interp_len = len(right_interp_indices)
        for _ in range(ndimension - 2):
            left_interp_indices.unsqueeze_(0)
            right_interp_indices.unsqueeze_(0)

        left_interp_indices = left_interp_indices.expand(
            *(batch_sizes + [left_interp_len, 1])
        )
        left_interp_values = self.tensor_cls(left_interp_indices.size()).fill_(1)
        right_interp_indices = right_interp_indices.expand(
            *(batch_sizes + [right_interp_len, 1])
        )
        right_interp_values = self.tensor_cls(right_interp_indices.size()).fill_(1)

        res = InterpolatedLazyVariable(
            self,
            Variable(left_interp_indices),
            Variable(left_interp_values),
            Variable(right_interp_indices),
            Variable(right_interp_values),
        )
        return res

    def _matmul(self, rhs):
        """
        Returns: matrix * rhs
        """
        raise NotImplementedError(
            "The class %s requires a _matmul function!" % self.__class__.__name__
        )

    def _t_matmul(self, rhs):
        """
        Returns: matrix^T * rhs

        Implementing this is not necessary, but it improves performance
        """
        return self.transpose(-1, -2)._matmul(rhs)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        """
        Given u (left_vecs) and v (right_vecs),
        Computes the derivatives of (u^t K v) w.r.t. K

        Returns: derivative w.r.t. the arguments
        """
        raise NotImplementedError(
            "The class %s requires a _quad_form_derivative function!"
            % self.__class__.__name__
        )

    def _size(self):
        """
        Returns the size of the resulting Variable that the lazy variable represents

        Implement this method, rather than size().
        This is because size does some additional work
        """
        raise NotImplementedError(
            "The class %s requires a _size function!" % self.__class__.__name__
        )

    def _transpose_nonbatch(self):
        """
        Transposes non-batch dimensions (e.g. last two)

        Implement this method, rather than transpose() or t().
        This is because size does some additional work
        """
        raise NotImplementedError(
            "The class %s requires a _transpose_nonbatch function!"
            % self.__class__.__name__
        )

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        """
        This function is necessary if you're using a LazyVariable as part of
        and InterpolatedLazyVariable

        Batch version of _get_indices
        For each matrix in batch_indices, returns entries of the matrix,
        indexed by left and right indices
        Only works for batch lazy variables
        """
        raise NotImplementedError(
            "The class %s requires a _batch_get_indices function!"
            % self.__class__.__name__
        )

    def _get_indices(self, left_indices, right_indices):
        """
        This function is necessary if you're using a LazyVariable as part of
        and InterpolatedLazyVariable

        Returns entries of the matrix, indexed by left and right indices
        Only works for non-batch lazy variables
        """
        raise NotImplementedError(
            "The class %s requires a _get_indices function!" % self.__class__.__name__
        )

    def add_diag(self, diag):
        """
        Adds an element to the diagonal of the matrix.

        Args:
            - diag (Scalar Variable)
        """
        from .diag_lazy_variable import DiagLazyVariable
        from .added_diag_lazy_variable import AddedDiagLazyVariable

        if self.size(-1) != self.size(-2):
            raise RuntimeError("add_diag only defined for square matrices")
        if self.ndimension() == 3:
            diag_lazy_var = DiagLazyVariable(
                diag.unsqueeze(0).expand(self.size(0), self.size(1))
            )
            return AddedDiagLazyVariable(self, diag_lazy_var)
        else:
            diag_lazy_var = DiagLazyVariable(diag.expand(self.size(0)))
            return AddedDiagLazyVariable(self, diag_lazy_var)

    def add_jitter(self):
        """
        Adds jitter (i.e., a small diagonal component) to the matrix this
        LazyVariable represents. This could potentially be implemented as a no-op,
        however this could lead to numerical instabilities, so this should only
        be done at the user's risk.
        """
        diag = Variable(self.tensor_cls(1).fill_(1e-3))
        return self.add_diag(diag)

    def cpu(self):
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

    def diag(self):
        size = self.size()
        if size[-1] != size[-2]:
            raise RuntimeError("Diag works on square matrices (or batches)")

        row_col_iter = Variable(self.tensor_cls(size[-1]).long())
        torch.arange(0, size[-1], out=row_col_iter.data)
        if self.ndimension() == 3:
            batch_iter = Variable(self.tensor_cls(size[0]).long())
            torch.arange(0, size[0], out=batch_iter.data)
            batch_iter = batch_iter.unsqueeze(1).repeat(1, size[1]).view(-1)
            row_col_iter = row_col_iter.unsqueeze(1).repeat(size[0], 1).view(-1)
            return self._batch_get_indices(batch_iter, row_col_iter, row_col_iter).view(
                size[0], size[1]
            )
        else:
            return self._get_indices(row_col_iter, row_col_iter)

    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyVariable represents. This
        function should return a Variable explicitly wrapping a Tensor storing
        an exact representation of this LazyVariable.
        """
        size = self.size()
        if len(size) == 2:
            batch_mode = False
            n_rows, n_cols = size
        else:
            batch_mode = True
            batch_size, n_rows, n_cols = size

        if n_rows < n_cols:
            eye = self.tensor_cls(n_rows).fill_(1).diag()
            if isinstance(self.representation()[0], Variable):
                eye = Variable(eye)
            if batch_mode:
                eye = eye.unsqueeze(0).expand(batch_size, n_rows, n_rows)
                return self.transpose(1, 2).matmul(eye).transpose(1, 2).contiguous()
            else:
                return self.t().matmul(eye).t().contiguous()
        else:
            eye = self.tensor_cls(n_cols).fill_(1).diag()
            if isinstance(self.representation()[0], Variable):
                eye = Variable(eye)
            if batch_mode:
                eye = eye.unsqueeze(0).expand(batch_size, n_cols, n_cols)
            return self.matmul(eye)

    def exact_predictive_mean(
        self, full_mean, train_labels, noise, precomputed_cache=None
    ):
        """
        Computes the posterior predictive covariance of a GP
        Assumes that self is the block prior covariance matrix of training and testing points
        [ K_XX, K_XX*; K_X*X, K_X*X* ]

        Args:
        - full_mean (n + t) - the training and test prior means, stacked on top of each other
        - train_labels (n) - the training labels minus the training prior mean
        - noise (1) - the observed noise (from the likelihood)
        - precomputed_cache - speeds up subsequent computations (default: None)

        Returns:
        - (t) - the predictive posterior mean of the test points
        """
        n_train = train_labels.size(0)
        if precomputed_cache is None:
            train_mean = full_mean[:n_train]
            train_train_covar = self[:n_train, :n_train].add_diag(noise)
            precomputed_cache = train_train_covar.inv_matmul(train_labels - train_mean)

        test_mean = full_mean[n_train:]
        test_train_covar = self[n_train:, :n_train]
        res = test_train_covar.matmul(precomputed_cache) + test_mean
        return res, precomputed_cache

    def _exact_predictive_covar_inv_quad_form_cache(
        self, train_train_covar_inv_root, test_train_covar
    ):
        """
        Computes a cache for K_X*X (K_XX + sigma^2 I)^-1 K_X*X

        Args:
        - train_train_covar_inv_root (n x k) - a root of (K_XX + sigma^2 I)^-1
        - test_train_covar (m x n) - the observed noise (from the likelihood)

        Returns
        - A precomputed cache
        """
        return train_train_covar_inv_root

    def _exact_predictive_covar_inv_quad_form_root(
        self, precomputed_cache, test_train_covar
    ):
        """
        Computes K_X*X S given a precomputed cache
        Where S is a tensor such that S S^T = (K_XX + sigma^2 I)^-1

        Args:
        - precomputed_cache - what was computed in _exact_predictive_covar_inv_quad_form_cache
        - test_train_covar (m x n) - the observed noise (from the likelihood)

        Returns
        - LazyVariable (m x k) - K_X^*X S
        """
        # Here the precomputed cache represents S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        return test_train_covar.matmul(precomputed_cache)

    def exact_predictive_covar(self, n_train, noise, precomputed_cache=None):
        """
        Computes the posterior predictive covariance of a GP
        Assumes that self is the block prior covariance matrix of training and testing points
        [ K_XX, K_XX*; K_X*X, K_X*X* ]

        Args:
        - n_train (int) - how many training points are there in the full covariance matrix
        - noise (1) - the observed noise (from the likelihood)
        - precomputed_cache - speeds up subsequent computations (default: None)

        Returns:
        - LazyVariable (t x t) - the predictive posterior covariance of the test points
        """
        if self.ndimension() == 3:
            train_train_covar = self[:, :n_train, :n_train].add_diag(noise)
            test_train_covar = self[:, n_train:, :n_train]
            test_test_covar = self[:, n_train:, n_train:]
        else:
            train_train_covar = self[:n_train, :n_train].add_diag(noise)
            test_train_covar = self[n_train:, :n_train]
            test_test_covar = self[n_train:, n_train:]

        if not beta_features.fast_pred_var.on():
            from .matmul_lazy_variable import MatmulLazyVariable

            test_train_covar = test_train_covar.evaluate()
            train_test_covar = test_train_covar.transpose(-1, -2)
            covar_correction_rhs = train_train_covar.inv_matmul(train_test_covar).mul(
                -1
            )
            res = test_test_covar + MatmulLazyVariable(
                test_train_covar, covar_correction_rhs
            )
            return res, None

        if precomputed_cache is None:
            train_train_covar_inv_root = train_train_covar.root_inv_decomposition()
            precomputed_cache = self._exact_predictive_covar_inv_quad_form_cache(
                train_train_covar_inv_root, test_train_covar
            )

        from .root_lazy_variable import RootLazyVariable

        covar_inv_quad_form_root = self._exact_predictive_covar_inv_quad_form_root(
            precomputed_cache, test_train_covar
        )
        res = test_test_covar + RootLazyVariable(covar_inv_quad_form_root).mul(-1)
        return res, precomputed_cache

    def inv_matmul(self, tensor):
        """
        Computes a linear solve (w.r.t self) with several right hand sides.

        Args:
            - tensor (tensor nxk) - Matrix or tensor

        Returns:
            - tensor - (self)^{-1} tensor
        """
        # Work out batch dimension, if necessary
        lazy_var = self
        if lazy_var.ndimension() == 3 and tensor.ndimension() == 3:
            if lazy_var.size(0) == 1 and tensor.size(0) > 1:
                lazy_var = lazy_var.repeat(tensor.size(0), 1, 1)
            elif tensor.size(0) == 1:
                tensor = tensor.expand(lazy_var.size(0), tensor.size(1), tensor.size(2))
        elif self.ndimension() > 3 or tensor.ndimension() > 3:
            raise RuntimeError

        func = InvMatmul(
            self.representation_tree(), preconditioner=self._preconditioner()[0]
        )
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
        lazy_var = self
        if inv_quad_rhs is not None:
            if lazy_var.ndimension() == 3 and inv_quad_rhs.ndimension() == 3:
                if lazy_var.size(0) == 1 and inv_quad_rhs.size(0) > 1:
                    lazy_var = lazy_var.repeat(inv_quad_rhs.size(0), 1, 1)
                elif inv_quad_rhs.size(0) == 1:
                    inv_quad_rhs = inv_quad_rhs.expand(
                        lazy_var.size(0), inv_quad_rhs.size(1), inv_quad_rhs.size(2)
                    )
            elif self.ndimension() > 3 or inv_quad_rhs.ndimension() > 3:
                raise RuntimeError

        matrix_size = self.size(-1)
        batch_size = self.size(0) if self.ndimension() == 3 else None
        tensor_cls = self.tensor_cls

        args = lazy_var.representation()
        if inv_quad_rhs is not None:
            args = [inv_quad_rhs] + list(args)

        res = InvQuadLogDet(
            representation_tree=self.representation_tree(),
            matrix_size=matrix_size,
            batch_size=batch_size,
            tensor_cls=tensor_cls,
            inv_quad=(inv_quad_rhs is not None),
            log_det=log_det,
            preconditioner=self._preconditioner()[0],
            log_det_correction=self._preconditioner()[1],
        )(
            *args
        )
        return res

    def log_det(self):
        """
        Computes an (approximate) log determinant of the matrix

        NOTE: Don't overwrite this function!
        Instead, overwrite inv_quad_log_det

        Returns:
            - scalar - log determinant
        """
        _, res = self.inv_quad_log_det(inv_quad_rhs=None, log_det=True)
        return res

    def matmul(self, tensor):
        """
        Multiplies self by a matrix

        Args:
            - tensor (matrix nxk) - Matrix or vector to multiply with

        Returns:
            - tensor
        """

        # Work out batch dimension, if necessary
        lazy_var = self
        if lazy_var.ndimension() == 3 and tensor.ndimension() == 3:
            if lazy_var.size(0) == 1 and tensor.size(0) > 1:
                lazy_var = lazy_var.repeat(tensor.size(0), 1, 1)
            elif tensor.size(0) == 1:
                tensor = tensor.expand(lazy_var.size(0), tensor.size(1), tensor.size(2))
        elif self.ndimension() > 3 or tensor.ndimension() > 3:
            raise RuntimeError

        func = Matmul(self.representation_tree())
        return func(tensor, *self.representation())

    def mul(self, other):
        """
        Multiplies the matrix by a constant, or elementwise the matrix by another matrix
        """
        if (
            not (isinstance(other, Variable) or isinstance(other, LazyVariable))
            or (isinstance(other, Variable) and other.numel() == 1)
        ):
            from .constant_mul_lazy_variable import ConstantMulLazyVariable

            return ConstantMulLazyVariable(self, other)
        else:
            from .mul_lazy_variable import MulLazyVariable

            return MulLazyVariable(self, other)

    def mul_batch(self, mul_batch_size=None):
        """
        """
        from .mul_lazy_variable import MulLazyVariable
        from .root_lazy_variable import RootLazyVariable

        if self.ndimension() < 3:
            raise RuntimeError("mul_batch only works with batched lazy variables")
        if self.size(0) == 1:
            return self.sum_batch()

        roots = self.root_decomposition()
        n_batch = roots.size(0) if mul_batch_size is None else mul_batch_size
        true_batch_size = roots.size(
            0
        ) // mul_batch_size if mul_batch_size is not None else 1

        while True:
            roots = roots.view(true_batch_size, n_batch, roots.size(1), roots.size(2))

            # Take care of extra roots (odd roots), if they exist
            if n_batch % 2:
                extra_root = Variable(
                    roots.data.new(roots.size(0), 1, roots.size(2), roots.size(3))
                )
                extra_root.data.normal_().mul_(1e-6 / math.sqrt(roots.size(3)))
                extra_root.data.add_(1. / math.sqrt(roots.size(3)))
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
                res = MulLazyVariable(RootLazyVariable(part1), RootLazyVariable(part2))
                break
            else:
                res = MulLazyVariable(RootLazyVariable(part1), RootLazyVariable(part2))
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
        Returns the variables that are used to define the LazyVariable
        """
        representation = []
        for arg in self._args:
            if isinstance(arg, Variable):
                representation.append(arg)
            elif isinstance(arg, LazyVariable):
                representation += list(arg.representation())
            else:
                raise RuntimeError(
                    "Representation of a LazyVariable should consist only of Variables"
                )
        return tuple(representation)

    def representation_tree(self):
        return LazyVariableRepresentationTree(self)

    def root_decomposition(self):
        """
        Returns a (usually low-rank) root decomposotion lazy variable of a PSD matrix.
        This can be used for sampling from a Gaussian distribution, or for obtaining a
        low-rank version of a matrix
        """
        batch_size = self.size(0) if self.ndimension() == 3 else None
        res, _ = RootDecomposition(
            self.representation_tree(),
            cls=self.tensor_cls,
            size=self.size(-1),
            max_iter=self.root_decomposition_size(),
            batch_size=batch_size,
        )(
            *self.representation()
        )
        return res

    def root_inv_decomposition(self, initial_vectors=None, test_vectors=None):
        """
        Returns a (usually low-rank) root decomposotion lazy variable of a PSD matrix.
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
                raise RuntimeError(
                    "You must supply test vectors if you supply more than one "
                    "initial vector"
                )

        batch_size = self.size(0) if self.ndimension() == 3 else None
        roots, inv_roots = RootDecomposition(
            self.representation_tree(),
            cls=self.tensor_cls,
            size=self.size(-1),
            max_iter=self.root_decomposition_size(),
            root=True,
            inverse=True,
            batch_size=batch_size,
        )(
            *self.representation()
        )

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
                mat_times_solves = mat_times_solves.view(n_dim, -1, n_probes).permute(
                    2, 0, 1
                )
            else:
                solves = solves.permute(1, 2, 3, 0).contiguous().view(
                    batch_size, n_dim, -1
                )
                mat_times_solves = self.matmul(solves)
                mat_times_solves = mat_times_solves.view(
                    batch_size, n_dim, -1, n_probes
                ).permute(
                    3, 0, 1, 2
                )

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
        Returns the size of the resulting Variable that the lazy variable represents
        """
        size = self._size()
        if val is not None:
            return size[val]
        return size

    def sum_batch(self, sum_batch_size=None):
        from .sum_batch_lazy_variable import SumBatchLazyVariable

        return SumBatchLazyVariable(self, sum_batch_size=sum_batch_size)

    def transpose(self, dim1, dim2):
        """
        Returns the transpose of the resulting Variable that the lazy variable represents
        """
        ndimension = self.ndimension()
        if dim1 < 0:
            dim1 = ndimension + dim1
        if dim2 < 0:
            dim2 = ndimension + dim2
        if (
            dim1 >= ndimension
            or dim2 >= ndimension
            or not isinstance(dim1, int)
            or not isinstance(dim2, int)
        ):
            raise RuntimeError("Invalid dimension")

        # Batch case
        if dim1 < ndimension - 2 and dim2 < ndimension - 2:
            res = self.__class__(
                *(arg.transpose(dim1, dim2) for arg in self._args), **self._kwargs
            )

        elif dim1 >= ndimension - 2 and dim2 >= ndimension - 2:
            res = self._transpose_nonbatch()

        else:
            raise RuntimeError(
                "Cannot transpose batch dimension with non-batch dimension"
            )

        return res

    def t(self):
        """
        Returns the transpose of the resulting Variable that the lazy variable represents
        """
        if self.ndimension() != 2:
            raise RuntimeError("Cannot call t for more than 2 dimensions")
        return self.transpose(0, 1)

    @property
    def tensor_cls(self):
        if not hasattr(self, "_tensor_cls"):
            first_item = self.representation()[0]
            if isinstance(first_item, Variable):
                first_item = first_item.data
            self._tensor_cls = _import_dotted_name(first_item.type())
        return self._tensor_cls

    def zero_mean_mvn_samples(self, n_samples):
        """
        Assumes that self is a covariance matrix, or a batch of covariance matrices.
        Returns samples from a zero-mean MVN, defined by self (as covariance matrix)

        Self should be symmetric, either (batch_size x n_dim x n_dim) or (n_dim x n_dim)

        Args:
        - n_samples: (int)

        Returns:
        - Samples from MVN (batch_size x n_samples)
        """
        covar_root = self.root_decomposition()
        if self.ndimension() == 3:
            base_samples = Variable(
                self.tensor_cls(self.size(0), covar_root.size(-1), n_samples).normal_()
            )
        else:
            base_samples = Variable(
                self.tensor_cls(covar_root.size(-1), n_samples).normal_()
            )
        samples = covar_root.matmul(base_samples)
        return samples

    def __add__(self, other):
        from .sum_lazy_variable import SumLazyVariable

        return SumLazyVariable(self, other)

    def __div__(self, other):
        return self.mul(1. / other)

    def __mul__(self, other):
        return self.mul(other)

    def __getitem__(self, index):
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

        new_lazy_variable = self.__class__(*components, **self._kwargs)

        # Handle index
        left_index = index[-2]
        right_index = index[-1]

        if (
            not torch.is_tensor(left_index)
            and left_index == slice(None, None, None)
            and not torch.is_tensor(right_index)
            and right_index == slice(None, None, None)
        ):
            return new_lazy_variable

        res = new_lazy_variable._getitem_nonbatch(left_index, right_index)
        if squeeze_left or squeeze_right:
            res = res.evaluate()
            if squeeze_left:
                res = res.squeeze(-2)
            if squeeze_right:
                res = res.squeeze(-1)

        return res

    def __setattr__(self, name, val):
        if (
            torch.is_tensor(val)
            or isinstance(val, Variable)
            or isinstance(val, LazyVariable)
        ):
            if not hasattr(self, "_args"):
                raise RuntimeError(
                    "Cannot assign {name} to LazyVariable before calling "
                    "LazyVariable.__init__()".format(name=name)
                )
        object.__setattr__(self, name, val)


def _import_dotted_name(name):
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj
