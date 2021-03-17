#!/usr/bin/env python3

import operator
from functools import reduce
from typing import Optional, Tuple

import torch
from torch import Tensor

from .. import settings
from ..utils.broadcasting import _matmul_broadcast_shape, _mul_broadcast_shape
from ..utils.memoize import cached
from .diag_lazy_tensor import ConstantDiagLazyTensor, DiagLazyTensor
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import lazify
from .triangular_lazy_tensor import TriangularLazyTensor


def _kron_diag(*lts) -> Tensor:
    """Compute diagonal of a KroneckerProductLazyTensor from the diagonals of the constituiting tensors"""
    lead_diag = lts[0].diag()
    if len(lts) == 1:  # base case:
        return lead_diag
    trail_diag = _kron_diag(*lts[1:])
    diag = lead_diag.unsqueeze(-2) * trail_diag.unsqueeze(-1)
    return diag.transpose(-1, -2).reshape(*diag.shape[:-2], -1)


def _prod(iterable):
    return reduce(operator.mul, iterable, 1)


def _matmul(lazy_tensors, kp_shape, rhs):
    output_shape = _matmul_broadcast_shape(kp_shape, rhs.shape)
    output_batch_shape = output_shape[:-2]

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for lazy_tensor in lazy_tensors:
        res = res.view(*output_batch_shape, lazy_tensor.size(-1), -1)
        factor = lazy_tensor._matmul(res)
        factor = factor.view(*output_batch_shape, lazy_tensor.size(-2), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


def _t_matmul(lazy_tensors, kp_shape, rhs):
    kp_t_shape = (*kp_shape[:-2], kp_shape[-1], kp_shape[-2])
    output_shape = _matmul_broadcast_shape(kp_t_shape, rhs.shape)
    output_batch_shape = torch.Size(output_shape[:-2])

    res = rhs.contiguous().expand(*output_batch_shape, *rhs.shape[-2:])
    num_cols = rhs.size(-1)
    for lazy_tensor in lazy_tensors:
        res = res.view(*output_batch_shape, lazy_tensor.size(-2), -1)
        factor = lazy_tensor._t_matmul(res)
        factor = factor.view(*output_batch_shape, lazy_tensor.size(-1), -1, num_cols).transpose(-3, -2)
        res = factor.reshape(*output_batch_shape, -1, num_cols)
    return res


class KroneckerProductLazyTensor(LazyTensor):
    r"""
    Returns the Kronecker product of the given lazy tensors

    Args:
        :`lazy_tensors`: List of lazy tensors
    """

    def __init__(self, *lazy_tensors):
        try:
            lazy_tensors = tuple(lazify(lazy_tensor) for lazy_tensor in lazy_tensors)
        except TypeError:
            raise RuntimeError("KroneckerProductLazyTensor is intended to wrap lazy tensors.")
        for prev_lazy_tensor, curr_lazy_tensor in zip(lazy_tensors[:-1], lazy_tensors[1:]):
            if prev_lazy_tensor.batch_shape != curr_lazy_tensor.batch_shape:
                raise RuntimeError(
                    "KroneckerProductLazyTensor expects lazy tensors with the "
                    "same batch shapes. Got {}.".format([lv.batch_shape for lv in lazy_tensors])
                )
        super().__init__(*lazy_tensors)
        self.lazy_tensors = lazy_tensors

    def __add__(self, other):
        if isinstance(other, (KroneckerProductDiagLazyTensor, ConstantDiagLazyTensor)):
            from .kronecker_product_added_diag_lazy_tensor import KroneckerProductAddedDiagLazyTensor

            return KroneckerProductAddedDiagLazyTensor(self, other)
        if isinstance(other, KroneckerProductLazyTensor):
            from .sum_kronecker_lazy_tensor import SumKroneckerLazyTensor

            return SumKroneckerLazyTensor(self, other)
        if isinstance(other, DiagLazyTensor):
            return self.add_diag(other.diag())
        return super().__add__(other)

    def add_diag(self, diag):
        r"""
        Adds a diagonal to a KroneckerProductLazyTensor
        """

        from .kronecker_product_added_diag_lazy_tensor import KroneckerProductAddedDiagLazyTensor

        if not self.is_square:
            raise RuntimeError("add_diag only defined for square matrices")

        diag_shape = diag.shape
        if len(diag_shape) == 0:
            # interpret scalar tensor as constant diag
            diag_tensor = ConstantDiagLazyTensor(diag.unsqueeze(-1), diag_shape=self.shape[-1])
        elif diag_shape[-1] == 1:
            # interpret single-trailing element as constant diag
            diag_tensor = ConstantDiagLazyTensor(diag, diag_shape=self.shape[-1])
        else:
            try:
                expanded_diag = diag.expand(self.shape[:-1])
            except RuntimeError:
                raise RuntimeError(
                    "add_diag for LazyTensor of size {} received invalid diagonal of size {}.".format(
                        self.shape, diag_shape
                    )
                )
            diag_tensor = DiagLazyTensor(expanded_diag)

        return KroneckerProductAddedDiagLazyTensor(self, diag_tensor)

    def diag(self):
        r"""
        As :func:`torch.diag`, returns the diagonal of the matrix :math:`K` this LazyTensor represents as a vector.

        :rtype: torch.tensor
        :return: The diagonal of :math:`K`. If :math:`K` is :math:`n \times n`, this will be a length
            n vector. If this LazyTensor represents a batch (e.g., is :math:`b \times n \times n`), this will be a
            :math:`b \times n` matrix of diagonals, one for each matrix in the batch.
        """
        if settings.debug.on():
            if not self.is_square:
                raise RuntimeError("Diag works on square matrices (or batches)")
        return _kron_diag(*self.lazy_tensors)

    def diagonalization(self, method: Optional[str] = None):
        if method is None:
            method = "symeig"
        return super().diagonalization(method=method)

    @cached
    def inverse(self):
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        # TODO: Investigate under what conditions computing individual individual inverses makes sense
        inverses = [lt.inverse() for lt in self.lazy_tensors]
        return self.__class__(*inverses)

    def inv_matmul(self, right_tensor, left_tensor=None):
        # TODO: Investigate under what conditions computing individual inverses makes sense
        # For now, retain existing behavior
        return super().inv_matmul(right_tensor=right_tensor, left_tensor=left_tensor)

    @cached(name="cholesky")
    def _cholesky(self, upper=False):
        chol_factors = [lt.cholesky(upper=upper) for lt in self.lazy_tensors]
        return KroneckerProductTriangularLazyTensor(*chol_factors, upper=upper)

    def _expand_batch(self, batch_shape):
        return self.__class__(*[lazy_tensor._expand_batch(batch_shape) for lazy_tensor in self.lazy_tensors])

    def _get_indices(self, row_index, col_index, *batch_indices):
        row_factor = self.size(-2)
        col_factor = self.size(-1)

        res = None
        for lazy_tensor in self.lazy_tensors:
            sub_row_size = lazy_tensor.size(-2)
            sub_col_size = lazy_tensor.size(-1)

            row_factor //= sub_row_size
            col_factor //= sub_col_size
            sub_res = lazy_tensor._get_indices(
                (row_index // row_factor).fmod(sub_row_size),
                (col_index // col_factor).fmod(sub_col_size),
                *batch_indices,
            )
            res = sub_res if res is None else (sub_res * res)

        return res

    def _inv_matmul(self, right_tensor, left_tensor=None):
        # Computes inv_matmul by exploiting the identity (A \kron B)^-1 = A^-1 \kron B^-1
        tsr_shapes = [q.size(-1) for q in self.lazy_tensors]
        n_rows = right_tensor.size(-2)
        batch_shape = _mul_broadcast_shape(self.shape[:-2], right_tensor.shape[:-2])
        perm_batch = tuple(range(len(batch_shape)))
        y = right_tensor.clone().expand(*batch_shape, *right_tensor.shape[-2:])
        for n, q in zip(tsr_shapes, self.lazy_tensors):
            # for KroneckerProductTriangularLazyTensor this inv_matmul is very cheap
            y = q.inv_matmul(y.reshape(*batch_shape, n, -1))
            y = y.reshape(*batch_shape, n, n_rows // n, -1).permute(*perm_batch, -2, -3, -1)
        res = y.reshape(*batch_shape, n_rows, -1)
        if left_tensor is not None:
            res = left_tensor @ res
        return res

    def _matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _matmul(self.lazy_tensors, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    @cached(name="root_decomposition")
    def root_decomposition(self, method: Optional[str] = None):
        from gpytorch.lazy import RootLazyTensor

        # return a dense root decomposition if the matrix is small
        if self.shape[-1] <= settings.max_cholesky_size.value():
            return super().root_decomposition(method=method)

        root_list = [lt.root_decomposition(method=method).root for lt in self.lazy_tensors]
        kronecker_root = KroneckerProductLazyTensor(*root_list)
        return RootLazyTensor(kronecker_root)

    @cached(name="root_inv_decomposition")
    def root_inv_decomposition(self, method=None, initial_vectors=None, test_vectors=None):
        from gpytorch.lazy import RootLazyTensor

        # return a dense root decomposition if the matrix is small
        if self.shape[-1] <= settings.max_cholesky_size.value():
            return super().root_inv_decomposition()

        root_list = [lt.root_inv_decomposition().root for lt in self.lazy_tensors]
        kronecker_root = KroneckerProductLazyTensor(*root_list)
        return RootLazyTensor(kronecker_root)

    @cached(name="size")
    def _size(self):
        left_size = _prod(lazy_tensor.size(-2) for lazy_tensor in self.lazy_tensors)
        right_size = _prod(lazy_tensor.size(-1) for lazy_tensor in self.lazy_tensors)
        return torch.Size((*self.lazy_tensors[0].batch_shape, left_size, right_size))

    @cached(name="svd")
    def _svd(self) -> Tuple[LazyTensor, Tensor, LazyTensor]:
        U, S, V = [], [], []
        for lt in self.lazy_tensors:
            U_, S_, V_ = lt.svd()
            U.append(U_)
            S.append(S_)
            V.append(V_)
        S = KroneckerProductLazyTensor(*[DiagLazyTensor(S_) for S_ in S]).diag()
        U = KroneckerProductLazyTensor(*U)
        V = KroneckerProductLazyTensor(*V)
        return U, S, V

    def _symeig(
        self, eigenvectors: bool = False, return_evals_as_lazy: bool = False
    ) -> Tuple[Tensor, Optional[LazyTensor]]:
        # return_evals_as_lazy is a flag to return the eigenvalues as a lazy tensor
        # which is useful for root decompositions here (see the root_decomposition
        # method above)
        evals, evecs = [], []
        for lt in self.lazy_tensors:
            evals_, evecs_ = lt.symeig(eigenvectors=eigenvectors)
            evals.append(evals_)
            evecs.append(evecs_)
        evals = KroneckerProductDiagLazyTensor(*[DiagLazyTensor(evals_) for evals_ in evals])

        if not return_evals_as_lazy:
            evals = evals.diag()

        if eigenvectors:
            evecs = KroneckerProductLazyTensor(*evecs)
        else:
            evecs = None
        return evals, evecs

    def _t_matmul(self, rhs):
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)

        res = _t_matmul(self.lazy_tensors, self.shape, rhs.contiguous())

        if is_vec:
            res = res.squeeze(-1)
        return res

    def _transpose_nonbatch(self):
        return self.__class__(*(lazy_tensor._transpose_nonbatch() for lazy_tensor in self.lazy_tensors), **self._kwargs)


class KroneckerProductTriangularLazyTensor(KroneckerProductLazyTensor):
    def __init__(self, *lazy_tensors, upper=False):
        if not all(isinstance(lt, TriangularLazyTensor) for lt in lazy_tensors):
            raise RuntimeError("Components of KroneckerProductTriangularLazyTensor must be TriangularLazyTensor.")
        super().__init__(*lazy_tensors)
        self.upper = upper

    @cached
    def inverse(self):
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        inverses = [lt.inverse() for lt in self.lazy_tensors]
        return self.__class__(*inverses, upper=self.upper)

    def inv_matmul(self, right_tensor, left_tensor=None):
        # For triangular components, using triangular-triangular substition should generally be good
        return self._inv_matmul(right_tensor=right_tensor, left_tensor=left_tensor)

    @cached(name="cholesky")
    def _cholesky(self, upper=False):
        raise NotImplementedError("_cholesky not applicable to triangular lazy tensors")

    def _cholesky_solve(self, rhs, upper=False):
        if upper:
            # res = (U.T @ U)^-1 @ v = U^-1 @ U^-T @ v
            w = self._transpose_nonbatch().inv_matmul(rhs)
            res = self.inv_matmul(w)
        else:
            # res = (L @ L.T)^-1 @ v = L^-T @ L^-1 @ v
            w = self.inv_matmul(rhs)
            res = self._transpose_nonbatch().inv_matmul(w)
        return res

    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LazyTensor]]:
        raise NotImplementedError("_symeig not applicable to triangular lazy tensors")


class KroneckerProductDiagLazyTensor(DiagLazyTensor, KroneckerProductTriangularLazyTensor):
    def __init__(self, *lazy_tensors):
        if not all(isinstance(lt, DiagLazyTensor) for lt in lazy_tensors):
            raise RuntimeError("Components of KroneckerProductDiagLazyTensor must be DiagLazyTensor.")
        super(KroneckerProductTriangularLazyTensor, self).__init__(*lazy_tensors)
        self.upper = False

    @cached(name="cholesky")
    def _cholesky(self, upper=False):
        chol_factors = [lt.cholesky(upper=upper) for lt in self.lazy_tensors]
        return KroneckerProductDiagLazyTensor(*chol_factors)

    @property
    def _diag(self):
        return _kron_diag(*self.lazy_tensors)

    def _expand_batch(self, batch_shape):
        return KroneckerProductTriangularLazyTensor._expand_batch(self, batch_shape)

    def _mul_constant(self, constant):
        return DiagLazyTensor(self._diag * constant.unsqueeze(-1))

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return KroneckerProductTriangularLazyTensor._quad_form_derivative(self, left_vecs, right_vecs)

    def sqrt(self):
        return self.__class__(*[lt.sqrt() for lt in self.lazy_tensors])

    def _symeig(
        self, eigenvectors: bool = False, return_evals_as_lazy: bool = False
    ) -> Tuple[Tensor, Optional[LazyTensor]]:
        # return_evals_as_lazy is a flag to return the eigenvalues as a lazy tensor
        # which is useful for root decompositions here (see the root_decomposition
        # method above)
        evals, evecs = [], []
        for lt in self.lazy_tensors:
            evals_, evecs_ = lt.symeig(eigenvectors=eigenvectors)
            evals.append(evals_)
            evecs.append(evecs_)
        evals = KroneckerProductDiagLazyTensor(*[DiagLazyTensor(evals_) for evals_ in evals])

        if not return_evals_as_lazy:
            evals = evals.diag()

        if eigenvectors:
            evecs = KroneckerProductDiagLazyTensor(*evecs)
        else:
            evecs = None
        return evals, evecs

    @cached
    def inverse(self):
        # here we use that (A \kron B)^-1 = A^-1 \kron B^-1
        inverses = [lt.inverse() for lt in self.lazy_tensors]
        return self.__class__(*inverses)
