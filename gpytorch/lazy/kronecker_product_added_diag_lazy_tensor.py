#!/usr/bin/env python3

import torch

from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import ConstantDiagLazyTensor, DiagLazyTensor
from .kronecker_product_lazy_tensor import KroneckerProductDiagLazyTensor, KroneckerProductLazyTensor
from .matmul_lazy_tensor import MatmulLazyTensor


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors, preconditioner_override=None):
        super().__init__(*lazy_tensors, preconditioner_override=preconditioner_override)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self.diag_tensor = lazy_tensors[0]
            self.lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self.diag_tensor = lazy_tensors[1]
            self.lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")
        self._diag_is_constant = isinstance(self.diag_tensor, ConstantDiagLazyTensor)

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        if self._diag_is_constant:
            # we want to call the standard InvQuadLogDet to easily get the probe vectors and do the
            # solve but we only want to cache the probe vectors for the backwards
            inv_quad_term, _ = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=False, reduce_inv_quad=reduce_inv_quad
            )
            logdet_term = self._logdet() if logdet else None
            return inv_quad_term, logdet_term
        return super().inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad)

    def _logdet(self):
        if self._diag_is_constant:
            # symeig requires computing the eigenvectors so that it's differentiable
            evals, _ = self.lazy_tensor.symeig(eigenvectors=True)
            evals_plus_diag = evals + self.diag_tensor.diag()
            return torch.log(evals_plus_diag).sum(dim=-1)
        return super()._logdet()

    def _preconditioner(self):
        # solves don't use CG so don't waste time computing it
        return None, None, None

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):
        rhs_dtype = rhs.dtype
        if self._diag_is_constant:
            # we can perform the solve using the Kronecker-structured eigendecomposition
            # we do the solve in double for numerical stability issues
            # TODO: Use fp64 registry once #1213 is addressed
            evals, q_matrix = self.lazy_tensor.to(torch.double).symeig(eigenvectors=True)

            evals_plus_diagonal = evals + self.diag_tensor.diag().double()
            evals_root = evals_plus_diagonal.pow(0.5)
            inv_mat_sqrt = DiagLazyTensor(evals_root.reciprocal())

            res = q_matrix.transpose(-2, -1).matmul(rhs.double())
            res2 = inv_mat_sqrt.matmul(res)

            lazy_lhs = q_matrix.matmul(inv_mat_sqrt)
            return lazy_lhs.matmul(res2).type(rhs_dtype)

        if isinstance(self.diag_tensor, KroneckerProductDiagLazyTensor):
            # If the diagonal has the same Kronecker structure as the full matrix, with each factor being
            # constatnt, wee can perform the solve efficiently by exploiting the Woodbury matrix identity
            if len(self.lazy_tensor.lazy_tensors) == len(self.diag_tensor.lazy_tensors) and all(
                tfull.shape == tdiag.shape and isinstance(tdiag, ConstantDiagLazyTensor)
                for tfull, tdiag in zip(self.lazy_tensor.lazy_tensors, self.diag_tensor.lazy_tensors)
            ):
                # again we do the solve in double precision
                # TODO: Use fp64 registry once #1213 is addressed
                rhs = rhs.double()
                lt = self.lazy_tensor.to(torch.double)
                dlt = self.diag_tensor.to(torch.double)

                # We have
                #   (K + D)^{-1} = K^{-1} - K^{-1} (K D^{-1} + I)^{-1}
                #                = K^{-1} - K^{-1} (\kron_i{K_i D_i^{-1}} + I)^{-1}
                #
                # and so with an eigendecomposition \kron_i{K_i D_i^{-1}} = S Lambda S, we can solve (K + D) = b as
                # K^{-1} b - K^{-1} (S (Lambda + I)^{-1} S^T b).
                # Each sub-matrix D_i^{-1} has constant diagonal, so we may scale the eigenvalues of the
                # eigendecomposition of K_i by its inverse to get an eigendecomposition of K_i D_i^{-1}.
                sub_evals, sub_evecs = [], []
                for lt_, dlt_ in zip(lt.lazy_tensors, dlt.lazy_tensors):
                    evals_, evecs_ = lt_.symeig(eigenvectors=True)
                    sub_evals.append(DiagLazyTensor(evals_ / dlt_.diag_values))
                    sub_evecs.append(evecs_)
                Lambda_I = KroneckerProductDiagLazyTensor(*sub_evals).add_jitter(1.0)
                S = KroneckerProductLazyTensor(*sub_evecs)
                tmp_term = S.matmul(Lambda_I.inv_matmul(S._transpose_nonbatch().matmul(rhs)))
                res = lt._inv_matmul(rhs - tmp_term)
                return res.to(rhs_dtype)

        # in all other cases we fall back to the default
        super()._solve(rhs, preconditioner=preconditioner, num_tridiag=num_tridiag)

    def _root_decomposition(self):
        if self._diag_is_constant:
            evals, q_matrix = self.lazy_tensor.symeig(eigenvectors=True)
            updated_evals = DiagLazyTensor((evals + self.diag_tensor.diag()).pow(0.5))
            return MatmulLazyTensor(q_matrix, updated_evals)
        super()._root_decomposition()

    def _root_inv_decomposition(self, initial_vectors=None):
        if self._diag_is_constant:
            evals, q_matrix = self.lazy_tensor.symeig(eigenvectors=True)
            inv_sqrt_evals = DiagLazyTensor((evals + self.diag_tensor.diag()).pow(-0.5))
            return MatmulLazyTensor(q_matrix, inv_sqrt_evals)
        super()._root_inv_decomposition(initial_vectors=initial_vectors)
