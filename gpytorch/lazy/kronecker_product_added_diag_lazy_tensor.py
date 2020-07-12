#!/usr/bin/env python3

import torch

from ..settings import skip_logdet_forward
from ..utils.memoize import get_from_cache
from .added_diag_lazy_tensor import AddedDiagLazyTensor
from .diag_lazy_tensor import DiagLazyTensor


class KroneckerProductAddedDiagLazyTensor(AddedDiagLazyTensor):
    def __init__(self, *lazy_tensors, preconditioner_override=None):
        # TODO: implement the woodbury formula for diagonal tensors that are non constants.

        super(KroneckerProductAddedDiagLazyTensor, self).__init__(
            *lazy_tensors, preconditioner_override=preconditioner_override
        )
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

    def inv_quad_logdet(self, inv_quad_rhs=None, logdet=False, reduce_inv_quad=True):
        # we want to call the standard InvQuadLogDet to easily get the probe vectors and do the
        # solve but we only want to cache the probe vectors for the backwards
        with skip_logdet_forward(True):
            inv_quad_term, func_logdet_term = super().inv_quad_logdet(
                inv_quad_rhs=inv_quad_rhs, logdet=logdet, reduce_inv_quad=reduce_inv_quad
            )

        if logdet is not None:
            if skip_logdet_forward.off():
                # we use the InvQuadLogDet backwards call to get the gradient
                logdet_term = self._logdet().detach()
                logdet_term = logdet_term + func_logdet_term
            else:
                logdet_term = func_logdet_term
        else:
            logdet_term = None

        return inv_quad_term, logdet_term

    def _logdet(self):
        try:
            evals, _ = get_from_cache(self.lazy_tensor, "symeig")
        except RuntimeError:
            evals, _ = self.lazy_tensor.symeig(eigenvectors=False)
        evals_plus_diag = evals + self.diag_tensor.diag()
        return torch.log(evals_plus_diag).sum(dim=-1)

    def _preconditioner(self):
        # solves don't use CG so don't waste time computing it
        return None, None, None

    def _solve(self, rhs, preconditioner=None, num_tridiag=0):
        # we do the solve in double for numerical stability issues
        # TODO: Use fp64 registry once #1213 is addressed

        rhs_dtype = rhs.dtype
        rhs = rhs.double()

        evals, q_matrix = self.lazy_tensor.symeig(eigenvectors=True)
        evals, q_matrix = evals.double(), q_matrix.double()

        evals_plus_diagonal = evals + self.diag_tensor.diag()
        evals_root = evals_plus_diagonal.pow(0.5)
        inv_mat_sqrt = DiagLazyTensor(evals_root.reciprocal())

        res = q_matrix.transpose(-2, -1).matmul(rhs)
        res2 = inv_mat_sqrt.matmul(res)

        lazy_lhs = q_matrix.matmul(inv_mat_sqrt)
        return lazy_lhs.matmul(res2).type(rhs_dtype)

    def _root_decomposition(self):
        evals, q_matrix = self.lazy_tensor.symeig(eigenvectors=True)
        updated_evals = DiagLazyTensor((evals + self.diag_tensor.diag()).pow(0.5))
        matrix_root = q_matrix.matmul(updated_evals)
        return matrix_root

    def _root_inv_decomposition(self, initial_vectors=None):
        evals, q_matrix = self.lazy_tensor.symeig(eigenvectors=True)
        inv_sqrt_evals = DiagLazyTensor((evals + self.diag_tensor.diag()).pow(-0.5))
        matrix_inv_root = q_matrix.matmul(inv_sqrt_evals)
        return matrix_inv_root
