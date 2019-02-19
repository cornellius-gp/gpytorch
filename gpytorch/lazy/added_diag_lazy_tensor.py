#!/usr/bin/env python3

import torch
import warnings
from .sum_lazy_tensor import SumLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from ..utils import broadcasting, pivoted_cholesky, woodbury
from .. import settings


class AddedDiagLazyTensor(SumLazyTensor):
    """
    A SumLazyTensor, but of only two lazy tensors, the second of which must be
    a DiagLazyTensor.
    """

    def __init__(self, *lazy_tensors):
        lazy_tensors = list(lazy_tensors)
        super(AddedDiagLazyTensor, self).__init__(*lazy_tensors)
        if len(lazy_tensors) > 2:
            raise RuntimeError("An AddedDiagLazyTensor can only have two components")

        broadcasting._mul_broadcast_shape(lazy_tensors[0].shape, lazy_tensors[1].shape)

        if isinstance(lazy_tensors[0], DiagLazyTensor) and isinstance(lazy_tensors[1], DiagLazyTensor):
            raise RuntimeError("Trying to lazily add two DiagLazyTensors. " "Create a single DiagLazyTensor instead.")
        elif isinstance(lazy_tensors[0], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[0]
            self._lazy_tensor = lazy_tensors[1]
        elif isinstance(lazy_tensors[1], DiagLazyTensor):
            self._diag_tensor = lazy_tensors[1]
            self._lazy_tensor = lazy_tensors[0]
        else:
            raise RuntimeError("One of the LazyTensors input to AddedDiagLazyTensor must be a DiagLazyTensor!")

    def _matmul(self, rhs):
        return torch.addcmul(
            self._lazy_tensor._matmul(rhs),
            self._diag_tensor._diag.unsqueeze(-1),
            rhs
        )

    def add_diag(self, added_diag):
        return AddedDiagLazyTensor(self._lazy_tensor, self._diag_tensor.add_diag(added_diag))

    def __add__(self, other):
        from .diag_lazy_tensor import DiagLazyTensor

        if isinstance(other, DiagLazyTensor):
            return AddedDiagLazyTensor(self._lazy_tensor, self._diag_tensor + other)
        else:
            return AddedDiagLazyTensor(self._lazy_tensor + other, self._diag_tensor)

    def _preconditioner(self):
        if settings.max_preconditioner_size.value() == 0:
            return None, None

        if not hasattr(self, "_woodbury_cache"):
            max_iter = settings.max_preconditioner_size.value()
            self._piv_chol_self = pivoted_cholesky.pivoted_cholesky(self._lazy_tensor, max_iter)
            if torch.any(torch.isnan(self._piv_chol_self)).item():
                warnings.warn(
                    "NaNs encountered in preconditioner computation. Attempting to continue without " "preconditioning."
                )
                return None, None
            self._woodbury_cache, self._precond_logdet_cache = woodbury.woodbury_factor(
                self._piv_chol_self, self._piv_chol_self, self._diag_tensor.diag(), logdet=True
            )

        # preconditioner
        def precondition_closure(tensor):
            return woodbury.woodbury_solve(
                tensor, self._piv_chol_self, self._woodbury_cache, self._diag_tensor.diag()
            )

        return precondition_closure, self._precond_logdet_cache
