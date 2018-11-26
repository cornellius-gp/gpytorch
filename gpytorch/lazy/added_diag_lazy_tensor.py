#!/usr/bin/env python3

import torch
import warnings
from .non_lazy_tensor import NonLazyTensor
from .sum_lazy_tensor import SumLazyTensor
from .diag_lazy_tensor import DiagLazyTensor
from ..utils import pivoted_cholesky
from .. import settings
from ..utils import broadcasting


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
            self._woodbury_cache = pivoted_cholesky.woodbury_factor(self._piv_chol_self, self._diag_tensor.diag())

        # preconditioner
        def precondition_closure(tensor):
            return pivoted_cholesky.woodbury_solve(
                tensor, self._piv_chol_self, self._woodbury_cache, self._diag_tensor.diag()
            )

        # log_det correction
        if not hasattr(self, "_precond_log_det_cache"):
            lr_flipped = self._piv_chol_self.matmul(
                self._piv_chol_self.transpose(-2, -1).div(self._diag_tensor.diag().unsqueeze(-1))
            )
            lr_flipped = lr_flipped + torch.eye(n=lr_flipped.size(-2), dtype=lr_flipped.dtype, device=lr_flipped.device)
            if lr_flipped.ndimension() == 3:
                ld_one = (NonLazyTensor(torch.cholesky(lr_flipped, upper=True)).diag().log().sum(-1)) * 2
                ld_two = self._diag_tensor.diag().log().sum(-1)
            else:
                ld_one = lr_flipped.cholesky(upper=True).diag().log().sum() * 2
                ld_two = self._diag_tensor.diag().log().sum().item()
            self._precond_log_det_cache = ld_one + ld_two

        return precondition_closure, self._precond_log_det_cache
