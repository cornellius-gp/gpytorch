from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .sum_lazy_variable import SumLazyVariable
from .diag_lazy_variable import DiagLazyVariable
from ..utils import pivoted_cholesky
from .. import settings


class AddedDiagLazyVariable(SumLazyVariable):
    """
    A SumLazyVariable, but of only two lazy variables, the second of which must be
    a DiagLazyVariable.
    """

    def __init__(self, *lazy_vars):
        lazy_vars = list(lazy_vars)
        super(AddedDiagLazyVariable, self).__init__(*lazy_vars)
        if len(lazy_vars) > 2:
            raise RuntimeError("An AddedDiagLazyVariable can only have two components")

        if (
            isinstance(lazy_vars[0], DiagLazyVariable)
            and isinstance(lazy_vars[1], DiagLazyVariable)
        ):
            raise RuntimeError(
                "Trying to lazily add two DiagLazyVariables. "
                "Create a single DiagLazyVariable instead."
            )
        elif isinstance(lazy_vars[0], DiagLazyVariable):
            self._diag_var = lazy_vars[0]
            self._lazy_var = lazy_vars[1]
        elif isinstance(lazy_vars[1], DiagLazyVariable):
            self._diag_var = lazy_vars[1]
            self._lazy_var = lazy_vars[0]
        else:
            raise RuntimeError(
                "One of the LazyVariables input to AddedDiagLazyVariable "
                " must be a DiagLazyVariable!"
            )

        if not (self._diag_var.diag().data == self._diag_var.diag().data[0]).all():
            raise RuntimeError(
                "AddedDiagLazyVariable only supports constant shifts "
                "(e.g. s in K + s*I)"
            )

    def _preconditioner(self):
        if settings.max_preconditioner_size.value() == 0:
            return None, None

        if not hasattr(self, "_woodbury_cache"):
            max_iter = settings.max_preconditioner_size.value()
            self._piv_chol_self = pivoted_cholesky.pivoted_cholesky(
                self._lazy_var, max_iter
            )
            self._woodbury_cache = pivoted_cholesky.woodbury_factor(
                self._piv_chol_self, self._diag_var.diag()
            )

        # preconditioner
        def precondition_closure(tensor):
            return pivoted_cholesky.woodbury_solve(
                tensor, self._piv_chol_self, self._woodbury_cache, self._diag_var.diag()
            )

        # log_det correction
        if not hasattr(self, "_precond_log_det_cache"):
            lr_flipped = self._piv_chol_self.matmul(
                self._piv_chol_self.transpose(-2, -1).
                div(self._diag_var.diag().unsqueeze(1))
            )
            lr_flipped = lr_flipped + torch.eye(
                n=lr_flipped.size(0), dtype=lr_flipped.dtype, device=lr_flipped.device
            )
            ld_one = lr_flipped.potrf().diag().log().sum() * 2
            ld_two = self._diag_var.diag().data.log().sum()
            self._precond_log_det_cache = ld_one + ld_two

        return precondition_closure, self._precond_log_det_cache
