from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
from ..utils import pivoted_cholesky
from .lazy_variable import LazyVariable
from ..functions import inv_matmul, inv_quad_log_det


class InvQuadLazyVariable(LazyVariable):

    def __init__(self, inv_mat, left_mat, right_mat, added_diag=None):
        for arg in [inv_mat, left_mat, right_mat]:
            if isinstance(arg, LazyVariable):
                raise RuntimeError(
                    "The arguments to InvQuadLazyVariable should not be LazyVariables!"
                )

        if added_diag is None:
            added_diag = right_mat.new(right_mat.size(0)).zero_()
            if right_mat.ndimension() == 3:
                added_diag = (
                    added_diag.unsqueeze(1).expand(right_mat.size(0), right_mat.size(1))
                )

        super(InvQuadLazyVariable, self).__init__(
            inv_mat, left_mat, right_mat, added_diag
        )
        self.inv_mat = inv_mat
        self.left_mat = left_mat
        self.right_mat = right_mat
        self.added_diag = added_diag

    def mul(self, other):
        if (
            isinstance(other, int)
            or isinstance(other, float)
            or (torch.is_tensor(other) and other.numel() == 1)
        ):
            return InvQuadLazyVariable(
                self.inv_mat,
                self.left_mat * other,
                self.right_mat,
                self.added_diag * other,
            )
        else:
            return super(InvQuadLazyVariable, self).mul(other)

    def _matmul(self, rhs):
        res = self.right_mat.transpose(-1, -2).matmul(rhs)
        res = inv_matmul(self.inv_mat, res)
        res = self.left_mat.matmul(res)
        if rhs.ndimension() == 1:
            res.addcmul_(self.added_diag, rhs)
        else:
            res.addcmul_(self.added_diag.unsqueeze(-1), rhs)
        return res

    def _preconditioner(self):
        if gpytorch.settings.max_preconditioner_size.value() == 0:
            return None

        if not hasattr(self, "_woodbury_cache"):
            max_iter = gpytorch.settings.max_preconditioner_size.value()
            lv_no_diag = InvQuadLazyVariable(
                self.inv_mat, self.left_mat, self.right_mat, None
            )
            self._piv_chol_self = pivoted_cholesky.pivoted_cholesky(
                lv_no_diag, max_iter
            )
            self._woodbury_cache = pivoted_cholesky.woodbury_factor(
                self._piv_chol_self, self.added_diag
            )

        def precondition_closure(tensor):
            return pivoted_cholesky.woodbury_solve(
                tensor, self._piv_chol_self, self._woodbury_cache, self.added_diag
            )

        return precondition_closure

    def _quad_form_derivative(self, left_vecs, right_vecs):
        left_mat_left_vecs = self.left_mat.transpose(-1, -2).matmul(left_vecs)
        right_mat_right_vecs = self.right_mat.transpose(-1, -2).matmul(right_vecs)
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(-1)
            right_vecs = right_vecs.unsqueeze(-1)
            left_mat_left_vecs = left_mat_left_vecs.unsqueeze(-1)
            right_mat_right_vecs = right_mat_right_vecs.unsqueeze(-1)

        solves = inv_matmul(
            self.inv_mat, torch.cat([left_mat_left_vecs, right_mat_right_vecs], -1)
        )
        left_solves = solves.narrow(-1, 0, left_mat_left_vecs.size(-1))
        right_solves = solves.narrow(
            -1, left_mat_left_vecs.size(-1), right_mat_right_vecs.size(-1)
        )

        inv_mat_grad = torch.matmul(left_solves, right_solves.transpose(-1, -2)).mul_(
            -1
        )
        left_mat_grad = torch.matmul(left_vecs, right_solves.transpose(-1, -2))
        right_mat_grad = torch.matmul(right_vecs, left_solves.transpose(-1, -2))
        diag_grad = (left_vecs * right_vecs).sum(-1)

        return inv_mat_grad, left_mat_grad, right_mat_grad, diag_grad

    def _size(self):
        if self.left_mat.ndimension() == 3:
            return torch.Size(
                (self.left_mat.size(0), self.left_mat.size(1), self.right_mat.size(1))
            )
        else:
            return torch.Size((self.left_mat.size(0), self.right_mat.size(0)))

    def _transpose_nonbatch(self):
        return InvQuadLazyVariable(
            self.inv_mat.transpose(-1, -2), self.right_mat, self.left_mat
        )

    def add_diag(self, diag):
        diag = diag.expand_as(self.added_diag)
        res = InvQuadLazyVariable(
            self.inv_mat, self.left_mat, self.right_mat, diag + self.added_diag
        )
        return res

    def diag(self):
        lhs = self.left_mat
        rhs = inv_matmul(self.inv_mat, self.right_mat.transpose(-1, -2))
        res = (lhs * rhs.transpose(-1, -2)).sum(-1)
        res.add_(self.added_diag)
        return res

    def inv_quad_log_det(self, inv_quad_rhs=None, log_det=False):
        if inv_quad_rhs is not None and inv_quad_rhs.ndimension() == 1:
            inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)

        inner_mat = self.left_mat.transpose(-1, -2).matmul(
            self.left_mat / self.added_diag.unsqueeze(-1)
        )
        inner_mat = inner_mat.add(self.inv_mat)

        left_mat_inv_quad_rhs = None
        if inv_quad_rhs is not None:
            left_mat_inv_quad_rhs = torch.matmul(
                self.left_mat.transpose(-1, -2),
                inv_quad_rhs / self.added_diag.unsqueeze(-1),
            )

        solve_mats = None
        if self.ndimension() == 3:
            solve_mats = torch.cat([inner_mat, self.inv_mat])
            if left_mat_inv_quad_rhs is not None:
                left_mat_inv_quad_rhs = left_mat_inv_quad_rhs.repeat(2, 1, 1)
        else:
            solve_mats = torch.cat([inner_mat.unsqueeze(0), self.inv_mat.unsqueeze(0)])
            if left_mat_inv_quad_rhs is not None:
                left_mat_inv_quad_rhs = left_mat_inv_quad_rhs.unsqueeze(0).repeat(
                    2, 1, 1
                )

        inv_quad_parts, log_det_parts = inv_quad_log_det(
            solve_mats, inv_quad_rhs=left_mat_inv_quad_rhs, log_det=log_det
        )

        inv_quad_res = None
        if inv_quad_rhs is not None:
            if self.ndimension() == 3:
                inv_quad_res = inv_quad_parts[: self.size(0)]
            else:
                inv_quad_res = inv_quad_parts[0]
            inv_quad_res = (inv_quad_rhs.pow(2) / self.added_diag.unsqueeze(-1)).sum(
                -1
            ).sum(
                -1
            ) - inv_quad_res

        log_det_res = None
        if log_det:
            if self.ndimension() == 3:
                log_det_res = log_det_parts[: self.size(0)] - log_det_parts[
                    self.size(0) :
                ]
            else:
                log_det_res = log_det_parts[0] - log_det_parts[1]
            log_det_res = log_det_res + self.added_diag.log().sum(-1)

        return inv_quad_res, log_det_res
