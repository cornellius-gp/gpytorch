from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_variable import LazyVariable
from ..functions import inv_matmul


class InvQuadLazyVariable(LazyVariable):
    def __init__(self, inv_mat, left_mat, right_mat):
        for arg in [inv_mat, left_mat, right_mat]:
            if isinstance(arg, LazyVariable):
                raise RuntimeError('The arguments to InvQuadLazyVariable should not be LazyVariables!')
        super(InvQuadLazyVariable, self).__init__(inv_mat, left_mat, right_mat)
        self.inv_mat = inv_mat
        self.left_mat = left_mat
        self.right_mat = right_mat

    def _matmul(self, rhs):
        res = self.right_mat.transpose(-1, -2).matmul(rhs)
        res = inv_matmul(self.inv_mat, res)
        res = self.left_mat.matmul(res)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        left_mat_left_vecs = self.left_mat.transpose(-1, -2).matmul(left_vecs)
        right_mat_right_vecs = self.right_mat.transpose(-1, -2).matmul(right_vecs)
        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(-1)
            right_vecs = right_vecs.unsqueeze(-1)
            left_mat_left_vecs = left_mat_left_vecs.unsqueeze(-1)
            right_mat_right_vecs = right_mat_right_vecs.unsqueeze(-1)

        solves = inv_matmul(self.inv_mat, torch.cat([
            left_mat_left_vecs,
            right_mat_right_vecs
        ], -1))
        left_solves = solves.narrow(-1, 0, left_mat_left_vecs.size(-1))
        right_solves = solves.narrow(-1, left_mat_left_vecs.size(-1), right_mat_right_vecs.size(-1))

        inv_mat_grad = torch.matmul(left_solves, right_solves.transpose(-1, -2)).mul_(-1)
        left_mat_grad = torch.matmul(left_vecs, right_solves.transpose(-1, -2))
        right_mat_grad = torch.matmul(right_vecs, left_solves.transpose(-1, -2))

        return inv_mat_grad, left_mat_grad, right_mat_grad

    def _size(self):
        if self.left_mat.ndimension() == 3:
            return torch.Size((
                self.left_mat.size(0),
                self.left_mat.size(1),
                self.right_mat.size(1),
            ))
        else:
            return torch.Size((
                self.left_mat.size(0),
                self.right_mat.size(0),
            ))

    def _transpose_nonbatch(self):
        return InvQuadLazyVariable(
            self.inv_mat.transpose(-1, -2),
            self.right_mat,
            self.left_mat,
        )

    def diag(self):
        lhs = self.left_mat
        rhs = inv_matmul(self.inv_mat, self.right_mat.transpose(-1, -2))
        return (lhs * rhs.transpose(-1, -2)).sum(-1)
