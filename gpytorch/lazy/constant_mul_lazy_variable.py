from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable


class ConstantMulLazyVariable(LazyVariable):

    def __init__(self, lazy_var, constant):
        if not isinstance(constant, Variable):
            tensor_cls = lazy_var.tensor_cls
            constant = Variable(tensor_cls(1).fill_(constant))
        super(ConstantMulLazyVariable, self).__init__(lazy_var, constant)
        self.lazy_var = lazy_var
        self.constant = constant

    def _matmul(self, rhs):
        res = self.lazy_var._matmul(rhs)
        res = res * self.constant.expand_as(res)
        return res

    def _t_matmul(self, rhs):
        res = self.lazy_var._t_matmul(rhs)
        res = res * self.constant.expand_as(res)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        res = list(self.lazy_var._quad_form_derivative(left_vecs, right_vecs))
        for i, item in enumerate(res):
            if torch.is_tensor(item) and res[i].sum():
                res[i] = res[i] * self.constant.expand_as(res[i])
        # Gradient with respect to the constant
        res.append(left_vecs.new(1).fill_((left_vecs * right_vecs).sum()))
        return res

    def _size(self):
        return self.lazy_var.size()

    def _transpose_nonbatch(self):
        return ConstantMulLazyVariable(
            self.lazy_var._transpose_nonbatch(), self.constant
        )

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = self.lazy_var._batch_get_indices(
            batch_indices, left_indices, right_indices
        )
        return self.constant.expand_as(res) * res

    def _get_indices(self, left_indices, right_indices):
        res = self.lazy_var._get_indices(left_indices, right_indices)
        return self.constant.expand_as(res) * res

    def _approx_diag(self):
        res = self.lazy_var._approx_diag()
        return res * self.constant.expand_as(res)

    def evaluate(self):
        res = self.lazy_var.evaluate()
        return res * self.constant.expand_as(res)

    def diag(self):
        res = self.lazy_var.diag()
        res = res * self.constant.expand_as(res)
        return res

    def repeat(self, *sizes):
        return ConstantMulLazyVariable(self.lazy_var.repeat(*sizes), self.constant)

    def __getitem__(self, i):
        return self.lazy_var.__getitem__(i) * self.constant
