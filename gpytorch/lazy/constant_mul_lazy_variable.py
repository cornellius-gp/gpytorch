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

    def _matmul_closure_factory(self, *args):
        lazy_var_closure = self.lazy_var._matmul_closure_factory(*args[:-1])
        constant = args[-1]

        def closure(rhs_mat):
            res = lazy_var_closure(rhs_mat)
            res = res * constant.expand_as(res)
            return res
        return closure

    def _derivative_quadratic_form_factory(self, *args):
        lazy_var_closure = self.lazy_var._derivative_quadratic_form_factory(*args[:-1])
        constant = args[-1]

        def closure(left_factor, right_factor):
            res = list(lazy_var_closure(left_factor, right_factor))
            for i, item in enumerate(res):
                if torch.is_tensor(item) and res[i].sum():
                    res[i] = res[i] * constant.expand_as(res[i])
            # Gradient with respect to the constant
            res.append(left_factor.new(1).fill_((left_factor * right_factor).sum()))
            return res
        return closure

    def _size(self):
        return self.lazy_var.size()

    def _transpose_nonbatch(self):
        return ConstantMulLazyVariable(self.lazy_var._transpose_nonbatch(), self.constant)

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = self.lazy_var._batch_get_indices(batch_indices, left_indices, right_indices)
        return self.constant.expand_as(res) * res

    def _get_indices(self, left_indices, right_indices):
        res = self.lazy_var._get_indices(left_indices, right_indices)
        return self.constant.expand_as(res) * res

    def repeat(self, *sizes):
        return ConstantMulLazyVariable(self.lazy_var.repeat(*sizes), self.constant)

    def __getitem__(self, i):
        return self.lazy_var.__getitem__(i) * self.constant
