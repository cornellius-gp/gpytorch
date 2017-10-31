import torch
from torch.autograd import Variable
from ..posterior import DefaultPosteriorStrategy
from .lazy_variable import LazyVariable


class ConstantMulLazyVariable(LazyVariable):
    def __init__(self, lazy_var, constant):
        super(ConstantMulLazyVariable, self).__init__(lazy_var)
        self.lazy_var = lazy_var
        if not isinstance(constant, Variable):
            tensor_cls = type(self.lazy_var.representation()[0].data)
            self.constant = Variable(tensor_cls(1).fill_(constant))
        else:
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
                if torch.is_tensor(item):
                    res[i] = res[i] * constant.expand_as(res[i])
            # Gradient with respect to the constant
            res.append(left_factor.new(1).fill_((left_factor * right_factor).sum()))
            return res
        return closure

    def diag(self):
        return self.lazy_var.diag() * self.constant

    def evaluate(self):
        return self.lazy_var.evaluate() * self.constant

    def representation(self):
        return tuple(list(self.lazy_var.representation()) + [self.constant])

    def posterior_strategy(self):
        return DefaultPosteriorStrategy(self)

    def _transpose_nonbatch(self):
        return ConstantMulLazyVariable(self.lazy_var._transpose_nonbatch(), self.constant)

    def _get_indices(self, left_indices, right_indices):
        res = self.lazy_var._get_indices(left_indices, right_indices)
        return self.constant.expand_as(res) * res

    def __getitem__(self, i):
        return self.lazy_var.__getitem__(i) * self.constant

    def size(self):
        return self.lazy_var.size()
