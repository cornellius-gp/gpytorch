import torch
from .lazy_variable import LazyVariable


class SumBatchLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable):
        if base_lazy_variable.ndimension() != 3:
            raise RuntimeError('Base lazy variable must be a batch matrix (i.e. 3 dimensions)')
        super(SumBatchLazyVariable, self).__init__(base_lazy_variable)
        self.base_lazy_variable = base_lazy_variable
        self.tensor_cls = type(self.base_lazy_variable.representation()[0].data)

    def _matmul_closure_factory(self, *args):
        super_closure = self.base_lazy_variable._matmul_closure_factory(*args)

        def closure(tensor):
            isvector = tensor.ndimension() == 1
            if isvector:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.unsqueeze(0)
            tensor_size = list(tensor.size())
            tensor_size[0] = self.batch_size()
            tensor = tensor.expand(*tensor_size)

            res = super_closure(tensor).sum(0)
            if isvector:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        super_closure = self.base_lazy_variable._derivative_quadratic_form_factory(*args)

        def closure(left_factor, right_factor):
            left_factor = left_factor.unsqueeze(0)
            left_factor_size = list(left_factor.size())
            left_factor_size[0] = self.batch_size()
            left_factor = left_factor.expand(*left_factor_size)

            right_factor = right_factor.unsqueeze(0)
            right_factor_size = list(right_factor.size())
            right_factor_size[0] = self.batch_size()
            right_factor = right_factor.expand(*right_factor_size)

            res = super_closure(left_factor, right_factor)
            return res

        return closure

    def batch_size(self):
        return self.base_lazy_variable.size()[0]

    def chol_approx_size(self):
        return self.base_lazy_variable.chol_approx_size()

    def chol_matmul(self, tensor):
        tensor = tensor.unsqueeze(0)
        tensor_size = list(tensor.size())
        tensor_size[0] = self.batch_size()
        tensor = tensor.expand(*tensor_size)
        return self.base_lazy_variable.chol_matmul(tensor).sum(0)

    def diag(self):
        return self.base_lazy_variable.diag().sum(0)

    def size(self):
        base_size = self.base_lazy_variable.size()
        return torch.Size(list(base_size)[1:])
