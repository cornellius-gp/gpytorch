import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable


class SumBatchLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable):
        if base_lazy_variable.ndimension() != 3:
            raise RuntimeError('Base lazy variable must be a batch matrix (i.e. 3 dimensions)')
        super(SumBatchLazyVariable, self).__init__(base_lazy_variable)
        self.base_lazy_variable = base_lazy_variable

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

    def _size(self):
        base_size = self.base_lazy_variable.size()
        return torch.Size(list(base_size)[1:])

    def _transpose_nonbatch(self):
        return SumBatchLazyVariable(self.base_lazy_variable._transpose_nonbatch())

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        raise RuntimeError('Batch get indices is not meant to work with a SumBatchLazyVariable')

    def _get_indices(self, left_indices, right_indices):
        batch_indices = Variable(self.tensor_cls(self.batch_size()).long())
        torch.arange(0, self.batch_size(), out=batch_indices.data)
        batch_indices = batch_indices.unsqueeze(1).repeat(1, len(left_indices)).view(-1)
        left_indices = left_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        right_indices = right_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        res = self.base_lazy_variable._batch_get_indices(batch_indices, left_indices, right_indices)
        return res.view(self.batch_size(), -1).sum(0)

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
