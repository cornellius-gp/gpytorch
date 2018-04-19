from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .lazy_variable import LazyVariable
import torch
from torch.autograd import Variable
import pdb


class ImplicitMulBatchLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable):
        super(ImplicitMulBatchLazyVariable, self).__init__(base_lazy_variable)
        if base_lazy_variable.ndimension() < 3:
            raise RuntimeError('Can only implicitly mul over a batch variable!')
        self.base_lazy_variable = base_lazy_variable

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

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        raise RuntimeError('An ImplicitMulBatchLazyVariable is by definition not batched')

    def __getitem__(self, index):
        new_index = (slice(None, None, None), *index)
        result = self.base_lazy_variable.__getitem__(new_index)
        if isinstance(result, LazyVariable):
            result = result.evaluate()
        return result.prod(0)

    def _get_indices(self, left_indices, right_indices):
        batch_indices = Variable(self.tensor_cls(self.batch_size()).long())
        torch.arange(0, self.batch_size(), out=batch_indices.data)
        batch_indices = batch_indices.unsqueeze(1).repeat(1, len(left_indices)).view(-1)
        left_indices = left_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        right_indices = right_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        pdb.set_trace()
        res = self.base_lazy_variable._batch_get_indices(batch_indices, left_indices, right_indices)
        return res.view(self.batch_size(), -1).prod(0)

    def batch_size(self):
        return self.base_lazy_variable.size(0)

    def diag(self):
        return self.base_lazy_variable.diag().prod(0)

    def _size(self):
        return self.base_lazy_variable.size()[1:]
