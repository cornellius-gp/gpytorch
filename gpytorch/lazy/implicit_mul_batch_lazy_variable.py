from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .lazy_variable import LazyVariable
import pdb


class ImplicitMulBatchLazyVariable(LazyVariable):
    def __init__(self, base_batch_lazy_variable):
        super(ImplicitMulBatchLazyVariable, self).__init__(base_batch_lazy_variable)
        if base_batch_lazy_variable.ndimension() < 3:
            raise RuntimeError('Can only implicitly mul over a batch variable!')
        self.base_batch_lazy_variable = base_batch_lazy_variable

    def __getitem__(self, index):
        new_index = (slice(None, None, None), *index)
        result = self.base_batch_lazy_variable.__getitem__(new_index)
        if isinstance(result, LazyVariable):
            result = result.evaluate()
        return result.prod(0)

    def diag(self):
        return self.base_batch_lazy_variable.diag().prod(0)

    def _size(self):
        return self.base_batch_lazy_variable.size()[1:]
