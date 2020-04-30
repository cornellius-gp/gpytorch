#!/usr/bin/env python3


from .lazy_tensor import LazyTensor


class PreconditionedLazyTensor(LazyTensor):
    def __init__(self, base_lazy_tensor, preconditioner):
        super().__init__(base_lazy_tensor, preconditioner)
        self.base_lazy_tensor = base_lazy_tensor
        self.preconditioner = preconditioner

    def _matmul(self, rhs):
        res = self.base_lazy_tensor._matmul(rhs)
        res = self.preconditioner(res)
        return res

    def _size(self):
        return self.base_lazy_tensor.size()

    def _transpose_nonbatch(self):
        return self.__class__(self.base_lazy_tensor._transpose_nonbatch(), self.preconditioner)
