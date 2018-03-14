from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ..module import Module


class Kernel(Module):
    def __init__(self, active_dims=None):
        self.active_dims = active_dims
        super(Kernel, self).__init__()

    def forward(self, x1, x2, **params):
        raise NotImplementedError()

    def __call__(self, x1_, x2_=None, **params):
        if self.active_dims is not None:
            x1 = x1_[:, self.active_dims]
            if x2_ is not None:
                x2 = x2_[:, self.active_dims]
        else:
            x1 = x1_
            x2 = x2_

        if x2 is None:
            x2 = x1

        # Give x1 and x2 a last dimension, if necessary
        if x1.data.ndimension() == 1:
            x1 = x1.unsqueeze(1)
        if x2.data.ndimension() == 1:
            x2 = x2.unsqueeze(1)
        if not x1.size(-1) == x2.size(-1):
            raise RuntimeError('x1 and x2 must have the same number of dimensions!')

        # Do everything in batch mode by default
        is_batch = x1.ndimension() == 3
        if not is_batch:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        res = super(Kernel, self).__call__(x1, x2, **params)
        if not is_batch:
            res = res[0]
        return res

    def __add__(self, other):
        return AdditiveKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)


class AdditiveKernel(Kernel):
    def __init__(self, kernel_1, kernel_2):
        super(AdditiveKernel, self).__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def forward(self, x1, x2):
        return self.kernel_1(x1, x2) + self.kernel_2(x1, x2)


class ProductKernel(Kernel):
    def __init__(self, kernel_1, kernel_2):
        super(ProductKernel, self).__init__()
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def forward(self, x1, x2):
        return self.kernel_1(x1, x2) * self.kernel_2(x1, x2)
