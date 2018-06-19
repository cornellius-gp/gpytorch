from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from ..module import Module
from ..lazy import LazyEvaluatedKernelVariable, ZeroLazyVariable
import gpytorch


class Kernel(Module):
    def __init__(
        self, has_lengthscale=False, ard_num_dims=None, log_lengthscale_bounds=(-10000, 10000), active_dims=None
    ):
        super(Kernel, self).__init__()
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.active_dims = active_dims
        self.ard_num_dims = ard_num_dims
        if has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                "log_lengthscale",
                torch.nn.Parameter(torch.Tensor(1, 1, lengthscale_num_dims).zero_()),
                bounds=log_lengthscale_bounds,
            )

    @property
    def lengthscale(self):
        if "log_lengthscale" in self.named_parameters().keys():
            return self.log_lengthscale.exp()
        else:
            return None

    def forward(self, x1, x2, **params):
        raise NotImplementedError()

    def __call__(self, x1_, x2_=None, **params):
        if self.active_dims is not None:
            x1 = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2 = x2_.index_select(-1, self.active_dims)
        else:
            x1 = x1_
            x2 = x2_

        if x2 is None:
            x2 = x1

        # Give x1 and x2 a last dimension, if necessary
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)
        if x2.ndimension() == 1:
            x2 = x2.unsqueeze(1)
        if not x1.size(-1) == x2.size(-1):
            raise RuntimeError("x1 and x2 must have the same number of dimensions!")

        return LazyEvaluatedKernelVariable(self, x1, x2)

    def __add__(self, other):
        return AdditiveKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)


class AdditiveKernel(Kernel):
    def __init__(self, *kernels):
        super(AdditiveKernel, self).__init__()
        self.kernels = kernels

    def forward(self, x1, x2):
        res = ZeroLazyVariable()
        for kern in self.kernels:
            res = res + kern(x1, x2).evaluate_kernel()

        return res


class ProductKernel(Kernel):
    def __init__(self, *kernels):
        super(ProductKernel, self).__init__()
        self.kernels = kernels

    def forward(self, x1, x2):
        return gpytorch.utils.prod([k(x1, x2).evaluate_kernel() for k in self.kernels])
