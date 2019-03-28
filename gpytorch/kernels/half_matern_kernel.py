from .matern_kernel import MaternKernel
from ..lazy import HalfNonLazyTensor


class HalfMaternKernel(MaternKernel):
    def forward(self, x1, x2, **params):
        base_out = super().forward(x1, x2, **params)
        print(x1.dtype, x2.dtype, base_out.dtype)
        return HalfNonLazyTensor(base_out.half())
