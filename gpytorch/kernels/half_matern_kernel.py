from .matern_kernel import MaternKernel
from ..lazy import HalfNonLazyTensor


class HalfMaternKernel(MaternKernel):
    def forward(self, x1, x2, diag=False, **params):
        base_out = super().forward(x1, x2, diag, **params)
        return base_out if diag else HalfNonLazyTensor(base_out.half())
