from .matern_kernel import MaternKernel
from ..lazy import HalfNonLazyTensor


class HalfRBFKernel(MaternKernel):
    def forward(self, x1, x2, **params):
        base_out = super().forward(x1, x2, **params)
        return HalfNonLazyTensor(base_out.half())
