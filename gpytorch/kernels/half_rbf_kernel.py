from .rbf_kernel import RBFKernel
from ..lazy import HalfNonLazyTensor


class HalfRBFKernel(RBFKernel):
    def forward(self, x1, x2, diag=False, **params):
        base_out = super().forward(x1, x2, diag, **params)
        return HalfNonLazyTensor(base_out.half())
