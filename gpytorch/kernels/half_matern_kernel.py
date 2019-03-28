import math
import torch
from ..functions import MaternCovariance
from .matern_kernel import MaternKernel
from ..lazy import HalfNonLazyTensor

class HalfMaternKernel(MaternKernel):
    # def forward(self, x1, x2, **params):
    #     base_out = super().forward(x1, x2, **params)
    #     return HalfNonLazyTensor(base_out.half())

    def forward(self, x1, x2, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or params.get('diag', False)
        ):
            mean = x1.contiguous().view(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale.half())
            x2_ = (x2 - mean).div(self.lengthscale.half())
            distance = self._covar_dist(x1_, x2_, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component
        return HalfNonLazyTensor(MaternCovariance().apply(x1, x2, self.lengthscale.half(), self.nu,
                                        lambda x1, x2: self._covar_dist(x1, x2, **params)))
