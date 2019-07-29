import torch
from pykeops.torch import LazyTensor as KEOLazyTensor
from ..kernel import Kernel
from ..rbf_kernel import postprocess_rbf
from gpytorch.lazy import KeOpsLazyTensor


class RBFKernel(Kernel):
    def __init__(self, **kwargs):
        super(RBFKernel, self).__init__(has_lengthscale=True, **kwargs)

    def covar_func(self, x1, x2, diag=False):
        # TODO: x1 / x2 size checks are a work around for a very minor bug in KeOps.
        # This bug is fixed on KeOps master, and we'll remove that part of the check
        # when they cut a new release.
        if diag or x1.size(-2) == 1 or x2.size(-2) == 1:
            return self.covar_dist(
                x1, x2, square_dist=True, diag=diag,
                dist_postprocess_func=postprocess_rbf,
                postprocess=True
            )
        else:
            with torch.autograd.enable_grad():
                x1_ = KEOLazyTensor(x1[:, None, :])
                x2_ = KEOLazyTensor(x2[None, :, :])

                K = (-((x1_ - x2_) ** 2).sum(-1) / 2).exp()

                return K

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        if diag:
            return self.covar_func(x1_, x2_, diag=True)

        covar_func = lambda x1, x2, diag=False: self.covar_func(x1, x2, diag)
        return KeOpsLazyTensor(x1_, x2_, covar_func)
