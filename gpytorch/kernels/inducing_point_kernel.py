from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from ..functions import add_jitter
from ..lazy import LazyVariable, DiagLazyVariable, MatmulLazyVariable, RootLazyVariable
from ..random_variables import GaussianRandomVariable
from ..variational import MVNVariationalStrategy


class InducingPointKernel(Kernel):
    def __init__(self, base_kernel_module, inducing_points, active_dims=None):
        super(InducingPointKernel, self).__init__(active_dims=active_dims)
        self.base_kernel_module = base_kernel_module

        if inducing_points.ndimension() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if inducing_points.ndimension() != 2:
            raise RuntimeError("Inducing points should be 2 dimensional")
        self.register_parameter(
            "inducing_points", torch.nn.Parameter(inducing_points.unsqueeze(0)), bounds=(-1e10, 1e10)
        )
        self.register_variational_strategy("inducing_point_strategy")

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(InducingPointKernel, self).train(mode)

    @property
    def _inducing_mat(self):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = self.base_kernel_module(self.inducing_points, self.inducing_points).evaluate()
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            inv_roots_list = []
            for i in range(self._inducing_mat.size(0)):
                jitter_mat = add_jitter(self._inducing_mat[i])
                chol = torch.potrf(jitter_mat)
                eye = torch.eye(chol.size(-1), device=chol.device)
                inv_roots_list.append(torch.trtrs(eye, chol)[0])

            res = torch.cat(inv_roots_list, 0)
            if not self.training:
                self._cached_kernel_inv_root = res
            return res

    def _get_covariance(self, x1, x2):
        k_ux1 = self.base_kernel_module(x1, self.inducing_points).evaluate()
        if torch.equal(x1, x2):
            covar = RootLazyVariable(k_ux1.matmul(self._inducing_inv_root))
        else:
            k_ux2 = self.base_kernel_module(x2, self.inducing_points).evaluate()
            covar = MatmulLazyVariable(
                k_ux1.matmul(self._inducing_inv_root), k_ux2.matmul(self._inducing_inv_root).transpose(-1, -2)
            )
        return covar

    def _covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)
        orig_size = list(inputs.size())

        # Resize inputs so that everything is batch
        inputs = inputs.unsqueeze(-2).view(-1, 1, inputs.size(-1))

        # Get diagonal of covar
        covar_diag = self.base_kernel_module(inputs)
        if isinstance(covar_diag, LazyVariable):
            covar_diag = covar_diag.evaluate()
        covar_diag = covar_diag.view(orig_size[:-1])
        return DiagLazyVariable(covar_diag)

    def forward(self, x1, x2, **kwargs):
        covar = self._get_covariance(x1, x2)

        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")
            zero_mean = torch.zeros_like(x1.select(-1, 0))
            new_variational_strategy = MVNVariationalStrategy(
                GaussianRandomVariable(zero_mean, self._covar_diag(x1)), GaussianRandomVariable(zero_mean, covar)
            )
            self.update_variational_strategy("inducing_point_strategy", new_variational_strategy)

        return covar
