#!/usr/bin/env python3
from .rbf_kernel import RBFKernel
from ..lazy import DiagLazyTensor, NonLazyTensor
from ..utils.deprecation import _deprecate_kwarg
from torch.nn.functional import softplus
import torch


class RBFKernelGrad(RBFKernel):
    def __init__(
        self,
        ard_num_dims=None,
        batch_size=1,
        active_dims=None,
        lengthscale_prior=None,
        param_transform=softplus,
        inv_param_transform=None,
        eps=1e-6,
        **kwargs
    ):
        # TODO: Add support for ARD
        if ard_num_dims is not None:
            raise RuntimeError('ARD is not supported with derivative observations yet!')
            
        super(RBFKernelGrad, self).__init__(
            ard_num_dims=None,
            batch_size=1,
            active_dims=None,
            lengthscale_prior=None,
            param_transform=softplus,
            inv_param_transform=None,
            eps=1e-6,
            **kwargs
        )

    def forward(self, x1, x2, diag=False, **params):
        _, n, d = x1.size()
        if not diag:
            K = torch.zeros(n*(d+1), n*(d+1))
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

            all_diff = (x1_ - x2_)
            diff = all_diff.norm(2, dim=-1)

            # 1) Kernel block
            K_11 = diff.pow(2).div(-2).exp()
            K[:n, :n] = K_11

            shape_list = [1] * len(K_11.shape[:-2])
            left_shape_list = shape_list + [d, 1]
            right_shape_list = shape_list + [1, d]

            # 2) Gradient block
            print(K_11.shape)
            K_rep = K_11.unsqueeze(-1).repeat(*(shape_list + [1, 1, d]))
            all_K = (all_diff / self.lengthscale) * K_rep
            deriv_K = all_K.transpose(-3, -1).transpose(-2, -1).contiguous().view(d * n, n)

            K[:n, n:] = deriv_K.t()
            K[n:, :n] = deriv_K
            
            # 3) Hessian block

            outer_prod = all_diff.transpose(-3, -1).transpose(-2, -1).contiguous().view(d * n, n)
            outer_prod = outer_prod.repeat(*right_shape_list)

            repeated_K = K_11.repeat(*left_shape_list).repeat(*right_shape_list)

            diag_part = DiagLazyTensor(NonLazyTensor(repeated_K).diag()).evaluate()
            diag_part = diag_part / self.lengthscale.pow(2)

            K[n:, n:] = diag_part - (outer_prod / self.lengthscale.pow(2)) * repeated_K

            pi = torch.arange(n * (d + 1)).view(d + 1, n).t().contiguous().view((n * (d + 1)))
            K = K[pi, :][:, pi]

            return K
        else:
            kernel_diag = super(RBFKernelGrad, self).forward(x1, x2, diag=True)
            # TODO: This will change when ARD is supported
            grad_diags = (1 / self.lengthscale.squeeze().pow(2)).expand_as(kernel_diag).repeat(1, d)

            k_diag = torch.cat((kernel_diag, grad_diags), dim=-1)
            pi = torch.arange(n * (d + 1)).view(d + 1, n).t().contiguous().view((n * (d + 1)))
            return k_diag[pi]


    def size(self, x1, x2):
        """
        Given `n` data points in `d` dimensions, RBFKernelGrad returns an `n(d+1) x n(d+1)` kernel
        matrix.
        """
        non_batch_size = ((x1.size(-1)+1) * x1.size(-2), (x2.size(-1)+1) * x2.size(-2))
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)