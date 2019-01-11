#!/usr/bin/env python3
from .rbf_kernel import RBFKernel
from ..lazy import DiagLazyTensor, NonLazyTensor
from ..utils.deprecation import _deprecate_kwarg
from torch.nn.functional import softplus
import torch
from ..lazy.kronecker_product_lazy_tensor import KroneckerProductLazyTensor


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
        b = 1
        if len(x1.size()) == 2:
            n1, d = x1.size()
            n2, _ = x2.size()
        else:
            b, n1, d = x1.size()
            _, n2, _ = x2.size()

        K = torch.zeros(b, n1 * (d + 1), n2 * (d + 1))  # batch x n1(d+1) x n2(d+1)

        if not diag:
            ell = self.lengthscale
            x1_ = x1 / ell
            x2_ = x2 / ell

            # Form all possible rank-1 product for the gradient and Hessian blocks
            outer = x1_.view([b, n1, 1, d]) - x2_.view([b, 1, n2, d])  
            outer = torch.transpose(outer, -1, -2).contiguous()

            # 1) Kernel block
            diff = self._covar_dist(x1_, x2_, square_dist=True, **params)
            K_11 = diff.div_(-2).exp_()
            K[..., :n1, :n2] = K_11

            # 2) First gradient block
            outer1 = outer.view([b, n1, n2*d])
            K[..., :n1, n2:] = (outer1  / ell) * K_11.repeat([1, 1, d])

            # 3) Second gradient block
            outer2 = outer.transpose(-1, -3).contiguous().view([b, n2, n1*d])
            outer2 = outer2.transpose(-1, -2) 
            K[..., n1:, :n2] = - (outer2  / ell) * K_11.repeat([1, d, 1])

            # 4) Hessian block
            outer3 = outer1.repeat([1, d, 1]) * outer2.repeat([1, 1, d])
            kp = KroneckerProductLazyTensor(torch.eye(d, d), torch.ones(n1, n2))
            chain_rule = kp.evaluate() / ell.pow(2) - outer3 / ell.pow(2)
            K[..., n1:, n2:] = chain_rule * K_11.repeat([1, d, d]) 

            # Symmetrize for stability
            if n1 == n2 and torch.eq(x1, x2).all():  
                K = 0.5*(K.transpose(-1, -2) + K)

            # Apply a perfect shuffle permutation to match the MutiTask ordering
            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().contiguous().view((n1 * (d + 1)))
            pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            K = K[..., pi1, :][..., :, pi2]

            return K

        else: # TODO: This will change when ARD is supported
            if not (n1 == n2 and torch.eq(x1, x2).all()):
                raise RuntimeError('diag=True only works when x1 == x2')

            kernel_diag = super(RBFKernelGrad, self).forward(x1, x2, diag=True)
            grad_diag = (1 / self.lengthscale.pow(2)) * torch.ones(1, n2*d)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            pi = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            return k_diag[..., pi]

    def size(self, x1, x2):
        """
        Given `n` data points in `d` dimensions, RBFKernelGrad returns an `n(d+1) x n(d+1)` kernel
        matrix.
        """
        non_batch_size = ((x1.size(-1) + 1) * x1.size(-2), (x2.size(-1) + 1) * x2.size(-2))
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)
