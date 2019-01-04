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
        if len(x1.size()) == 1:
            d = 1
            n1, n2 = x1.size()[0], x2.size()[0]
        elif len(x1.size()) == 2:
            n1, d = x1.size()
            n2, _ = x2.size()
        else:
            _, n1, d = x1.size()
            _, n2, _ = x2.size()

        if not diag:
            K = torch.zeros(*x1.shape[:-2], n1 * (d + 1), n2 * (d + 1))
            x1_ = x1 / self.lengthscale
            x2_ = x2 / self.lengthscale
            x1_, x2_ = self._create_input_grid(x1_, x2_, **params)

            all_diff = (x1_ - x2_)
            diff = all_diff.norm(2, dim=-1)
            all_diff = all_diff.squeeze(len(all_diff.size()) - 1)

            # 1) Kernel block
            K_11 = diff.pow(2).div(-2).exp()
            K[..., :n1, :n2] = K_11

            shape_list = [1] * len(K_11.shape[:-2])

            # 2) First gradient block
            K_rep = K_11.repeat(*(shape_list + [1, d]))
            all_K = (all_diff / self.lengthscale) * K_rep
            deriv_K = all_K.contiguous().view(d * n1, n2)
            K[..., :n1, n2:] = deriv_K

            # 3) Second gradient block
            K_rep = K_11.repeat(*(shape_list + [d, 1]))
            all_K = (all_diff / self.lengthscale) * K_rep
            deriv_K = all_K.permute(0, 2, 1).contiguous().view(d * n2, n1)
            K[..., n1:, :n2] = -deriv_K.transpose(0, 1)

            # 4) Hessian block
            outer_prod = all_diff.contiguous().view(d * n1, n2)
            outer_prod = - (outer_prod * outer_prod) / self.lengthscale.pow(2)
            I = torch.eye(d) / self.lengthscale.pow(2)
            outer_prod += I.repeat(shape_list + [n1, n2])
            repeated_K = K_11.repeat(shape_list + [d, d])
            K[..., n1:, n2:] = outer_prod * repeated_K

            pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().contiguous().view((n1 * (d + 1)))
            pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            K = K[..., pi1, :][..., :, pi2]

            return K
        else: # TODO: This will change when ARD is supported
            kernel_diag = super(RBFKernelGrad, self).forward(x1, x2, diag=True)
            grad_diag = (1 / self.lengthscale.item() ** 2) * torch.ones(1, n2*d)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            pi = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().contiguous().view((n2 * (d + 1)))
            return k_diag[..., pi]

    def _create_input_grid(self, x1, x2, diag=False, batch_dims=None, **params):  # TODO: Remove me please
        """
        This is a helper method for creating a grid of the kernel's inputs.
        Use this helper rather than maually creating a meshgrid.
        The grid dimensions depend on the kernel's evaluation mode.
        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`)
            :attr:`x2` (Tensor `m x d` or `b x m x d`) - for diag mode, these must be the same inputs
        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the gridded `x1` and `x2`.
            The shape depends on the kernel's mode
            * `full_covar`: (`b x n x 1 x d` and `b x 1 x m x d`)
            * `full_covar` with `batch_dims=(0, 2)`: (`b x k x n x 1 x 1` and `b x k x 1 x m x 1`)
            * `diag`: (`b x n x d` and `b x n x d`)
            * `diag` with `batch_dims=(0, 2)`: (`b x k x n x 1` and `b x k x n x 1`)
        """
        x1_, x2_ = x1, x2
        if batch_dims == (0, 2):
            x1_ = x1_.view(*x1.size()[:-1], -1, 1)
            x1_ = x1_.permute(0, -2, *list(range(1, x1_.dim() - 2)), -1).contiguous()
            x1_ = x1_.view(-1, *x1_.size()[2:])
            if torch.equal(x1, x2):
                x2_ = x1_
            else:
                x2_ = x2_.view(*x2.size()[:-1], -1, 1)
                x2_ = x2_.permute(0, -2, *list(range(1, x2_.dim() - 2)), -1).contiguous()
                x2_ = x2_.view(-1, *x2_.size()[2:])

        if diag:
            return x1_, x2_
        else:
            return x1_.unsqueeze(-2), x2_.unsqueeze(-3)

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
