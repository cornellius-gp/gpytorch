#!/usr/bin/env python3

import torch

from . import ProductKernel, GridInterpolationKernel, ProductStructureKernel, SpectralGPKernel
from ..utils.grid import choose_grid_size
from ..lazy import lazify, delazify, LazyTensor

class WrappedSpectralGPKernel(ProductKernel):
    def __init__(self, train_x, train_y, shared=True, transform=torch.exp,
                 omega=None, **kwargs):
        kernels = []

        for d in range(0,train_x.size(-1)):
            if omega is not None:
                print("passing omega")
                kernels.append(SpectralGPKernel(omega=omega, **kwargs))
            else:
                kernels.append(SpectralGPKernel(**kwargs))

        super(WrappedSpectralGPKernel, self).__init__(*kernels)

        # print(self.kernels[0])
        self.base_kernels = [kern for kern in self.kernels]
        for d in range(0,train_x.size(-1)):
            self.base_kernels[d].initialize_from_data(train_x[:,d], train_y, **kwargs)

        self.transform = transform
        self.shared = shared

        if self.shared:
            for d in range(1, train_x.size(-1)):
                self.base_kernels[d].latent_mod = self.base_kernels[0].latent_mod
                self.base_kernels[d].latent_lh = self.base_kernels[0].latent_lh
                self.base_kernels[d].latent_params = torch.nn.Parameter(self.base_kernels[0].latent_params.clone())
                max_omega = self.base_kernels[0].omega[-1]
                max_omega_idk = 0
                for idk,_ in enumerate(self.base_kernels):
                    if self.base_kernels[idk].omega[-1] > max_omega:
                        max_omega = self.base_kernels[idk].omega[-1]
                        max_omega_idk = idk

                self.base_kernels[d].omega = self.base_kernels[max_omega_idk].omega

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        x1_ = x1
        x2_ = x1 if x2 is None else x2
        # print("x1.shape", x1_.shape)
        # print("x2.shape", x2_.shape)

        if last_dim_is_batch:
            # print("calling good kernel")
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            tau = x1_ - x2_.transpose(-2, -1)
            # print("tau", tau.shape)

            output = torch.zeros_like(tau)
            for dim in range(len(self.base_kernels)):
                dens = self.base_kernels[dim].get_latent_params()
                dens = self.transform(dens)
                output[dim, :, :] = self.base_kernels[dim].compute_kernel_values(tau[dim, :, :], density=dens)

            return output

        else:

            # start matrix #
            res = self.base_kernels[0](x1[:, 0], x2[:, 0],
                                                diag=diag, **kwargs)
            if isinstance(res, LazyTensor):
                res = delazify(res)

            # multiply in other dimensions #
            for ind, kern in enumerate(self.base_kernels[1:]):
                next_term = kern(x1[:, ind+1], x2[:, ind+1],
                                 diag=diag)
                if isinstance(next_term, LazyTensor):
                    next_term = delazify(next_term)

                res = res * next_term

            return res

    def get_latent_mod(self, idx=None):
        return self.base_kernels[idx].latent_mod

    def get_latent_lh(self, idx=None):
        return self.base_kernels[idx].latent_lh

    def get_omega(self, idx=None):
        return self.base_kernels[idx].omega

    def get_latent_params(self, idx=None):
        return self.base_kernels[idx].latent_params

    def set_latent_params(self, x, idx=None):
        self.base_kernels[idx].set_latent_params(x)
