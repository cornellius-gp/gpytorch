import torch

from gpytorch.kernels import ProductKernel
from torch.nn import ModuleList

from .spectral_gp_kernel import SpectralGPKernel

class ProductSpectralGPKernel(ProductKernel):
    def __init__(self, train_x, train_y, shared, **kwargs):
        kernels = []

        for d in range(0,train_x.size(-1)):
            kernels.append(SpectralGPKernel(active_dims=torch.tensor([d]), **kwargs))

        super(ProductSpectralGPKernel, self).__init__(*kernels)

        for d in range(0,train_x.size(-1)):
            self.kernels[d].initialize_from_data(train_x[:,d], train_y, **kwargs)

        self.shared = shared

        if self.shared:
            for d in range(1, train_x.size(-1)):
                self.kernels[d].latent_mod = self.kernels[0].latent_mod
                self.kernels[d].latent_lh = self.kernels[0].latent_lh
                self.kernels[d].latent_params = self.kernels[0].latent_params
                max_omega = self.kernels[0].omega[-1]
                max_omega_idk = 0
                for idk,_ in enumerate(self.kernels):
                    if self.kernels[idk].omega[-1] > max_omega:
                        max_omega = self.kernels[idk].omega[-1]
                        max_omega_idk = idk
                print(max_omega_idk)
                print(max_omega)
                self.kernels[d].omega = self.kernels[max_omega_idk].omega

    def get_latent_mod(self, idx=None):
        # print(hex(id(self.kernels[idx].latent_mod)))
        return self.kernels[idx].latent_mod

    def get_latent_lh(self, idx=None):
        return self.kernels[idx].latent_lh

    def get_omega(self, idx=None):
        return self.kernels[idx].omega

    def get_latent_params(self, idx=None):
        # print(hex(id(self.kernels[idx].latent_params)))
        return self.kernels[idx].latent_params

    def set_latent_params(self, x, idx=None):
        self.kernels[idx].set_latent_params(x)
