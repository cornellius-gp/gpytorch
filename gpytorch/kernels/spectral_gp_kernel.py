#!/usr/bin/env python3

#TODO: this needs to be cleaned up to gpytorch standards.
# it probably won't load as is

import math
import torch
import copy

from scipy.signal import periodogram, welch
from scipy.interpolate import interp1d

from torch.nn import ModuleList

from . import Kernel, ScaleKernel, MaternKernel, GridKernel
from ..likelihoods import GaussianLikelihood
from ..priors import SmoothedBoxPrior, NormalPrior
from ..models import ExactGP
from ..distributions import MultivariateNormal
from ..means import LogRBFMean, ZeroMean
#from ..utils import spectral_init


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=ZeroMean, grid = True):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean()

        self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5,
                    lengthscale_prior = NormalPrior(torch.zeros(1), torch.ones(1),
                                            transform=torch.exp)
                    ),
                    outputscale_prior = NormalPrior(torch.zeros(1), torch.ones(1),
                                            transform=torch.exp)
                )

        self.grid = grid
        if self.grid:
            self.grid_module = GridKernel(self.covar_module, grid = train_x.unsqueeze(1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        if self.grid:
            covar_x = self.grid_module(x)
        else:
            covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)

def spectral_init(x, y, spacing, num_freq=500, omega_lim=None,
                    rtn_oneside=True):
    if spacing == 'even':
        f, den = periodogram(y.cpu().numpy(), fs=1/(x[1] - x[0]).item(),
                            return_onesided=rtn_oneside)
    elif spacing == 'random':
        interp_x = torch.linspace(x.min(), x.max()-1e-6, num_freq)
        interp_y = interp1d(x.cpu().numpy(), y.cpu().numpy())(interp_x.cpu().numpy())

        cand = 0.0
        idx = 0
        while cand < 1.e-23:
            cand = (interp_x[idx+1] - interp_x[idx])
            idx += 1
        f, den = periodogram(interp_y, fs=1/(cand).item(), return_onesided=rtn_oneside)

    ## get omegas ##
    print(omega_lim)
    if omega_lim is not None:
        lw_bd = max(min(f), omega_lim[0])
        up_bd = min(max(f), omega_lim[1])
        omega = torch.linspace(lw_bd + 1e-7, up_bd - 1e-7, num_freq)
    else:
        omega = torch.linspace(min(f)+1e-6, 0.5 * (max(f)-1e-6), num_freq)

    interp_den = interp1d(f, den)(omega.cpu().numpy())
    interp_den += 1e-10
    log_periodogram = torch.Tensor(interp_den).log()

    return omega, log_periodogram

class SpectralGPKernel(Kernel):
    def __init__(self, integration='U', omega = None, num_locs = 50, omega_max = 0.2,
                    normalize = False, transform = torch.exp, symmetrize = False,
                    register_latent_params = True,
                    **kwargs):
        r"""
        integration: {U, MC} U is trapezoidal rule, MC is Mone Carlo w/ w \sim U(-pi, pi)
        omega (default None)
        num_locs: number of omegas
        normalize: enforce that integral = 1
        transform: S(\omega) := transform(latent_params(\omega)), default: torch.exp
        symmetrize: use g^*(\omega) := 1/2 * (g(-\omega) + g(\omega))
        """
        super(SpectralGPKernel, self).__init__(**kwargs)

        self.normalize = normalize
        self.integration = integration
        self.transform = transform
        self.symmetrize = symmetrize

        if omega is None:
            if integration == 'U' and omega_max is not None:
                omega = torch.linspace(0., omega_max, num_locs)
            else:
                omega = torch.randn(num_locs) * math.pi
           
        self.register_parameter('omega', torch.nn.Parameter(omega))
        self.num_locs = len(omega)
        self.omega.requires_grad = False
        
        if register_latent_params:
            self.register_parameter('latent_params', torch.nn.Parameter(torch.zeros(self.num_locs)))
        else:
            self.latent_params = None

    def compute_kernel_values(self, tau, density, integration='U', normalize=True):

        # expand tau \in \mathbb{batch x n x n x M}
        expanded_tau = tau.unsqueeze(-1)
        if len(tau.size()) == 3:
            dims = [1,1,1,self.num_locs]
        else:
            dims = [1,1,self.num_locs]
        expanded_tau = expanded_tau.repeat(*dims)#.float()

        # numerically compute integral in expanded space
        integrand = density * torch.cos(2.0 * math.pi * self.omega * expanded_tau)

        if integration == 'MC':
            # take mean
            integral = integrand.mean(-1)
        elif integration == 'U':
            # compute trapezoidal rule for now
            diff = self.omega[1:] - self.omega[:-1]
            integral = (diff * (integrand[...,1:] + integrand[...,:-1]) / 2.0).sum(-1,keepdim=False)

            # divide by integral of density
            if normalize:
                norm_constant = (diff * (density[1:] + density[:-1]) / 2.0).sum()

                integral = integral / norm_constant

        return integral

    def initialize_from_data(self, train_x, train_y, num_locs=100, omega_max=None, spacing='random',
            latent_lh = None, latent_mod = None, pretrain = False, nonstat = False, **kwargs):
        #from ..models import ExactGPModel
        from ..priors import GaussianProcessPrior
        #from ..trainer import trainer

        # get omega and periodogram
        if omega_max is not None:
            omega_lim = (1.e-10, omega_max)
        else:
            omega_lim = None

        self.omega.data, log_periodogram = spectral_init(train_x, train_y, spacing, num_freq=num_locs, omega_lim=omega_lim)

        if nonstat:
            self.omega.data = torch.linspace(1.e-10, omega_max, num_locs)
            log_periodogram = torch.ones_like(self.omega)


        self.num_locs = len(self.omega)

        # if latent model is passed in, use that
        if latent_lh is None:
            self.latent_lh = GaussianLikelihood(noise_prior=SmoothedBoxPrior(1e-8, 1e-3))
        else:
            self.latent_lh = latent_lh

        if latent_mod is None:
            self.latent_mod = ExactGPModel(self.omega, log_periodogram, self.latent_lh, mean=LogRBFMean)
        else:
            self.latent_mod = latent_mod
            #update the training data to include this set of omega and log_periodogram
            self.latent_mod.set_train_data(self.omega, log_periodogram, strict=False)

        self.latent_mod.train()
        self.latent_lh.train()

        # now fit the latent model to the log periodogram
        if pretrain:
            trainer(self.omega, log_periodogram, self.latent_mod, self.latent_lh)

        self.latent_mod.train()
        self.latent_lh.train()

        # set the latent g to be the demeaned periodogram
        # and make it not require a gradient (we're using ESS for it)
        self.latent_params.data = log_periodogram
        self.latent_params.requires_grad = False

        # clear cache and reset training data
        self.latent_mod.set_train_data(inputs = self.omega, targets=self.latent_params.data, strict=False)

        # register prior for latent_params as latent mod
        latent_prior = GaussianProcessPrior(self.latent_mod, self.latent_lh)
        self.register_prior('latent_prior', latent_prior, lambda: self.latent_params)
        # print('max of self.omega', self.omega.max())
        return self.latent_lh, self.latent_mod

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False,
                **kwargs):
        x1_ = x1
        x2_ = x1 if x2 is None else x2
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        tau = x1_ - x2_.transpose(-2, -1)

        # transform to enforce positivity
        density = self.transform(self.latent_params)

        if self.symmetrize:
            density = 0.5 * (density + torch.flip(density, [0]))

        output = self.compute_kernel_values(tau, density=density,
                                            integration=self.integration,
                                            normalize=self.normalize)
        if diag:
            output = output.diag()
        return output

    def get_latent_mod(self, idx=None):
        return self.latent_mod

    def get_latent_lh(self, idx=None):
        return self.latent_lh

    def get_omega(self, idx=None):
        return self.omega

    def get_latent_params(self, idx=None):
        return self.latent_params

    def set_latent_params(self, g, idx=None):
        self.latent_params.data = g
