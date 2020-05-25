#!/usr/bin/env python3

# TODO: this needs to be cleaned up to gpytorch standards.
# it probably won't load as is

import math

import torch

from . import Kernel


class SpectralGPKernel(Kernel):
    def __init__(
        self,
        train_inputs,
        latent_lh=None,
        latent_model=None,
        num_locs=50,
        period_factor=8.0,
        normalize=False,
        transform=torch.exp,
        register_latent_params=True,
        use_latent_model=False,
        **kwargs,
    ):
        r"""
        integration: {U, MC} U is trapezoidal rule, MC is Mone Carlo w/ w \sim U(-pi, pi)
        omega (default None)
        num_locs: number of omegas
        normalize: enforce that integral = 1
        transform: S(\omega) := transform(latent_params(\omega)), default: torch.exp
        """
        super(SpectralGPKernel, self).__init__(**kwargs)

        self.normalize = normalize
        self.transform = transform
        self.num_locs = num_locs
        self.period_factor = period_factor
        self.use_latent_model = use_latent_model

        if register_latent_params:
            self.register_parameter("latent_params", torch.nn.Parameter(torch.ones(self.num_locs)))
        else:
            self.latent_params = None

        # begin initialization by choosing omega
        self.omega = self._choose_frequencies(train_inputs, period_factor)

        if use_latent_model:
            # now initialize from data if we want to store a latent model
            self._initialize_from_data(train_inputs, latent_lh, latent_model, period_factor, **kwargs)

    def _choose_frequencies(self, train_inputs, period_factor=8.0):
        x1 = train_inputs.unsqueeze(-1)

        max_tau = x1.max() - x1.min()
        max_tau = period_factor * max_tau
        omega = math.pi * 2.0 * torch.arange(self.num_locs, dtype=x1.dtype, device=x1.device).div(max_tau)
        return omega

    def _initialize_from_data(
        self, train_inputs, latent_lh=None, latent_model=None, period_factor=8.0, latent_params_grad=True, **kwargs
    ):
        """
        Spectral methods require a bit of hand-initialization - this is analogous to the SM
        intialize_from_data method.
        train_inputs: train locations
        latent_lh: latent model's likliehood function, if non-standard
        latent_model: latent model
        period_factor: constant to multiply on period
        """
        from ..priors import GaussianProcessPrior

        device = train_inputs.device

        log_periodogram = torch.ones_like(self.omega)

        self.num_locs = len(self.omega)

        # if latent model is passed in, use that
        if latent_lh is None:
            from ..likelihoods import GaussianLikelihood
            from ..priors import SmoothedBoxPrior
            from ..constraints import Positive

            self.latent_lh = GaussianLikelihood(
                noise_prior=SmoothedBoxPrior(1e-8, 1e-3), noise_constraint=Positive()
            ).to(device)
        else:
            self.latent_lh = latent_lh

        if latent_model is None:
            from ..models import ExactGP
            from ..means import QuadraticMean
            from ..kernels import ScaleKernel, MaternKernel
            from ..constraints import Positive, LessThan
            from ..priors import LogNormalPrior
            from ..distributions import MultivariateNormal
            from ..likelihoods import GaussianLikelihood

            class ExactGPModel(ExactGP):
                def __init__(self, train_x, train_y, likelihood):
                    super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                    # Mean, covar, likelihood
                    self.mean_module = QuadraticMean(
                        input_size=1,
                        raw_quadratic_weights_constraint=LessThan(0.0),
                        raw_bias_constraint=Positive(),
                        use_weights=False,
                        bias_prior=LogNormalPrior(0.0, 0.3),
                    )

                    self.mean_module.register_prior(
                        "quadratic_weights_prior",
                        LogNormalPrior(0.0, 0.3),
                        lambda: -self.mean_module.quadratic_weights,
                        lambda x: self.mean_module.initialize(**{"quadratic_weights": -x}),
                    )
                    self.mean_module.quadratic_weights = -10.0
                    self.mean_module.bias = 1.0

                    self.covar_module = ScaleKernel(
                        MaternKernel(nu=1.5, lengthscale_prior=LogNormalPrior(0.0, 0.3)),
                        outputscale_prior=LogNormalPrior(0.0, 0.3),
                    )

                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return MultivariateNormal(mean_x, covar_x)

            latent_lh = GaussianLikelihood(noise_constraint=Positive())
            latent_lh.register_prior("latent_noise_prior", SmoothedBoxPrior(1e-5, 1e-4), "noise")

            latent_gp = ExactGPModel(self.omega, log_periodogram, latent_lh)
            self.register_prior("latent_gp_prior", GaussianProcessPrior(latent_gp), "latent_params")
        else:
            self.latent_model = latent_model
            # update the training data to include this set of omega and log_periodogram
            self.latent_model.set_train_data(self.omega, log_periodogram, strict=False)

        # self.latent_model.train()
        # self.latent_lh.train()

        # # set the latent g to be the demeaned periodogram
        # # and make it not require a gradient (we're using ESS for it)
        # if self.latent_params is None:
        #     self.latent_params = log_periodogram
        # else:
        #     self.latent_params.data = log_periodogram
        # if not latent_params_grad:
        #     self.latent_params.requires_grad = False

        # register prior for latent_params as latent mod
        # latent_prior = GaussianProcessPrior(self.latent_model, self.latent_lh)
        # self.register_prior("latent_gp_prior", latent_prior, lambda: self.latent_params)

        # return self.latent_lh, self.latent_model

    def compute_kernel_values(self, tau, density, normalize=False):
        # numerically compute integral in expanded space
        integrand = density * torch.cos(2.0 * math.pi * self.omega * tau)

        # compute trapezoidal rule for now
        # TODO: use torch.trapz and/or KeOps instead
        diff = self.omega[1:] - self.omega[:-1]
        integral = (diff * (integrand[..., 1:] + integrand[..., :-1]) / 2.0).sum(-1, keepdim=False)

        # divide by integral of density
        if normalize:
            norm_constant = (diff * (density[1:] + density[:-1]) / 2.0).sum()

            integral = integral / norm_constant

        return integral

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **kwargs):
        x1_ = x1
        x2_ = x1 if x2 is None else x2
        # batch_shape = x1.shape[:-2]
        n, num_dims = x1.shape[-2:]

        # Expand x1 and x2 to account for the number of mixtures
        tau = x1_[..., :, None, :] - x2_[..., None, :, :]

        # transform to enforce positivity
        density = self.transform(self.latent_params)  # .unsqueeze(1).unsqueeze(1)
        if len(density.shape) > 1:
            density = density.unsqueeze(1).unsqueeze(1)

        # TODO: more efficient diagonal
        output = self.compute_kernel_values(tau, density=density, normalize=self.normalize).squeeze(0)
        if diag:
            output = output.diag()
        return output

    def set_latent_params(self, new_latent_params):
        # updates latent parameters
        # warning: requires a .train(), .eval() call to empty the cache
        self.latent_params = new_latent_params
