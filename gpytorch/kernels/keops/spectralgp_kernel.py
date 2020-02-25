import torch
import math

from ...lazy import KeOpsLazyTensor
from .keops_kernel import KeOpsKernel

#from . import Kernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class SpectralGPKernel(KeOpsKernel):
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
                self.register_parameter("latent_params", torch.nn.Parameter(torch.zeros(self.num_locs)))
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

                self.latent_lh = GaussianLikelihood(noise_prior=SmoothedBoxPrior(1e-8, 1e-3)).to(device)
            else:
                self.latent_lh = latent_lh

            if latent_model is None:
                from ..means import QuadraticMean
                from ..priors import NormalPrior
                from ..constraints import LessThan
                from ..kernels import ScaleKernel, MaternKernel
                from ..models import PyroVariationalGP
                from ..distributions import MultivariateNormal
                from ..variational import CholeskyVariationalDistribution, VariationalStrategy

                # we construct a Scale(Matern 3/2) kernel with a quadratic mean by default
                class PyroGPModel(PyroGP):
                    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
                        # Define all the variational stuff
                        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=int(train_x.numel()))
                        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)

                        super(PyroGPModel, self).__init__(variational_strategy, likelihood, num_data=train_x.numel())
                        self.mean_module = mean_module
                        self.covar_module = covar_module

                    def forward(self, x):
                        mean = self.mean_module(x)
                        covar = self.covar_module(x)
                        return MultivariateNormal(mean, covar)

                # construct by default a LogRBF prior on the latent spectral GP
                latent_mean = QuadraticMean().to(device)
                latent_mean.register_prior(
                    "bias_prior",
                    prior=NormalPrior(torch.zeros(1, device=device), 100.0 * torch.ones(1, device=device), transform=None),
                    param_or_closure="bias",
                )
                latent_mean.register_constraint(
                    "quadratic_weights", constraint=LessThan(upper_bound=0.0),
                )
                latent_mean.register_prior(
                    "quadratic_weights_prior",
                    prior=NormalPrior(
                        torch.zeros(1, device=device),
                        100.0 * torch.ones(1, device=device),
                        transform=torch.nn.functional.softplus,
                    ),
                    param_or_closure="quadratic_weights",
                )

                latent_covar = ScaleKernel(
                    MaternKernel(nu=1.5, lengthscale_prior=NormalPrior(torch.zeros(1), torch.ones(1), transform=torch.exp)),
                    outputscale_prior=NormalPrior(torch.zeros(1), torch.ones(1), transform=torch.exp),
                )

                self.latent_model = PyroGPModel(
                    self.omega, log_periodogram, self.latent_lh, mean_module=latent_mean, covar_module=latent_covar
                )
            else:
                self.latent_model = latent_model
                # update the training data to include this set of omega and log_periodogram
                self.latent_model.set_train_data(self.omega, log_periodogram, strict=False)

            self.latent_model.train()
            self.latent_lh.train()

            # set the latent g to be the demeaned periodogram
            # and make it not require a gradient (we're using ESS for it)
            if self.latent_params is None:
                self.latent_params = log_periodogram
            else:
                self.latent_params.data = log_periodogram
            if not latent_params_grad:
                self.latent_params.requires_grad = False

            # register prior for latent_params as latent mod
            latent_prior = GaussianProcessPrior(self.latent_model, self.latent_lh)
            self.register_prior("latent_prior", latent_prior, lambda: self.latent_params)

            return self.latent_lh, self.latent_model

        def covar_func(self, x1, x2, omega, density, diag=False):
            if not diag:
                # this is trapezoid rule
                with torch.autograd.enable_grad():
                    x1_ = KEOLazyTensor(x1[..., :, None, :])
                    x2_ = KEOLazyTensor(x2[..., None, :, :])
                    density = KEOLazyTensor(density)
                    integrand = ((x1_ - x2_) * (2 * math.pi * omega)).cos() * density

                    diff = omega[1:] - omega[:-1]
                    integral = (integrand[1:] + integrand[:(integrand.shape[-1]-1)]) / 2.0 * diff

                    return integral.sum(-1)
            else:
                # we will do this the somewhat expensive way
                tau = x1 - x2
                integrand = (tau * (2 * math.pi * omega)).cos() * density
                return torch.trapz(integrand, omega)

        def forward(self, x1, x2, diag=False, **params):
            x1_ = x1
            x2_ = x2

            # transform to enforce positivity
            density = self.transform(self.latent_params)
            if len(density.shape) > 1:
                density = density.unsqueeze(1).unsqueeze(1)

            integral = KeOpsLazyTensor(x1_, x2_, lambda x1, x2, diag=False: self.covar_func(x1, x2, self.omega, density, diag=diag))
            if self.normalize:
                norm_constant = torch.trapz(density, self.omega)

                integral = integral / norm_constant
            
            if not diag:
                return integral
            else:
                return integral.diag()

        def set_latent_params(self, new_latent_params):
            # updates latent parameters
            # warning: requires a .train(), .eval() call to empty the cache
            self.latent_params = new_latent_params
            
except ImportError:

    class SpectralGPKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__()