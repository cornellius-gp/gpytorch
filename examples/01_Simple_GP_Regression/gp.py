import logging
import math
from collections import namedtuple

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.spectral_autoregressive_flow_kernel import RFNSSpectralNFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# Confidence intervals for a Gaussian likelihood
CI_95 = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1 - 0.05 / 2)).item()
CI_99 = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1 - 0.01 / 2)).item()

# Jitter added during training
JITTER = 1e-6

# Hyper bounds settings
Bounds = namedtuple("Bounds", ("lower", "upper"))
LENGTHSCALE_BOUNDS = Bounds(lower=0.01, upper=2.0)
SIGNAL_VAR_BOUNDS = Bounds(lower=0.05, upper=50.0)
NOISE_VAR_BOUNDS = Bounds(lower=1e-4, upper=0.5)

# Default hyper settings
SIGNAL_VAR = "covar_module.outputscale"
LENGTHSCALE = "covar_module.base_kernel.lengthscale"
NOISE_VAR = "likelihood.noise"
TORCH_DTYPE = torch.float64
TORCH_DEVICE = torch.device('cuda:0')


def standardize(y, return_parameters=False, copy=True):
    assert y.ndim == 1 and torch.all(torch.isfinite(y))
    if copy:
        y = y.clone()
    y_mean, y_std = y.mean().item(), y.std().item()
    if y.max() - y.min() <= 1e-10:
        y_std = 1.0  # We don't want to divide by zero

    if return_parameters:
        return (y - y_mean) / y_std, y_mean, y_std
    else:
        return (y - y_mean) / y_std


class _ExactGPModel(ExactGP):
    """Helper class for a GP model in GPyTorch."""

    def __init__(self, train_x, train_y, likelihood, ard_dims, nu, mean, jitter):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)
        if mean is None or mean == "zero":
            self.mean_module = ZeroMean()
        elif mean == "constant":
            self.mean_module = ConstantMean()
        elif mean == "linear":
            self.mean_module = LinearMean(train_x.shape[-1])
        else:
            raise RuntimeError("mean must be either None, 'zero', 'constant', or 'linear'")
        self.jitter = jitter
        self.ard_dims = ard_dims
        if train_x is None:
            base_covar_module = MaternKernel()
        else:
            base_covar_module = RFNSSpectralNFKernel(
                num_dims=train_x.size(-1),
                nonstationary=False,
            )
        self.covar_module = ScaleKernel(
            base_covar_module, outputscale_constraint=Interval(*SIGNAL_VAR_BOUNDS, transform=None)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.add_jitter(self.jitter)
        return MultivariateNormal(mean_x, covar_x)

    def sample_hypers_from_prior(self):
        hypers = {}
        # Draw the signal variance from a LogNormal(0, 1) distribution
        outputscale = torch.clamp(
            torch.distributions.LogNormal(0.0, 1.0).sample(torch.Size([1])),
            SIGNAL_VAR_BOUNDS.lower,
            SIGNAL_VAR_BOUNDS.upper,
        )
        hypers[SIGNAL_VAR] = outputscale

        # Draw the lengthscales from a top-hat distribution
        hypers[LENGTHSCALE] = torch.distributions.Uniform(LENGTHSCALE_BOUNDS.lower, LENGTHSCALE_BOUNDS.upper).sample(
            torch.Size([self.ard_dims])
        )

        # Draw the noise variance from a LogUniform[1e-4, 1e-1] distribution
        hypers[NOISE_VAR] = (
            torch.distributions.Uniform(math.log(NOISE_VAR_BOUNDS.lower), math.log(NOISE_VAR_BOUNDS.upper))
            .sample(torch.Size([1]))
            .exp()
        )

        # Return the hypers
        return hypers


def _params_to_np(model):
    params = []
    for param_name, param in model.named_parameters():
        params.append(param.view(-1))

    param_vec = torch.cat(params).cpu().detach().numpy().astype(np.float64)
    return param_vec


def _grads_to_np(model):
    param_grads = []
    for param_name, param in model.named_parameters():
        param_grads.append(param.grad.view(-1))

    grad_vec = torch.cat(param_grads).cpu().detach().numpy().astype(np.float64)
    return grad_vec


def clip_to_bounds_np(x, lb, ub):
    assert x.ndim == 1 and lb.ndim == 1 and ub.ndim == 1
    assert np.all(np.less_equal(lb, x) | np.isclose(lb, x))
    assert np.all(np.less_equal(x, ub) | np.isclose(x, ub))
    x = np.clip(x, lb, ub)
    return x


def _np_to_hypers(model, x):
    """Helper function for converting hypers from array to dict."""

    # Clip the values to the bounds in case they are slightly outside
    bounds = _gp_hyper_bounds(model)
    x = clip_to_bounds_np(x, bounds[:, 0], bounds[:, 1])

    result = {}

    i_start = 0
    for name, param in model.named_parameters():
        result[name] = torch.from_numpy(x[i_start : i_start + param.numel()])
        i_start = i_start + param.numel()

    return result


def _gp_hyper_bounds(model):
    """Helper function for extracting the bounds for the hypers."""
    all_bounds = []
    for name, param, constraint in model.named_parameters_and_constraints():
        if constraint is None:
            lb = -np.inf * torch.ones(param.view(-1).shape, dtype=param.dtype, device=param.device).unsqueeze(0)
            ub = np.inf * torch.ones(param.view(-1).shape, dtype=param.dtype, device=param.device).unsqueeze(0)
        else:
            lb = constraint.lower_bound.expand_as(param.view(-1)).unsqueeze(0)
            ub = constraint.upper_bound.expand_as(param.view(-1)).unsqueeze(0)

        bounds = torch.cat([lb, ub], dim=0).transpose(-2, -1)
        all_bounds.append(bounds)

    return torch.cat(all_bounds).cpu().detach().numpy()


def _gp_mll_objective(hypers_np, X, y, model):
    """Computes marginal log likelihood and derivatives wrt the hypers.

    Parameters
    ----------
    hypers_np : numpy.ndarray, shape (n_hypers,)
        Hyperparameters in the order matching `_hypers_to_np`.
    X : torch tensor, shape (n, d)
        Training inputs.
    y : torch tensor, shape (n,)
        Training outputs.
    model : _ExactGPModel
        Instance of GP model.

    Returns
    -------
    loss : float
        Value at the marginal log likelihood with input hypers.
    dloss : numpy.ndarray, shape (n_hypers,)
        Derivatives wrt the hyper parameters, ordered accordig to `_hypers_to_np`.
    """
    model.initialize(**_np_to_hypers(model, hypers_np))  # Set model hypers

    # Compute loss and back derivatives
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with gpytorch.settings.fast_computations(log_prob=False):
        model.zero_grad()  # Set gradients to zero
        output = model(X)
        loss = -mll(output, y)
        loss.backward()

    dloss = _grads_to_np(model)

    # Turn the loss into a scalar and turn the derivatives into size (n_hypers,)
    return loss.item(), dloss.ravel()


class SingleTaskGP(torch.nn.Module):
    def __init__(
        self, ard_dims=1, nu="2.5", mean=None, jitter=JITTER, standardize_y=True, dtype=TORCH_DTYPE, device=TORCH_DEVICE
    ):
        super().__init__()
        self.ard_dims = ard_dims
        self.nu = nu
        self.mean = mean
        self.jitter = jitter
        self.standardize_y = standardize_y
        self.register_buffer("_mean", torch.tensor(0.0, dtype=dtype, device=device))
        self.register_buffer("_std", torch.tensor(1.0, dtype=dtype, device=device))
        self.reset()

    def reset(self):
        """Reset current GP model to have no data."""
        self._mean.fill_(0.0)
        self._std.fill_(1.0)
        likelihood = GaussianLikelihood(noise_constraint=None).to(device=None, dtype=None)
        # likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
        model = _ExactGPModel(None, None, likelihood, self.ard_dims, self.nu, self.mean, self.jitter).to(
            device=None, dtype=None
        )
        self.model = model

    def train(self, train_x, train_y, n_iterations=50, n_init=50):
        """Create a new GP model and likelihood."""
        assert train_x.ndim == 2
        assert train_y.ndim == 1
        assert train_x.shape[0] == train_y.shape[0]
        assert self.ard_dims == 1 or self.ard_dims == train_x.shape[-1]
        assert n_init >= 1
        assert train_x.min() >= 0 and train_x.max() <= 1.0

        # Standardize data and save the parameters
        if self.standardize_y:
            train_y, mean, std = standardize(train_y, return_parameters=True, copy=True)
        else:
            mean, std = 0.0, 1.0
        self._mean.fill_(mean)
        self._std.fill_(std)

        # Create a new model and a likelihood
        new_likelihood = GaussianLikelihood(noise_constraint=Interval(*NOISE_VAR_BOUNDS, transform=None)).to(
            device=train_x.device, dtype=train_x.dtype
        )
        # new_likelihood.register_prior("noise_prior", gpytorch.priors.HorseshoePrior(0.1), "noise")
        new_model = _ExactGPModel(train_x, train_y, new_likelihood, self.ard_dims, self.nu, self.mean, self.jitter).to(
            device=train_x.device, dtype=train_y.dtype
        )
        self.model = new_model
        self.model.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Find a good starting point
        hypers_best, mll_best = {}, math.inf
        for i in range(n_init):
            if i == 0:
                hypers = {SIGNAL_VAR: 1.0, LENGTHSCALE: 0.5, NOISE_VAR: 0.01}
            else:
                hypers = self.model.sample_hypers_from_prior()
            self.model.initialize(**hypers)

            # Compute MLL
            with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
                output = self.model(train_x)
                mll_next = -mll(output, train_y)
                if mll_next < mll_best:
                    hypers_best, mll_best = hypers, mll_next

        # Initialize model from best hypers
        self.model.initialize(**hypers_best)

        # Find intial point and bounds
        x0 = _params_to_np(self.model)
        bounds = _gp_hyper_bounds(self.model)

        # Run scipy L-BFGS-B
        res = minimize(
            _gp_mll_objective, x0, args=(train_x, train_y, self.model), method="L-BFGS-B", jac=True, bounds=bounds
        )
        if res.status != 0:
            logger.warning("L-BFGS-B unexpected termination: {0}".format(res.message))
        x = res.x

        # Store the new hypers
        hypers = _np_to_hypers(self.model, x)

        # Set model hyper parameters
        self.model.initialize(**hypers)

        return self

    def predict(self, X, return_var=False, return_cov=False):
        """Predict from GP model (without noise) at X."""
        self.model.eval()

        with gpytorch.settings.debug(False), gpytorch.settings.fast_computations(
            log_prob=False, covar_root_decomposition=False, solves=False
        ):
            pred = self.model(X)
            assert not (return_var and return_cov), "Can't return both var and covar"
            if return_var:
                return self._mean + self._std * pred.mean, (self._std ** 2) * pred.variance
            elif return_cov:
                return self._mean + self._std * pred.mean, (self._std ** 2) * pred.covariance_matrix
            else:
                return self._mean + self._std * pred.mean

    def sample(self, X, n_samples=1, sample_y=True):
        """Sample GP model (without noise) at X."""
        self.model.eval()
        with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves=False):
            if sample_y:
                samples = self.model.likelihood(self.model(X)).rsample(torch.Size([n_samples])).t()
            else:
                pred = self.model(X)
                jittered_preds = MultivariateNormal(pred.mean, pred.lazy_covariance_matrix.add_jitter(1e-4))
                samples = jittered_preds.rsample(torch.Size([n_samples])).t()
                samples = self.model(X).rsample(torch.Size([n_samples])).t()
        return self._mean + self._std * samples

    def importance_weights(self):
        """Return an importance weight for each dimension."""
        weights = self.model.covar_module.base_kernel.lengthscale.clone().squeeze(0)
        weights = weights / weights.mean()  # Make them sum to dim
        return weights

    def recommend(self, test_x, n_recommendations=1):
        """Recommend the best points from a set of test points."""
        mu, var = self.predict(test_x, return_var=True)

        # Select the best points nad the mean + variance
        ind_sorted = torch.argsort(mu)[:n_recommendations]
        X_rec = test_x[ind_sorted, :].clone()
        mean_rec, std_rec = mu[ind_sorted], var[ind_sorted].sqrt()

        # Create dictionaries
        recommendations = []
        for i in range(n_recommendations):
            recommendations.append(
                {
                    "recommended_x": X_rec[i, :].unsqueeze(0),
                    "expected_y": mean_rec[i].item(),
                    "ci_95": ((mean_rec[i] - CI_95 * std_rec[i]).item(), (mean_rec[i] + CI_95 * std_rec[i]).item()),
                    "ci_99": ((mean_rec[i] - CI_99 * std_rec[i]).item(), (mean_rec[i] + CI_99 * std_rec[i]).item()),
                }
            )
        return recommendations
