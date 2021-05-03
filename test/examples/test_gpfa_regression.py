#!/usr/bin/env python3

import os
import random
import unittest
import numpy as np

import gpytorch
import torch
from gpytorch.kernels import GPFAKernel
from gpytorch.lazy import DiagLazyTensor, KroneckerProductLazyTensor, KroneckerProductDiagLazyTensor


def generate_GPFA_Data(seed=0,
                       n_timepoints=100,
                       n_latents=2,
                       num_obs=50,
                       length_scales=[.01, 10],
                       start_time=-5,
                       end_time=5,
                       zero_mean=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    timepoints = torch.linspace(start_time, end_time, n_timepoints)
    tau = torch.tensor(length_scales)
    C = torch.tensor(
        np.random.normal(scale=1. / np.sqrt(n_latents),
                         size=(num_obs, n_latents))).float()
    if zero_mean:
        d = torch.zeros(size=(num_obs, 1)).float()
    else:
        d = torch.tensor(np.random.uniform(size=(num_obs, 1))).float()
    R = torch.tensor(np.diag(np.random.uniform(size=(num_obs, ),
                                               low=.1))).float()
    kernels = [gpytorch.kernels.RBFKernel() for t in tau]
    for t in range(len(tau)):
        kernels[t].lengthscale = tau[t]

    xs = torch.stack([
        gpytorch.distributions.MultivariateNormal(torch.zeros(n_timepoints),
                                                  k(timepoints,
                                                    timepoints)).sample()
        for k in kernels
    ])

    ys = gpytorch.distributions.MultivariateNormal((C @ xs + d).T, R).sample()

    xs = xs.T.contiguous()

    return timepoints, tau, C, d, R, kernels, xs, ys


n_latents = 2
num_obs = 20
n_timepoints = 100
length_scales = [.1, .2]
start_time = 0
end_time = 1
train_x, tau, C, d, R, kernels, xs, train_y = generate_GPFA_Data(
    seed=10,
    n_timepoints=n_timepoints,
    n_latents=n_latents,
    num_obs=num_obs,
    length_scales=length_scales,
    start_time=start_time,
    end_time=end_time,
    zero_mean=True)

# For now, just test that GPFA more or less recovers the noiseless version of train_y
test_y = (C @ xs.T).T


class GPFAModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, latent_covar_modules,
                 num_latents, num_obs):
        super(GPFAModel, self).__init__(train_x, train_y, likelihood)

        self.num_latents = num_latents
        self.num_obs = num_obs

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_obs)
        self.covar_module = GPFAKernel(latent_covar_modules, num_latents,
                                       num_obs)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(x), self.covar_module(x))

    # Not currently used in this test. TODO: Test recovery of latents
    def latent_posterior(self, x):
        r'''
        See equations 4 and 5 in `Non-reversible Gaussian processes for
        identifying latent dynamical structure in neural data`_

        .. _Non-reversible Gaussian processes for identifying latent dynamical structure in neural data:
        https://papers.nips.cc/paper/2020/file/6d79e030371e47e6231337805a7a2685-Paper.pdf
        '''
        I_t = DiagLazyTensor(torch.ones(len(x)))
        combined_noise = (self.likelihood.task_noises if self.likelihood.has_task_noise
                          else torch.zeros(self.likelihood.num_tasks)) + (
                              self.likelihood.noise
                              if self.likelihood.has_global_noise else 0)
        Kyy = self.covar_module(x) + KroneckerProductDiagLazyTensor(
            I_t, DiagLazyTensor(combined_noise))

        Kxx = self.covar_module.latent_covar_module(x)

        C_Kron_I = KroneckerProductLazyTensor(I_t, self.covar_module.C)

        mean_rhs = (train_y - self.mean_module(x)).view(
            *(train_y.numel(),
              ))  # vertically stacks after doing the subtraction

        latent_mean = Kxx @ C_Kron_I.t() @ Kyy.inv_matmul(mean_rhs)
        latent_mean = latent_mean.view(*(len(x),
                                         int(latent_mean.shape[0] / len(x))))

        cov_rhs = C_Kron_I @ Kxx
        latent_cov = Kxx - Kxx @ C_Kron_I.t() @ Kyy.inv_matmul(
            cov_rhs.evaluate())
        return gpytorch.distributions.MultitaskMultivariateNormal(
            latent_mean, latent_cov)


class TestGPFARegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv(
                "UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_multitask_gp_mean_abs_error(self):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_obs)
        kernels = [gpytorch.kernels.RBFKernel() for t in range(n_latents)]
        model = GPFAModel(train_x, train_y, likelihood, kernels, n_latents,
                          num_obs)
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        n_iter = 50
        for _ in range(n_iter):
            # Zero prev backpropped gradients
            optimizer.zero_grad()
            # Make predictions from training data
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Test the model predictions on noiseless test_ys
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model.eval()
            likelihood.eval()
            preds = likelihood(model(train_x))
            pred_mean = preds.mean
            mean_abs_error = torch.mean(torch.abs(test_y - pred_mean), axis=0)
            self.assertFalse(torch.sum(mean_abs_error > (2 * torch.diagonal(R))))

    def test_multitask_gp_mean_abs_error_one_kernel(self):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_obs)
        model = GPFAModel(train_x, train_y, likelihood, gpytorch.kernels.RBFKernel(), n_latents,
                          num_obs)
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        n_iter = 50
        for _ in range(n_iter):
            # Zero prev backpropped gradients
            optimizer.zero_grad()
            # Make predictions from training data
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Test the model predictions on noiseless test_ys
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model.eval()
            likelihood.eval()
            preds = likelihood(model(train_x))
            pred_mean = preds.mean
            mean_abs_error = torch.mean(torch.abs(test_y - pred_mean), axis=0)
            self.assertFalse(torch.sum(mean_abs_error > (2 * torch.diagonal(R))))


if __name__ == "__main__":
    unittest.main()
