#!/usr/bin/env python3

import unittest
import warnings
from math import exp, pi

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior, UniformPrior
from gpytorch.constraints import Positive
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device
from torch import optim


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestSimpleGPRegression(BaseTestCase, unittest.TestCase):
    seed = 1

    def _get_data(self, cuda=False, num_data=11, add_noise=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        # Simple training data: let's try to learn a sine function
        train_x = torch.linspace(0, 1, num_data, device=device)
        train_y = torch.sin(train_x * (2 * pi))
        if add_noise:
            train_y.add_(torch.randn_like(train_x).mul_(0.1))
        test_x = torch.linspace(0, 1, 51, device=device)
        test_y = torch.sin(test_x * (2 * pi))
        return train_x, test_x, train_y, test_y

    def test_prior(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to be ridiculous
        likelihood = GaussianLikelihood(
            noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1),
            noise_constraint=Positive(),  # Prior for this test is looser than default bound
        )
        gp_model = ExactGPModel(None, None, likelihood)
        # Update lengthscale prior to accommodate extreme parameters
        gp_model.covar_module.base_kernel.register_prior(
            "lengthscale_prior", SmoothedBoxPrior(exp(-10), exp(10), sigma=0.5), "raw_lengthscale"
        )
        gp_model.mean_module.initialize(constant=1.5)
        gp_model.covar_module.base_kernel.initialize(lengthscale=1)
        likelihood.initialize(noise=0)

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Compute posterior distribution
        gp_model.eval()
        likelihood.eval()

        # The model should predict in prior mode
        function_predictions = likelihood(gp_model(train_x))
        correct_variance = gp_model.covar_module.outputscale + likelihood.noise

        self.assertAllClose(function_predictions.mean, torch.full_like(function_predictions.mean, fill_value=1.5))
        self.assertAllClose(
            function_predictions.variance, correct_variance.squeeze().expand_as(function_predictions.variance)
        )

    def test_prior_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_prior(cuda=True)

    def test_recursive_initialize(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)

        likelihood_1 = GaussianLikelihood()
        gp_model_1 = ExactGPModel(train_x, train_y, likelihood_1)

        likelihood_2 = GaussianLikelihood()
        gp_model_2 = ExactGPModel(train_x, train_y, likelihood_2)

        gp_model_1.initialize(**{"likelihood.noise": 1e-2, "covar_module.base_kernel.lengthscale": 1e-1})
        gp_model_2.likelihood.initialize(noise=1e-2)
        gp_model_2.covar_module.base_kernel.initialize(lengthscale=1e-1)
        self.assertTrue(torch.equal(gp_model_1.likelihood.noise, gp_model_2.likelihood.noise))
        self.assertTrue(
            torch.equal(
                gp_model_1.covar_module.base_kernel.lengthscale, gp_model_2.covar_module.base_kernel.lengthscale
            )
        )

    def test_posterior_latent_gp_and_likelihood_without_optimization(self, cuda=False):
        warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to be ridiculous
        likelihood = GaussianLikelihood(noise_constraint=Positive())  # This test actually wants a noise < 1e-4
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        gp_model.covar_module.base_kernel.initialize(lengthscale=exp(-15))
        likelihood.initialize(noise=exp(-15))

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Compute posterior distribution
        gp_model.eval()
        likelihood.eval()

        # Let's see how our model does, conditioned with weird hyperparams
        # The posterior should fit all the data
        with gpytorch.settings.debug(False):
            function_predictions = likelihood(gp_model(train_x))

        self.assertAllClose(function_predictions.mean, train_y)
        self.assertAllClose(function_predictions.variance, torch.zeros_like(function_predictions.variance))

        # It shouldn't fit much else though
        test_function_predictions = gp_model(torch.tensor([1.1]).type_as(test_x))

        self.assertAllClose(test_function_predictions.mean, torch.zeros_like(test_function_predictions.mean))
        self.assertAllClose(
            test_function_predictions.variance,
            gp_model.covar_module.outputscale.expand_as(test_function_predictions.variance),
        )

    def test_posterior_latent_gp_and_likelihood_without_optimization_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_gp_and_likelihood_without_optimization(cuda=True)

    def test_gp_posterior_mean_skip_variances_fast_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            train_x, test_x, train_y, _ = self._get_data(cuda=True)
            likelihood = GaussianLikelihood()
            gp_model = ExactGPModel(train_x, train_y, likelihood)

            gp_model.cuda()
            likelihood.cuda()

            # Compute posterior distribution
            gp_model.eval()
            likelihood.eval()

            with gpytorch.settings.skip_posterior_variances(True):
                mean_skip_var = gp_model(test_x).mean
            mean = gp_model(test_x).mean
            likelihood_mean = likelihood(gp_model(test_x)).mean

            self.assertTrue(torch.allclose(mean_skip_var, mean))
            self.assertTrue(torch.allclose(mean_skip_var, likelihood_mean))

    def test_gp_posterior_mean_skip_variances_slow_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            train_x, test_x, train_y, _ = self._get_data(cuda=True)
            likelihood = GaussianLikelihood()
            gp_model = ExactGPModel(train_x, train_y, likelihood)

            gp_model.cuda()
            likelihood.cuda()

            # Compute posterior distribution
            gp_model.eval()
            likelihood.eval()

            with gpytorch.settings.fast_pred_var(False):
                with gpytorch.settings.skip_posterior_variances(True):
                    mean_skip_var = gp_model(test_x).mean
                mean = gp_model(test_x).mean
                likelihood_mean = likelihood(gp_model(test_x)).mean
            self.assertTrue(torch.allclose(mean_skip_var, mean))
            self.assertTrue(torch.allclose(mean_skip_var, likelihood_mean))

    def test_gp_posterior_single_training_point_smoke_test(self):
        train_x, test_x, train_y, _ = self._get_data()
        train_x = train_x[0].unsqueeze(-1).unsqueeze(-1)
        train_y = train_y[0].unsqueeze(-1)
        likelihood = GaussianLikelihood()
        gp_model = ExactGPModel(train_x, train_y, likelihood)

        gp_model.eval()
        likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            preds = gp_model(test_x)
            single_mean = preds.mean
            single_variance = preds.variance

        self.assertFalse(torch.any(torch.isnan(single_variance)))
        self.assertFalse(torch.any(torch.isnan(single_mean)))

        gp_model.train()
        gp_model.eval()

        preds = gp_model(test_x)
        single_mean = preds.mean
        single_variance = preds.variance

        self.assertFalse(torch.any(torch.isnan(single_variance)))
        self.assertFalse(torch.any(torch.isnan(single_mean)))

    def test_posterior_latent_gp_and_likelihood_with_optimization(self, cuda=False, checkpoint=0):
        train_x, test_x, train_y, test_y = self._get_data(
            cuda=cuda, num_data=(1000 if checkpoint else 11), add_noise=bool(checkpoint)
        )
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(noise=exp(1))

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.15)
        with gpytorch.beta_features.checkpoint_kernel(checkpoint), gpytorch.settings.fast_pred_var():
            for _ in range(20 if checkpoint else 50):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()

        # Test the model
        gp_model.eval()
        likelihood.eval()
        with gpytorch.settings.skip_posterior_variances(True):
            test_function_predictions = likelihood(gp_model(test_x))
        mean_abs_error = torch.mean(torch.abs(test_y - test_function_predictions.mean))

        self.assertLess(mean_abs_error.item(), 0.05)

    def test_gp_with_checkpointing(self, cuda=False):
        return self.test_posterior_latent_gp_and_likelihood_with_optimization(cuda=cuda, checkpoint=250)

    def test_fantasy_updates_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_fantasy_updates(cuda=True)

    def test_fantasy_updates(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood()
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(noise=exp(1))

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.15)
        for _ in range(50):
            optimizer.zero_grad()
            with gpytorch.settings.debug(False):
                output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in gp_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        train_x.requires_grad = True
        gp_model.set_train_data(train_x, train_y)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches(False):
            # Test the model
            gp_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(gp_model(test_x))
            test_function_predictions.mean.sum().backward()

            real_fant_x_grad = train_x.grad[5:].clone()
            train_x.grad = None
            train_x.requires_grad = False
            gp_model.set_train_data(train_x, train_y)

            # Cut data down, and then add back via the fantasy interface
            gp_model.set_train_data(train_x[:5], train_y[:5], strict=False)
            likelihood(gp_model(test_x))

            fantasy_x = train_x[5:].clone().detach().requires_grad_(True)
            fant_model = gp_model.get_fantasy_model(fantasy_x, train_y[5:])
            fant_function_predictions = likelihood(fant_model(test_x))

            self.assertAllClose(test_function_predictions.mean, fant_function_predictions.mean, atol=1e-4)

            fant_function_predictions.mean.sum().backward()
            self.assertTrue(fantasy_x.grad is not None)

            relative_error = torch.norm(real_fant_x_grad - fantasy_x.grad) / fantasy_x.grad.norm()
            self.assertLess(relative_error, 15e-1)  # This was only passing by a hair before

    def test_fantasy_updates_batch_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_fantasy_updates_batch(cuda=True)

    def test_fantasy_updates_batch(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood()
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(noise=exp(1))

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.15)
        for _ in range(50):
            optimizer.zero_grad()
            with gpytorch.settings.debug(False):
                output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        for param in gp_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        with gpytorch.settings.fast_pred_var():
            # Test the model
            gp_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(gp_model(test_x))

            # Cut data down, and then add back via the fantasy interface
            gp_model.set_train_data(train_x[:5], train_y[:5], strict=False)
            likelihood(gp_model(test_x))

            fantasy_x = train_x[5:].clone().unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1).requires_grad_(True)
            fantasy_y = train_y[5:].unsqueeze(0).repeat(3, 1)
            fant_model = gp_model.get_fantasy_model(fantasy_x, fantasy_y)
            fant_function_predictions = likelihood(fant_model(test_x))

            self.assertAllClose(test_function_predictions.mean, fant_function_predictions.mean[0], atol=1e-4)

            fant_function_predictions.mean.sum().backward()
            self.assertTrue(fantasy_x.grad is not None)

    def test_posterior_latent_gp_and_likelihood_with_optimization_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_gp_and_likelihood_with_optimization(cuda=True)

    def test_posterior_with_exact_computations(self):
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False):
            self.test_posterior_latent_gp_and_likelihood_with_optimization(cuda=False)

    def test_posterior_with_exact_computations_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False):
                    self.test_posterior_latent_gp_and_likelihood_with_optimization(cuda=True)

    def test_posterior_latent_gp_and_likelihood_fast_pred_var(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to
            # something they shouldn't be
            likelihood = GaussianLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
            gp_model = ExactGPModel(train_x, train_y, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            gp_model.covar_module.base_kernel.initialize(lengthscale=exp(1))
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(noise=exp(1))

            if cuda:
                gp_model.cuda()
                likelihood.cuda()

            # Find optimal model hyperparameters
            gp_model.train()
            likelihood.train()
            optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
            for _ in range(50):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()

            # Test the model
            gp_model.eval()
            likelihood.eval()
            # Set the cache
            test_function_predictions = likelihood(gp_model(train_x))

            # Now bump up the likelihood to something huge
            # This will make it easy to calculate the variance
            likelihood.noise_covar.raw_noise.data.fill_(3)
            test_function_predictions = likelihood(gp_model(train_x))

            noise = likelihood.noise_covar.noise
            var_diff = (test_function_predictions.variance - noise).abs()

            self.assertLess(torch.max(var_diff / noise), 0.05)

    def test_pyro_sampling(self):
        try:
            import pyro  # noqa
            from pyro.infer.mcmc import NUTS, MCMC
        except ImportError:
            return
        train_x, test_x, train_y, test_y = self._get_data(cuda=False)
        likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        gp_model = ExactGPModel(train_x, train_y, likelihood)

        # Register normal GPyTorch priors
        gp_model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
        gp_model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
        gp_model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
        likelihood.register_prior("noise_prior", UniformPrior(0.05, 0.3), "noise")

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        def pyro_model(x, y):
            gp_model.pyro_sample_from_prior()
            output = gp_model(x)
            mll.pyro_factor(output, y)
            return y

        nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
        mcmc_run = MCMC(nuts_kernel, num_samples=3, warmup_steps=20)
        mcmc_run.run(train_x, train_y)

        gp_model.pyro_load_from_samples(mcmc_run.get_samples())

        gp_model.eval()
        expanded_test_x = test_x.unsqueeze(-1).repeat(3, 1, 1)
        output = gp_model(expanded_test_x)

        self.assertEqual(output.mean.size(0), 3)

        # All 3 samples should do reasonably well on a noiseless dataset.
        self.assertLess(torch.norm(output.mean[0] - test_y) / test_y.norm(), 0.2)
        self.assertLess(torch.norm(output.mean[1] - test_y) / test_y.norm(), 0.2)
        self.assertLess(torch.norm(output.mean[2] - test_y) / test_y.norm(), 0.2)

    def test_posterior_latent_gp_and_likelihood_fast_pred_var_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_gp_and_likelihood_fast_pred_var(cuda=True)


if __name__ == "__main__":
    unittest.main()
