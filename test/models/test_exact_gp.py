#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch import settings
from gpytorch.kernels import GridInterpolationKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import InterpolatedPredictionStrategy
from gpytorch.test.model_test_case import BaseModelTestCase

N_PTS = 50


class GridInterpolationKernelMock(GridInterpolationKernel):
    def __init__(self, should_use_wiski=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_use_wiski = should_use_wiski

    def prediction_strategy(self, *args, **kwargs):
        return InterpolatedPredictionStrategy(uses_wiski=self.should_use_wiski, *args, **kwargs)


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class InterpolatedExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, should_use_wiski=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = GridInterpolationKernelMock(
            base_kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            grid_size=128,
            num_dims=1,
            should_use_wiski=should_use_wiski,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SumExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        covar_a = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        covar_b = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        covar_c = gpytorch.kernels.LinearKernel()  # this one is important because its covariance matrix can be lazy
        self.covar_module = covar_a + covar_b + covar_c

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestExactGP(BaseModelTestCase, unittest.TestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactGPModel(train_x, train_y, likelihood)
        return model

    def create_test_data(self):
        return torch.randn(N_PTS, 1)

    def create_likelihood_and_labels(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        labels = torch.randn(N_PTS) + 2
        return likelihood, labels

    def create_batch_test_data(self, batch_shape=torch.Size([3])):
        return torch.randn(*batch_shape, N_PTS, 1)

    def create_batch_likelihood_and_labels(self, batch_shape=torch.Size([3])):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
        labels = torch.randn(*batch_shape, N_PTS) + 2
        return likelihood, labels

    def test_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_forward_eval()

    def test_batch_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_batch_forward_eval()

    def test_multi_batch_forward_eval_fast(self):
        with gpytorch.settings.max_eager_kernel_size(1), gpytorch.settings.fast_pred_var(True):
            self.test_multi_batch_forward_eval()

    def test_batch_forward_then_nonbatch_forward_eval(self):
        batch_data = self.create_batch_test_data()
        likelihood, labels = self.create_batch_likelihood_and_labels()
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)

        # Smoke test derivatives working
        output.mean.sum().backward()

        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Create non-batch data
        data = self.create_test_data()
        output = model(data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == data.size(-2))

        # Smoke test derivatives working
        output.mean.sum().backward()

    def test_batch_forward_then_different_batch_forward_eval(self):
        non_batch_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(non_batch_data, labels, likelihood)
        model.eval()

        # Batch size 3
        batch_data = self.create_batch_test_data()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now Batch size 2
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2]))
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 3 again
        batch_data = self.create_batch_test_data()
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 1
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([1]))
        output = model(batch_data)
        self.assertTrue(output.lazy_covariance_matrix.dim() == 3)
        self.assertTrue(output.lazy_covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.lazy_covariance_matrix.size(-2) == batch_data.size(-2))

    def test_prior_mode(self):
        train_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        prior_model = self.create_model(None, None, likelihood)
        model = self.create_model(train_data, labels, likelihood)
        prior_model.eval()
        model.eval()

        test_data = self.create_test_data()
        prior_out = prior_model(test_data)
        with gpytorch.settings.prior_mode(True):
            prior_out_cm = model(test_data)
        self.assertTrue(torch.allclose(prior_out.mean, prior_out_cm.mean))
        self.assertTrue(torch.allclose(prior_out.covariance_matrix, prior_out_cm.covariance_matrix))

    def test_lanczos_fantasy_model(self):
        lanczos_thresh = 10
        n = lanczos_thresh + 1
        n_dims = 2
        with settings.max_cholesky_size(lanczos_thresh):
            x = torch.ones((n, n_dims))
            y = torch.randn(n)
            likelihood = GaussianLikelihood()
            model = ExactGPModel(x, y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            mll.train()
            mll.eval()

            # get a posterior to fill in caches
            model(torch.randn((1, n_dims)))

            new_n = 2
            new_x = torch.randn((new_n, n_dims))
            new_y = torch.randn(new_n)
            # just check that this can run without error
            model.get_fantasy_model(new_x, new_y)


def _build_trained_exact_gp(n: int = 20, seed: int = 0, dtype: torch.dtype = torch.float64) -> ExactGPModel:
    """Helper: build a trained 1D ExactGPModel with fixed, non-degenerate hyperparameters.

    Non-trivial lengthscale/outputscale/noise (not near-identity, not near-zero) so the
    posterior covariance isn't a degenerate case that would mask a formula bug.
    """
    gen = torch.Generator().manual_seed(seed)
    train_x = torch.rand(n, 1, generator=gen, dtype=dtype)
    train_y = torch.sin(3 * train_x.squeeze(-1)) + 0.05 * torch.randn(n, generator=gen, dtype=dtype)
    likelihood = GaussianLikelihood().to(dtype=dtype)
    model = ExactGPModel(train_x, train_y, likelihood).to(dtype=dtype)
    model.covar_module.base_kernel.lengthscale = 0.3
    model.covar_module.outputscale = 1.2
    likelihood.noise = 0.04
    model.mean_module.constant.data.fill_(0.2)
    model.eval()
    likelihood.eval()
    return model


def _reference_posterior(model: ExactGPModel, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute posterior mean and covariance from first principles.

    Returns (mean, covar) matching the raw GP MVN (no outcome transform / likelihood wrapping).
    """
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    noise = model.likelihood.noise_covar.noise.squeeze()
    K_TT = model.covar_module(train_x).to_dense()
    K_Tt = model.covar_module(train_x, test_x).to_dense()
    K_tt = model.covar_module(test_x).to_dense()
    eye = torch.eye(K_TT.shape[-1], dtype=K_TT.dtype, device=K_TT.device)
    K_TT_noisy = K_TT + noise * eye
    mean_T = model.mean_module(train_x)
    mean_t = model.mean_module(test_x)
    alpha = torch.linalg.solve(K_TT_noisy, (train_y - mean_T).unsqueeze(-1))
    K_Tt_solved = torch.linalg.solve(K_TT_noisy, K_Tt)
    mean_ref = (K_Tt.transpose(-1, -2) @ alpha).squeeze(-1) + mean_t
    cov_ref = K_tt - K_Tt.transpose(-1, -2) @ K_Tt_solved
    return mean_ref, cov_ref


class TestExactPredictiveCovar(unittest.TestCase):
    """Parity of posterior mean/covar values and gradients against a from-scratch reference."""

    def test_posterior_matches_math_reference(self):
        # Covers non-batch and batched test inputs. Checks values and gradients w.r.t.
        # test_x. Values protect the forward formula; gradients protect the autograd path
        # (catches a backward bug even when the forward happens to agree by cancellation).
        for test_shape in [(10, 1), (4, 10, 1)]:
            model = _build_trained_exact_gp(n=20)
            gen = torch.Generator().manual_seed(1)
            base = torch.rand(*test_shape, generator=gen, dtype=torch.float64)

            test_x_got = base.clone().requires_grad_(True)
            model.prediction_strategy = None
            out = model(test_x_got)
            (out.mean.sum() + out.covariance_matrix.sum()).backward()

            test_x_ref = base.clone().requires_grad_(True)
            mean_ref, cov_ref = _reference_posterior(model, test_x_ref)
            (mean_ref.sum() + cov_ref.sum()).backward()

            self.assertTrue(
                torch.allclose(out.mean, mean_ref, atol=1e-10, rtol=0),
                f"mean diff {(out.mean - mean_ref).abs().max().item():.3e} for shape {test_shape}",
            )
            self.assertTrue(
                torch.allclose(out.covariance_matrix, cov_ref, atol=1e-10, rtol=0),
                f"covar diff {(out.covariance_matrix - cov_ref).abs().max().item():.3e} for shape {test_shape}",
            )
            self.assertTrue(
                torch.allclose(test_x_got.grad, test_x_ref.grad, atol=1e-9, rtol=0),
                f"grad diff {(test_x_got.grad - test_x_ref.grad).abs().max().item():.3e} for shape {test_shape}",
            )


class TestInterpolatedExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = InterpolatedExactGPModel(train_x, train_y, likelihood)
        return model


class TestWiskiExactGP(TestInterpolatedExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = InterpolatedExactGPModel(train_x, train_y, likelihood, should_use_wiski=True)
        return model

    def test_fantasy_model(self):
        x = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(x, labels, likelihood)
        test_x = self.create_test_data()
        _, test_labels = self.create_likelihood_and_labels()
        with torch.no_grad():
            model.eval()
            model(test_x)
        new_model = model.get_fantasy_model(test_x, test_labels)

        self.assertEqual(type(new_model), type(model))
        self.assertTrue(new_model.prediction_strategy.uses_wiski)

    def test_nonbatch_to_batch_fantasy_model(self, batch_shape=torch.Size([3])):
        x = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(x, labels, likelihood)
        test_x = self.create_batch_test_data(batch_shape=batch_shape)
        _, test_labels = self.create_batch_likelihood_and_labels(batch_shape=batch_shape)
        with torch.no_grad():
            model.eval()
            model(test_x)
        new_model = model.get_fantasy_model(test_x, test_labels)

        self.assertEqual(type(new_model), type(model))
        self.assertTrue(new_model.prediction_strategy.uses_wiski)

    def test_nonbatch_to_multibatch_fantasy_model(self):
        self.test_nonbatch_to_batch_fantasy_model(batch_shape=torch.Size([2, 3]))


class TestSumExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = SumExactGPModel(train_x, train_y, likelihood)
        return model

    def test_cache_across_lazy_threshold(self):
        x = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(x, labels, likelihood)
        model.eval()
        model(x)  # populate caches

        with settings.max_eager_kernel_size(2 * N_PTS - 1), settings.fast_pred_var(True):
            # now we'll cross the threshold and use lazy tensors
            new_x = self.create_test_data()
            _, new_y = self.create_likelihood_and_labels()
            model = model.get_fantasy_model(new_x, new_y)
            predicted = model(self.create_test_data())

            # the main purpose of the test was to ensure there was no error, but we can verify shapes too
            self.assertEqual(predicted.mean.shape, torch.Size([N_PTS]))
            self.assertEqual(predicted.variance.shape, torch.Size([N_PTS]))


class TestExactGPHooks(unittest.TestCase):
    """Tests for the new modular prediction hooks in ExactGP."""

    def test_get_train_prior_distribution_hook(self):
        """Test that _get_train_prior_distribution can be overridden."""
        call_count = [0]

        class CustomExactGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            def _get_train_prior_distribution(self, train_inputs, **kwargs):
                call_count[0] += 1
                return super()._get_train_prior_distribution(train_inputs, **kwargs)

        train_x = torch.randn(N_PTS, 1)
        train_y = torch.randn(N_PTS)
        likelihood = GaussianLikelihood()
        model = CustomExactGP(train_x, train_y, likelihood)
        model.eval()

        # First call should invoke _get_train_prior_distribution
        test_x = torch.randn(10, 1)
        _ = model(test_x)
        self.assertEqual(call_count[0], 1)

        # Second call should use cached prediction_strategy
        _ = model(test_x)
        self.assertEqual(call_count[0], 1)

    def test_get_test_prior_mean_and_covariances_hook(self):
        """Test that _get_test_prior_mean_and_covariances can be overridden."""
        call_count = [0]

        class CustomExactGP(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            def _get_test_prior_mean_and_covariances(self, train_inputs, test_inputs, **kwargs):
                call_count[0] += 1
                return super()._get_test_prior_mean_and_covariances(train_inputs, test_inputs, **kwargs)

        train_x = torch.randn(N_PTS, 1)
        train_y = torch.randn(N_PTS)
        likelihood = GaussianLikelihood()
        model = CustomExactGP(train_x, train_y, likelihood)
        model.eval()

        # Each prediction call should invoke _get_test_prior_mean_and_covariances
        test_x = torch.randn(10, 1)
        _ = model(test_x)
        self.assertEqual(call_count[0], 1)

        # Second call should also invoke it (not cached)
        _ = model(test_x)
        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    unittest.main()
