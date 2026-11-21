#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP


class _SimpleGP(ExactGP):
    def __init__(self, train_inputs: torch.Tensor, train_targets: torch.Tensor) -> None:
        batch_shape = train_inputs.shape[:-2]
        super().__init__(train_inputs, train_targets, GaussianLikelihood(batch_shape=batch_shape))
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = RBFKernel(batch_shape=batch_shape)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestFantasyBatchMemory(unittest.TestCase):
    def test_broadcast_rhs_predictions_match_dense_and_preserve_gradients(self) -> None:
        torch.manual_seed(0)
        train_x = torch.rand(3, 8, 2, dtype=torch.double)
        train_y = torch.randn(3, 8, dtype=torch.double)
        gp = _SimpleGP(train_x, train_y).double().eval()
        gp(torch.rand(3, 2, 2, dtype=torch.double))

        fant_x = torch.rand(3, 2, 2, dtype=torch.double, requires_grad=True)
        fant_y = torch.randn(4, 3, 2, dtype=torch.double, requires_grad=True)

        test_x = torch.rand(5, 4, 3, 2, 2, dtype=torch.double, requires_grad=True)
        for fast_var in (True, False):
            with self.subTest(fast_pred_var=fast_var):
                with gpytorch.settings.fast_pred_var(fast_var), gpytorch.settings.detach_test_caches(False):
                    fantasy_model = gp.get_fantasy_model(fant_x, fant_y).eval()
                    pred = fantasy_model(test_x)
                    # Reference: evaluate each leading slice independently (no extra batch broadcast)
                    ref_means = []
                    ref_covars = []
                    for i in range(test_x.size(0)):
                        sub_pred = fantasy_model(test_x[i])
                        ref_means.append(sub_pred.mean)
                        ref_covars.append(sub_pred.covariance_matrix)
                    expected_mean = torch.stack(ref_means, dim=0)
                    expected_covar = torch.stack(ref_covars, dim=0)

                    torch.testing.assert_close(pred.mean, expected_mean, rtol=1e-10, atol=1e-10)
                    torch.testing.assert_close(pred.covariance_matrix, expected_covar, rtol=1e-10, atol=1e-10)

                    loss = pred.mean.sin().sum() + 0.3 * pred.covariance_matrix.square().sum()
                    ref_loss = expected_mean.sin().sum() + 0.3 * expected_covar.square().sum()
                    grads = torch.autograd.grad(loss, (test_x, fant_x, fant_y), retain_graph=True)
                    ref_grads = torch.autograd.grad(ref_loss, (test_x, fant_x, fant_y))
                    for g, rg in zip(grads, ref_grads, strict=True):
                        torch.testing.assert_close(g, rg, rtol=1e-9, atol=1e-9)

    def test_large_multi_batch_fantasy_posterior_avoids_broadcast_oom(self) -> None:
        # Reproducer from cornellius-gp/gpytorch#2652: previously tried to allocate 66.8 GB
        # by broadcasting precomputed_cache (256, 101, 101) to (50, 64, 256, 101, 101).
        torch.manual_seed(1)
        gp = (
            _SimpleGP(
                train_inputs=torch.rand(100, 2, dtype=torch.double),
                train_targets=torch.rand(100, dtype=torch.double),
            )
            .double()
            .eval()
        )
        gp(torch.rand(5, 2, dtype=torch.double))

        num_fantasies = 64
        x_fant = torch.rand(256, 1, 2, dtype=torch.double)
        y_fant = torch.rand(num_fantasies, 256, 1, dtype=torch.double)
        fantasy_model = gp.get_fantasy_model(x_fant, y_fant).eval()

        x_test = torch.rand(50, num_fantasies, 256, 1, 2, dtype=torch.double)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = fantasy_model(x_test)
            self.assertEqual(posterior.mean.shape, torch.Size([50, num_fantasies, 256, 1]))
            self.assertEqual(posterior.variance.shape, torch.Size([50, num_fantasies, 256, 1]))

            # Spot-check slice [7, 13, 19] against unbatched slice call
            sub_post = fantasy_model(x_test[7])
            torch.testing.assert_close(posterior.mean[7], sub_post.mean, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(posterior.variance[7], sub_post.variance, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
