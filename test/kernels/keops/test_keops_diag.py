#!/usr/bin/env python3

import unittest
import warnings

import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel as GMaternKernel, PeriodicKernel as GPeriodicKernel, RBFKernel as GRBFKernel
from gpytorch.kernels.keops import MaternKernel, PeriodicKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.models import ExactGP


class TestKeOpsDiagAndPredictiveVariance(unittest.TestCase):
    def test_diag_and_last_dim_is_batch_match_standard_kernels(self) -> None:
        torch.manual_seed(0)
        pairs = [
            (RBFKernel, GRBFKernel, {"ard_num_dims": 3}),
            (MaternKernel, GMaternKernel, {"nu": 1.5, "ard_num_dims": 3}),
            (PeriodicKernel, GPeriodicKernel, {"ard_num_dims": 3}),
        ]
        for keops_cls, std_cls, kwargs in pairs:
            for batch in ((), (2,)):
                with self.subTest(kernel=keops_cls.__name__, batch=batch):
                    k_keops = keops_cls(batch_shape=torch.Size(batch), **kwargs).double()
                    k_std = std_cls(batch_shape=torch.Size(batch), **kwargs).double()
                    k_std.load_state_dict(k_keops.state_dict())

                    x1 = torch.randn(*batch, 12, 3, dtype=torch.double, requires_grad=True)
                    x2 = torch.randn(*batch, 12, 3, dtype=torch.double, requires_grad=True)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        torch.testing.assert_close(k_keops(x1, diag=True), k_std(x1, diag=True))
                        torch.testing.assert_close(k_keops(x1, x2, diag=True), k_std(x1, x2, diag=True))
                        torch.testing.assert_close(
                            k_keops(x1, x2, last_dim_is_batch=True).to_dense(),
                            k_std(x1, x2, last_dim_is_batch=True).to_dense(),
                        )
                        torch.testing.assert_close(
                            k_keops(x1, x2, diag=True, last_dim_is_batch=True),
                            k_std(x1, x2, diag=True, last_dim_is_batch=True),
                        )

    def test_exact_gp_predictive_variance_with_scaled_keops_rbf(self) -> None:
        torch.manual_seed(1)
        train_x = torch.randn(16, 3, dtype=torch.double)
        train_y = torch.randn(16, dtype=torch.double)

        class KeOpsGP(ExactGP):
            def __init__(self) -> None:
                super().__init__(train_x, train_y, GaussianLikelihood())
                self.mean_module = LinearMean(input_size=3)
                self.covar_module = gpytorch.kernels.ScaleKernel(RBFKernel(ard_num_dims=3))

            def forward(self, x: torch.Tensor) -> MultivariateNormal:
                return MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = KeOpsGP().double().eval()
        model.likelihood.eval()
        test_x = torch.randn(25, 3, dtype=torch.double)
        with warnings.catch_warnings(), torch.no_grad(), gpytorch.settings.fast_pred_var():
            warnings.simplefilter("ignore", RuntimeWarning)
            observed_pred = model.likelihood(model(test_x))
            self.assertEqual(observed_pred.mean.shape, torch.Size([25]))
            self.assertEqual(observed_pred.variance.shape, torch.Size([25]))
            torch.testing.assert_close(
                observed_pred.variance,
                observed_pred.covariance_matrix.diagonal(dim1=-2, dim2=-1),
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
