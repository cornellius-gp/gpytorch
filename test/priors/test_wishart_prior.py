#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import Wishart

from gpytorch.priors import InverseWishartPrior, WishartPrior


class TestWishartPrior(unittest.TestCase):
    def test_wishart_log_prob_matches_torch_wishart(self) -> None:
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.float64):
            for batch in ((), (2,)):
                for nu_val in (3.0, 5.5):  # includes boundary-adjacent nu = n = 3 > n - 1 = 2
                    with self.subTest(dtype=dtype, batch=batch, nu=nu_val):
                        a = torch.randn(*batch, 3, 3, dtype=dtype)
                        scale = a @ a.mT + torch.eye(3, dtype=dtype)
                        b = torch.randn(*batch, 3, 3, dtype=dtype)
                        x = (b @ b.mT + 0.5 * torch.eye(3, dtype=dtype)).requires_grad_(True)

                        prior = WishartPrior(nu=nu_val, K=scale)
                        ref = Wishart(df=torch.tensor(nu_val, dtype=dtype), covariance_matrix=scale)

                        actual = prior.log_prob(x)
                        expected = ref.log_prob(x)
                        tol = 1e-5 if dtype == torch.float32 else 1e-12
                        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

                        grad_actual = torch.autograd.grad(actual.sum(), x)[0]
                        grad_expected = torch.autograd.grad(expected.sum(), x)[0]
                        torch.testing.assert_close(grad_actual, grad_expected, rtol=tol, atol=tol)

        with self.assertRaises(ValueError):
            WishartPrior(nu=2.0, K=torch.eye(3))

    def test_inverse_wishart_log_prob_matches_jacobian_change_of_variables(self) -> None:
        torch.manual_seed(1)
        n = 3
        nu = 4.2
        a = torch.randn(2, n, n, dtype=torch.double)
        scale = a @ a.mT + torch.eye(n, dtype=torch.double)
        b = torch.randn(2, n, n, dtype=torch.double)
        sigma = (b @ b.mT + torch.eye(n, dtype=torch.double)).requires_grad_(True)

        iw = InverseWishartPrior(nu=nu, K=scale)
        # If W = Sigma^-1 ~ Wishart(df=nu + n - 1, covariance_matrix=K^-1),
        # then log p_IW(Sigma) = log p_W(Sigma^-1) - (n + 1) log |Sigma|.
        w_ref = Wishart(df=torch.tensor(nu + n - 1, dtype=torch.double), precision_matrix=scale)
        expected = w_ref.log_prob(torch.linalg.inv(sigma)) - (n + 1) * torch.logdet(sigma)
        actual = iw.log_prob(sigma)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

        with self.assertRaises(ValueError):
            InverseWishartPrior(nu=2.0, K=torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
