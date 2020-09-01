import unittest

import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import CholLazyTensor, TriangularLazyTensor
from gpytorch.variational import NaturalVariationalDistribution, TrilNaturalVariationalDistribution


class Float64Test(unittest.TestCase):
    def setUp(self):
        self.prev_type = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.prev_type)


class TestNatVariational(Float64Test):
    def test_invertible_init(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D).tril_()
        dist = MultivariateNormal(mu, CholLazyTensor(TriangularLazyTensor(cov)))

        v_dist = NaturalVariationalDistribution(D, mean_init_std=0.0)
        v_dist.initialize_variational_distribution(dist)

        out_dist = v_dist()

        assert torch.allclose(out_dist.mean, dist.mean, rtol=1e-04, atol=1e-06)
        assert torch.allclose(out_dist.covariance_matrix, dist.covariance_matrix)

    def test_natgrad(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D).tril_()
        dist = MultivariateNormal(mu, CholLazyTensor(TriangularLazyTensor(cov)))
        sample = dist.sample()

        v_dist = NaturalVariationalDistribution(D)
        v_dist.initialize_variational_distribution(dist)
        mu = v_dist().mean.detach()

        v_dist().log_prob(sample).squeeze().backward()

        eta1 = mu.clone().requires_grad_(True)
        eta2 = (mu[:, None] * mu + cov @ cov.t()).requires_grad_(True)
        L = torch.cholesky(eta2 - eta1[:, None] * eta1)
        dist2 = MultivariateNormal(eta1, CholLazyTensor(TriangularLazyTensor(L)))
        dist2.log_prob(sample).squeeze().backward()

        assert torch.allclose(v_dist.natural_vec.grad, eta1.grad)
        assert torch.allclose(v_dist.natural_mat.grad, eta2.grad)

    def test_optimization_optimal_error(self, num_inducing=16, num_data=32, D=2):
        inducing_points = torch.randn(num_inducing, D)

        class SVGP(gpytorch.models.ApproximateGP):
            def __init__(self):
                v_dist = NaturalVariationalDistribution(num_inducing)
                v_strat = gpytorch.variational.UnwhitenedVariationalStrategy(self, inducing_points, v_dist)
                super().__init__(v_strat)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.RBFKernel()

            def forward(self, x):
                return MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = SVGP().train()
        likelihood = gpytorch.likelihoods.GaussianLikelihood().train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data)
        X = torch.randn((num_data, D))
        y = torch.randn(num_data)

        def loss():
            return -mll(model(X), y)

        optimizer = torch.optim.SGD(
            model.variational_strategy._variational_distribution.parameters(), lr=float(num_data)
        )

        optimizer.zero_grad()
        loss().backward()
        optimizer.step()  # Now we should be at the optimum

        optimizer.zero_grad()
        loss().backward()
        natgrad_natural_vec2, natgrad_natural_mat2 = (
            model.variational_strategy._variational_distribution.natural_vec.grad.clone(),
            model.variational_strategy._variational_distribution.natural_mat.grad.clone(),
        )
        # At the optimum, the (natural) gradients are zero:
        assert torch.allclose(natgrad_natural_vec2, torch.zeros(()))
        assert torch.allclose(natgrad_natural_mat2, torch.zeros(()))


class TestTrilNatVariational(Float64Test):
    def test_invertible_init(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D).tril_()
        dist = MultivariateNormal(mu, CholLazyTensor(TriangularLazyTensor(cov)))

        v_dist = TrilNaturalVariationalDistribution(D, mean_init_std=0.0)
        v_dist.initialize_variational_distribution(dist)

        out_dist = v_dist()

        assert torch.allclose(out_dist.mean, dist.mean)
        assert torch.allclose(out_dist.covariance_matrix, dist.covariance_matrix)

    def test_natgrad(self, D=5):
        mu = torch.randn(D)
        cov = torch.randn(D, D)
        cov = cov @ cov.t()
        dist = MultivariateNormal(mu, CholLazyTensor(TriangularLazyTensor(cov.cholesky())))
        sample = dist.sample()

        v_dist = TrilNaturalVariationalDistribution(D, mean_init_std=0.0)
        v_dist.initialize_variational_distribution(dist)
        v_dist().log_prob(sample).squeeze().backward()
        dout_dnat1 = v_dist.natural_vec.grad
        dout_dnat2 = v_dist.natural_tril_mat.grad

        # mean_init_std=0. because we need to ensure both have the same distribution
        v_dist_ref = NaturalVariationalDistribution(D, mean_init_std=0.0)
        v_dist_ref.initialize_variational_distribution(dist)
        v_dist_ref().log_prob(sample).squeeze().backward()
        dout_dnat1_noforward_ref = v_dist_ref.natural_vec.grad
        dout_dnat2_noforward_ref = v_dist_ref.natural_mat.grad

        def f(natural_vec, natural_tril_mat):
            "Transform natural_tril_mat to L"
            Sigma = torch.inverse(-2 * natural_tril_mat)
            mu = natural_vec
            return mu, Sigma.cholesky().inverse().tril()

        (mu_ref, natural_tril_mat_ref), (dout_dmu_ref, dout_dnat2_ref) = jvp(
            f,
            (v_dist_ref.natural_vec.detach(), v_dist_ref.natural_mat.detach()),
            (dout_dnat1_noforward_ref, dout_dnat2_noforward_ref),
        )

        assert torch.allclose(natural_tril_mat_ref, v_dist.natural_tril_mat), "Sigma transformation"
        assert torch.allclose(dout_dnat2_ref, dout_dnat2), "Sigma gradient"

        assert torch.allclose(mu_ref, v_dist.natural_vec), "mu transformation"
        assert torch.allclose(dout_dmu_ref, dout_dnat1), "mu gradient"


def jvp(f, x, v):
    "Simulate forward-mode AD using two reverse-mode AD"
    x = tuple(xx.requires_grad_(True) for xx in x)
    v = tuple(vv.requires_grad_(True) for vv in v)
    y = f(*x)
    grad_x = torch.autograd.grad(y, x, v, create_graph=True)
    jvp_val = torch.autograd.grad(grad_x, v, v)
    return y, jvp_val
