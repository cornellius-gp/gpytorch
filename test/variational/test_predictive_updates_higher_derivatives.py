import unittest
from unittest.mock import patch

import torch

import gpytorch
from gpytorch.variational.variational_strategy import ComputePredictiveUpdates


class _DenseRBF(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **kwargs):
        if diag:
            return (-0.5 * ((x1 - x2) / self.lengthscale).square().sum(-1)).exp()
        difference = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / self.lengthscale.unsqueeze(-2)
        return (-0.5 * difference.square().sum(-1)).exp()


class _VariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, points, batch_shape):
        distribution = gpytorch.variational.CholeskyVariationalDistribution(3, batch_shape=batch_shape, mean_init_std=0)
        strategy = gpytorch.variational.VariationalStrategy(self, points, distribution)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = _DenseRBF(batch_shape=batch_shape)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _direct_updates(chol, cross, middle, inducing_values):
    interpolation = torch.linalg.solve_triangular(chol, cross, upper=False)
    mean = (interpolation.mT @ inducing_values.unsqueeze(-1)).squeeze(-1)
    variance = (interpolation * (middle @ interpolation)).sum(-2)
    return mean, variance


class TestPredictiveUpdatesHigherDerivatives(unittest.TestCase):
    def test_mixed_second_and_third_derivatives(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch_shape in ((), (2,)):
                with self.subTest(device=device, batch_shape=batch_shape):
                    torch.manual_seed(32)
                    chol = (torch.randn(*batch_shape, 3, 3, device=device, dtype=torch.double) * 0.1).tril()
                    chol = (chol + 2 * torch.eye(3, device=device, dtype=torch.double)).requires_grad_()
                    cross = torch.randn(*batch_shape, 3, 5, device=device, dtype=torch.double, requires_grad=True)
                    middle = torch.randn(*batch_shape, 3, 3, device=device, dtype=torch.double, requires_grad=True)
                    inducing_values = torch.randn(
                        *batch_shape, 3, device=device, dtype=torch.double, requires_grad=True
                    )
                    params = (chol, cross, middle, inducing_values)
                    actual = ComputePredictiveUpdates.apply(chol, cross, (middle + middle.mT) / 2, inducing_values)
                    expected = _direct_updates(chol, cross, (middle + middle.mT) / 2, inducing_values)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b)
                    actual_loss = actual[0].sin().sum() + actual[1].square().sum()
                    expected_loss = expected[0].sin().sum() + expected[1].square().sum()
                    for order in range(3):
                        observed = torch.autograd.grad(actual_loss, params, create_graph=order < 2, retain_graph=True)
                        reference = torch.autograd.grad(
                            expected_loss, params, create_graph=order < 2, retain_graph=True
                        )
                        for a, b in zip(observed, reference):
                            torch.testing.assert_close(a, b, rtol=2e-7, atol=2e-8)
                        if order < 2:
                            actual_loss = sum(g.square().sum() for g in observed)
                            expected_loss = sum(g.square().sum() for g in reference)

    def test_gradcheck_and_gradgradcheck_broadcast_inputs(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                torch.manual_seed(34)
                chol = torch.eye(2, device=device, dtype=torch.double).unsqueeze(0).requires_grad_()
                cross = torch.randn(2, 2, 4, device=device, dtype=torch.double, requires_grad=True)
                middle = torch.randn(1, 2, 2, device=device, dtype=torch.double, requires_grad=True)
                inducing_values = torch.randn(2, device=device, dtype=torch.double, requires_grad=True)

                def evaluate(root, covar, correction, values):
                    return ComputePredictiveUpdates.apply(root, covar, (correction + correction.mT) / 2, values)

                args = (chol, cross, middle, inducing_values)
                self.assertTrue(torch.autograd.gradcheck(evaluate, args, fast_mode=True))
                self.assertTrue(torch.autograd.gradgradcheck(evaluate, args, fast_mode=True))

    def test_first_backward_reuses_forward_interpolation(self):
        chol = torch.eye(2, dtype=torch.double, requires_grad=True)
        cross = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        middle = torch.eye(2, dtype=torch.double, requires_grad=True)
        mean = torch.randn(2, dtype=torch.double, requires_grad=True)
        with patch.object(torch.linalg, "solve_triangular", wraps=torch.linalg.solve_triangular) as solve:
            updates = ComputePredictiveUpdates.apply(chol, cross, middle, mean)
            self.assertEqual(solve.call_count, 1)
            torch.autograd.grad(sum(update.sum() for update in updates), (chol, cross, middle, mean))
            self.assertEqual(solve.call_count, 3)

    def test_training_elbo_all_leaf_hessians(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for batch_shape in ((), (2,)):
                for likelihood_type in (
                    gpytorch.likelihoods.GaussianLikelihood,
                    gpytorch.likelihoods.LaplaceLikelihood,
                ):
                    with self.subTest(device=device, batch_shape=batch_shape, likelihood=likelihood_type.__name__):
                        torch.manual_seed(36)
                        x = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]], device=device, dtype=torch.double)
                        x = x.expand(*batch_shape, 5, 1).clone().requires_grad_()
                        y = x.new_tensor([0.2, 0.7, 0.4, 0.8, 0.3]).expand(*batch_shape, 5).clone().requires_grad_()
                        model = _VariationalGP(x.detach()[..., [0, 2, 4], :].clone(), batch_shape).to(x)
                        likelihood = likelihood_type(batch_shape=batch_shape).to(x)
                        model.covar_module.lengthscale = 0.5
                        model.mean_module.constant = 0.13
                        strategy = model.variational_strategy
                        distribution = strategy._variational_distribution
                        with torch.no_grad():
                            strategy.variational_params_initialized.fill_(1)
                            distribution.variational_mean.copy_(x.new_tensor([-0.3, 0.1, 0.4]))
                            distribution.chol_variational_covar.copy_(
                                torch.eye(3, device=x.device, dtype=x.dtype) * 0.8
                            )
                        output = model(x)
                        actual = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=11, beta=0.7)(output, y)
                        points = strategy.inducing_points
                        kernel = model.covar_module
                        kzz = kernel.forward(points, points) + strategy.jitter_val * torch.eye(
                            3, device=x.device, dtype=x.dtype
                        )
                        interpolation = torch.linalg.solve_triangular(
                            torch.linalg.cholesky(kzz), kernel.forward(points, x), upper=False
                        )
                        mu, root = distribution.variational_mean, distribution.chol_variational_covar.tril()
                        covariance = root @ root.mT
                        mean = model.mean_module(x) + (interpolation.mT @ mu.unsqueeze(-1)).squeeze(-1)
                        variance = (
                            1
                            + strategy.jitter_val
                            + (
                                interpolation
                                * ((covariance - torch.eye(3, device=x.device, dtype=x.dtype)) @ interpolation)
                            ).sum(-2)
                        )
                        if likelihood_type is gpytorch.likelihoods.GaussianLikelihood:
                            log_prob = -0.5 * (
                                ((y - mean).square() + variance) / likelihood.noise
                                + likelihood.noise.log()
                                + torch.log(x.new_tensor(2 * torch.pi))
                            )
                            expected_likelihood = log_prob.mean(-1)
                        else:
                            nodes = likelihood.quadrature.locations.reshape(-1, *((1,) * mean.ndim))
                            weights = likelihood.quadrature.weights.reshape_as(nodes)
                            samples = mean + (2 * variance).sqrt() * nodes
                            expected_likelihood = (likelihood.forward(samples).log_prob(y) * weights).sum(0).mean(
                                -1
                            ) / torch.pi**0.5
                        kl = 0.5 * (
                            mu.square().sum(-1)
                            + covariance.diagonal(dim1=-1, dim2=-2).sum(-1)
                            - 3
                            - 2 * root.diagonal(dim1=-1, dim2=-2).log().sum(-1)
                        )
                        expected = expected_likelihood - 0.7 / 11 * kl
                        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
                        params = (x, y, *model.parameters(), *likelihood.parameters())
                        actual_grad = torch.autograd.grad(actual.sum(), params, create_graph=True, retain_graph=True)
                        expected_grad = torch.autograd.grad(
                            expected.sum(), params, create_graph=True, retain_graph=True
                        )
                        for a, b in zip(actual_grad, expected_grad):
                            torch.testing.assert_close(a, b, rtol=3e-7, atol=3e-8)
                        actual_hessian = torch.autograd.grad(sum(g.square().sum() for g in actual_grad), params)
                        expected_hessian = torch.autograd.grad(sum(g.square().sum() for g in expected_grad), params)
                        for a, b in zip(actual_hessian, expected_hessian):
                            torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)
