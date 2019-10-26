#!/usr/bin/env python3

from math import pi
import os
import random
import torch
import unittest
import gpytorch
from gpytorch.test.base_test_case import BaseTestCase


try:
    import pyro

    class ClusterGaussianLikelihood(gpytorch.likelihoods.Likelihood):
        def __init__(self, num_tasks, num_clusters, name_prefix=""):
            super().__init__()
            self.register_buffer("prior_cluster_logits", torch.zeros(num_tasks, num_clusters))
            self.register_parameter("variational_cluster_logits", torch.nn.Parameter(torch.randn(num_tasks, num_clusters)))
            self.register_parameter("raw_noise", torch.nn.Parameter(torch.tensor(0.)))
            self.num_tasks = num_tasks
            self.num_clusters = num_clusters
            self.name_prefix = name_prefix
            self.max_plate_nesting = 1

        @property
        def noise(self):
            return torch.nn.functional.softplus(self.raw_noise)

        def _cluster_dist(self, logits):
            dist = pyro.distributions.OneHotCategorical(logits=logits).to_event(1)
            return dist

        def guide(self, *args, **kwargs):
            pyro.sample(
                self.name_prefix + ".cluster_logits",
                self._cluster_dist(self.variational_cluster_logits)
            )

        def forward(self, function_samples, *args, **kwargs):
            cluster_assignment_samples = pyro.sample(
                self.name_prefix + ".cluster_logits",
                self._cluster_dist(self.prior_cluster_logits)
            )
            res = pyro.distributions.Normal(
                loc=(function_samples.unsqueeze(-2) * cluster_assignment_samples).sum(-1),
                scale=self.noise.sqrt(),
            ).to_event(1)
            return res


    class MultitaskVariationalStrategy(gpytorch.variational.VariationalStrategy):
        def forward(self, inputs):
            function_dist = super().forward(inputs)
            function_dist = gpytorch.distributions.MultitaskMultivariateNormal(
                mean=function_dist.mean.transpose(-1, -2),
                covariance_matrix=gpytorch.lazy.BlockDiagLazyTensor(function_dist.lazy_covariance_matrix),
                interleaved=True,
            )
            return function_dist


    class ClusterMultitaskGPModel(gpytorch.models.pyro.PyroGP):
        def __init__(self, train_x, train_y, num_functions=2):
            num_data = train_y.size(-2)
            
            # Define all the variational stuff
            inducing_points = torch.linspace(0, 1, 32).unsqueeze(-1).repeat(num_functions, 1, 1)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.size(-2),
                batch_shape=torch.Size([num_functions])
            )
            variational_strategy = MultitaskVariationalStrategy(self, inducing_points, variational_distribution)

            # Standard initializtation
            likelihood = ClusterGaussianLikelihood(train_y.size(-1), num_functions, name_prefix="likelihood")
            super().__init__(variational_strategy, likelihood, num_data=num_data, name_prefix="cluster_model")
            self.likelihood = likelihood

            # Mean, covar
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_functions])),
                batch_shape=torch.Size([num_functions]),
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            res = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            return res


    class TestPyroIntegration(BaseTestCase, unittest.TestCase):
        seed = 1

        def test_multitask_gp_mean_abs_error(self):
            # Simple training data: let's try to learn sine and cosine functions
            train_x = torch.linspace(0, 1, 100)

            # y1 and y4 functions are sin(2*pi*x) with noise N(0, 0.05)
            train_y1 = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.03
            train_y4 = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.03
            # y2 and y3 functions are -sin(2*pi*x) with noise N(0, 0.05)
            train_y2 = -torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.03
            train_y3 = -torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.03
            # Create a train_y which interleaves the four
            train_y = torch.stack([train_y1, train_y2, train_y3, train_y4], -1)

            model = ClusterMultitaskGPModel(train_x, train_y)
            # Find optimal model hyperparameters
            model.train()

            # Use the adam optimizer
            optimizer = pyro.optim.Adam({"lr": 0.03})
            elbo = pyro.infer.Trace_ELBO(num_particles=16, vectorize_particles=True, retain_graph=True)
            svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

            n_iter = 300
            for _ in range(n_iter):
                loss = svi.step(train_x, train_y)

            # Test the model
            with gpytorch.settings.num_likelihood_samples(128):
                model.eval()
                test_x = torch.linspace(0, 1, 51)
                test_y1 = torch.sin(test_x * (2 * pi))
                test_y2 = -torch.sin(test_x * (2 * pi))
                test_y3 = -torch.sin(test_x * (2 * pi))
                test_y4 = torch.sin(test_x * (2 * pi))
                test_preds = model.likelihood(model(test_x)).mean
                mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds.mean(0)[:, 0]))
                mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds.mean(0)[:, 1]))
                mean_abs_error_task_3 = torch.mean(torch.abs(test_y3 - test_preds.mean(0)[:, 2]))
                mean_abs_error_task_4 = torch.mean(torch.abs(test_y4 - test_preds.mean(0)[:, 3]))

            self.assertLess(mean_abs_error_task_1.squeeze().item(), 0.1)
            self.assertLess(mean_abs_error_task_2.squeeze().item(), 0.1)
            self.assertLess(mean_abs_error_task_3.squeeze().item(), 0.1)
            self.assertLess(mean_abs_error_task_4.squeeze().item(), 0.1)

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
