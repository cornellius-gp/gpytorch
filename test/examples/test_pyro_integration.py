#!/usr/bin/env python3

from math import pi
import torch
import time
import unittest
import gpytorch
from gpytorch.test.base_test_case import BaseTestCase


try:
    import pyro

    class ClusterGaussianLikelihood(gpytorch.likelihoods.Likelihood):
        def __init__(self, num_tasks, num_clusters, name_prefix=str(time.time()), reparam=False):
            super().__init__()
            self.register_buffer("prior_cluster_logits", torch.zeros(num_tasks, num_clusters))
            self.register_parameter(
                "variational_cluster_logits", torch.nn.Parameter(torch.randn(num_tasks, num_clusters))
            )
            self.register_parameter("raw_noise", torch.nn.Parameter(torch.tensor(0.0)))
            self.reparam = reparam
            if self.reparam:
                self.register_buffer("temp", torch.tensor(1.0))
            self.num_tasks = num_tasks
            self.num_clusters = num_clusters
            self.name_prefix = name_prefix
            self.max_plate_nesting = 1

        @property
        def noise(self):
            return torch.nn.functional.softplus(self.raw_noise)

        def _cluster_dist(self, logits):
            if self.reparam:
                dist = pyro.distributions.RelaxedOneHotCategorical(temperature=self.temp, logits=logits).to_event(1)
            else:
                dist = pyro.distributions.OneHotCategorical(logits=logits).to_event(1)
            return dist

        def pyro_guide(self, function_dist, target):
            pyro.sample(self.name_prefix + ".cluster_logits", self._cluster_dist(self.variational_cluster_logits))
            return super().pyro_guide(function_dist, target)

        def pyro_model(self, function_dist, target):
            cluster_assignment_samples = pyro.sample(
                self.name_prefix + ".cluster_logits", self._cluster_dist(self.prior_cluster_logits)
            )
            return super().pyro_model(function_dist, target, cluster_assignment_samples=cluster_assignment_samples)

        def forward(self, function_samples, cluster_assignment_samples=None):
            if cluster_assignment_samples is None:
                cluster_assignment_samples = pyro.sample(
                    self.name_prefix + ".cluster_logits", self._cluster_dist(self.variational_cluster_logits)
                )
            res = pyro.distributions.Normal(
                loc=(function_samples.unsqueeze(-2) * cluster_assignment_samples).sum(-1), scale=self.noise.sqrt()
            ).to_event(1)
            return res

    class ClusterMultitaskGPModel(gpytorch.models.pyro.PyroGP):
        def __init__(self, train_x, train_y, num_functions=2, reparam=False):
            num_data = train_y.size(-2)

            # Define all the variational stuff
            inducing_points = torch.linspace(0, 1, 64).unsqueeze(-1)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.size(-2), batch_shape=torch.Size([num_functions])
            )
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution),
                num_tasks=num_functions,
            )

            # Standard initializtation
            likelihood = ClusterGaussianLikelihood(
                train_y.size(-1), num_functions, name_prefix="likelihood", reparam=reparam
            )
            super().__init__(variational_strategy, likelihood, num_data=num_data, name_prefix=str(time.time()))
            self.likelihood = likelihood
            self.num_functions = num_functions

            # Mean, covar
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            res = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            return res

    class SVGPModel(gpytorch.models.pyro.PyroGP):
        def __init__(self, train_x, train_y):
            num_data = train_y.size(-1)

            # Define all the variational stuff
            inducing_points = torch.linspace(0, 1, 64).unsqueeze(-1)
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(inducing_points.size(-2))
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution
            )

            # Standard initializtation
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            super().__init__(variational_strategy, likelihood, num_data=num_data, name_prefix=str(time.time()))
            self.likelihood = likelihood

            # Mean, covar
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            res = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            return res

    class LowLevelInterfaceClusterMultitaskGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, train_x, train_y, num_functions=2):
            # Define all the variational stuff
            inducing_points = torch.linspace(0, 1, 64).unsqueeze(-1)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.size(-2), batch_shape=torch.Size([num_functions])
            )
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution),
                num_tasks=num_functions,
            )
            super().__init__(variational_strategy)

            # Mean, covar
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            # Values to store
            self.name_prefix = "llcmgp"
            self.num_data, self.num_tasks = train_y.shape
            self.num_functions = num_functions

            # Define likelihood stuff
            self.register_parameter(
                "variational_logits", torch.nn.Parameter(torch.randn(self.num_tasks, self.num_functions))
            )
            self.register_parameter("raw_noise", torch.nn.Parameter(torch.tensor(0.0)))

        @property
        def noise(self):
            return torch.nn.functional.softplus(self.raw_noise)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            res = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            return res

        def guide(self, x, y):
            function_dist = self.pyro_guide(x, name_prefix=self.name_prefix)
            pyro.sample(
                self.name_prefix + ".cluster_logits",
                pyro.distributions.OneHotCategorical(logits=self.variational_logits).to_event(1),
            )
            with pyro.plate(self.name_prefix + ".output_values_plate", function_dist.batch_shape[-1], dim=-1):
                pyro.sample(self.name_prefix + ".f", function_dist)

        def model(self, x, y):
            pyro.module(self.name_prefix + ".gp", self)

            # Draw sample from q(f)
            function_dist = self.pyro_model(x, name_prefix=self.name_prefix)

            # Draw samples of cluster assignments
            cluster_assignment_samples = pyro.sample(
                self.name_prefix + ".cluster_logits",
                pyro.distributions.OneHotCategorical(logits=torch.zeros(self.num_tasks, self.num_functions)).to_event(
                    1
                ),
            )

            # Sample from observation distribution
            with pyro.plate(self.name_prefix + ".output_values_plate", function_dist.batch_shape[-1], dim=-1):
                function_samples = pyro.sample(self.name_prefix + ".f", function_dist)
                obs_dist = pyro.distributions.Normal(
                    loc=(function_samples.unsqueeze(-2) * cluster_assignment_samples).sum(-1), scale=self.noise.sqrt()
                ).to_event(1)
                with pyro.poutine.scale(scale=(self.num_data / y.size(-2))):
                    return pyro.sample(self.name_prefix + ".y", obs_dist, obs=y)

    class TestPyroIntegration(BaseTestCase, unittest.TestCase):
        seed = 1

        def test_simple_high_level_interface(self):
            # Simple training data: let's try to learn a sine function
            train_x = torch.linspace(0, 1, 100)
            train_y = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.03

            model = SVGPModel(train_x, train_y)
            # Find optimal model hyperparameters
            model.train()

            # Use the adam optimizer
            optimizer = pyro.optim.Adam({"lr": 0.02})
            elbo = pyro.infer.TraceMeanField_ELBO(num_particles=16, vectorize_particles=True, retain_graph=True)
            svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

            n_iter = 250
            for _ in range(n_iter):
                svi.step(train_x, train_y)

            # Test the model
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(128):
                model.eval()
                test_x = torch.linspace(0, 1, 51)
                test_y = torch.sin(test_x * (2 * pi))
                test_preds = model.likelihood(model(test_x)).mean
                mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
                self.assertLess(mean_abs_error.squeeze().item(), 0.15)

        def test_high_level_interface(self, mean_field=False):
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

            model = ClusterMultitaskGPModel(train_x, train_y, reparam=mean_field)
            # Find optimal model hyperparameters
            model.train()

            # Use the adam optimizer
            optimizer = pyro.optim.Adam({"lr": 0.03})
            if mean_field:
                elbo = pyro.infer.Trace_ELBO(num_particles=16, vectorize_particles=True, retain_graph=True)
            else:
                elbo = pyro.infer.Trace_ELBO(num_particles=16, vectorize_particles=True, retain_graph=True)
            svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

            n_iter = 200
            for _ in range(n_iter):
                svi.step(train_x, train_y)

            # Test the model
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(128):
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

            self.assertLess(mean_abs_error_task_1.squeeze().item(), 0.15)
            self.assertLess(mean_abs_error_task_2.squeeze().item(), 0.15)
            self.assertLess(mean_abs_error_task_3.squeeze().item(), 0.15)
            self.assertLess(mean_abs_error_task_4.squeeze().item(), 0.15)

        def test_high_level_interface_mean_field(self):
            return self.test_high_level_interface(mean_field=True)

        def test_low_level_interface(self):
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
            model = LowLevelInterfaceClusterMultitaskGPModel(train_x, train_y)
            # Find optimal model hyperparameters
            model.train()

            # Use the adam optimizer
            optimizer = pyro.optim.Adam({"lr": 0.03})
            elbo = pyro.infer.Trace_ELBO(num_particles=16, vectorize_particles=True, retain_graph=True)
            svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

            n_iter = 200
            for _ in range(n_iter):
                svi.step(train_x, train_y)

            # Test the model inference
            cluster_1_idx = model.variational_logits[0].max(dim=-1)[1].item()
            cluster_2_idx = 1 - cluster_1_idx
            cluster_probs = torch.softmax(model.variational_logits, dim=-1)
            self.assertGreater(cluster_probs[0, cluster_1_idx].squeeze().item(), 0.9)
            self.assertGreater(cluster_probs[1, cluster_2_idx].squeeze().item(), 0.9)
            self.assertGreater(cluster_probs[2, cluster_2_idx].squeeze().item(), 0.9)
            self.assertGreater(cluster_probs[3, cluster_1_idx].squeeze().item(), 0.9)


except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
