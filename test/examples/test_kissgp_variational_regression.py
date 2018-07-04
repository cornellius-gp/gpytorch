from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import exp, pi

import os
import torch
import unittest
import gpytorch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.random_variables import GaussianRandomVariable


# Simple training data: let's try to learn a sine function,
# but with KISS-GP let's use 100 training examples.
def make_data():
    train_x = torch.linspace(0, 1, 1000)
    train_y = torch.sin(train_x * (2 * pi)) + 0.01 * torch.randn(1000)
    test_x = torch.linspace(0, 1, 51)
    test_y = torch.sin(test_x * (2 * pi))
    return train_x, train_y, test_x, test_y


class GPRegressionModel(gpytorch.models.GridInducingVariationalGP):
    def __init__(self):
        super(GPRegressionModel, self).__init__(grid_size=20, grid_bounds=[(-0.05, 1.05)])
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-5, 5))
        self.covar_module = RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1, log_transform=True)
        )
        self.register_parameter(
            name="log_outputscale",
            parameter=torch.nn.Parameter(torch.Tensor([0])),
            prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1, log_transform=True),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) * self.log_outputscale.exp()
        return GaussianRandomVariable(mean_x, covar_x)


class TestKissGPVariationalRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(2)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_kissgp_gp_mean_abs_error(self):
        train_x, train_y, test_x, test_y = make_data()
        train_dataset = TensorDataset(train_x, train_y)
        loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

        gp_model = GPRegressionModel()
        likelihood = GaussianLikelihood()
        mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, gp_model, n_data=len(train_y))

        # Optimize the model
        gp_model.train()
        likelihood.train()

        with gpytorch.beta_features.diagonal_correction():
            optimizer = optim.SGD(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
            for _ in range(20):
                scheduler.step()
                batchnum = 0
                for x_batch, y_batch in loader:
                    batchnum += 1
                    x_batch = Variable(x_batch.float())
                    y_batch = Variable(y_batch.float())

                    optimizer.zero_grad()
                    output = gp_model(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    for name, param in gp_model.named_parameters():
                        print('minibatch {} grad norm'.format(batchnum), name, param.grad.norm())
                    optimizer.step()

            for name, param in gp_model.named_parameters():
                print(name, param.grad.norm())
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()

            # Test the model
            gp_model.eval()
            likelihood.eval()

            test_preds = likelihood(gp_model(Variable(test_x))).mean()
            mean_abs_error = torch.mean(torch.abs(Variable(test_y) - test_preds))

        self.assertLess(mean_abs_error.data.squeeze().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
