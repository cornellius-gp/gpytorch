from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
import gpytorch
from torch import nn, optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.random_variables import GaussianRandomVariable

n = 4
train_x = torch.zeros(pow(n, 2), 2)
train_y = torch.zeros(pow(n, 2))
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
        train_y[i * n + j] = pow(-1, int(i / 2) + int(j / 2))
train_x = Variable(train_x)
train_y = Variable(train_y)


class GPClassificationModel(gpytorch.models.GridInducingVariationalGP):

    def __init__(self):
        super(GPClassificationModel, self).__init__(
            grid_size=8,
            grid_bounds=[(0, 3), (0, 3)],
        )
        self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-10, 12))
        self.register_parameter(
            'log_outputscale',
            nn.Parameter(torch.Tensor([0])),
            bounds=(-10, 12),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred


class TestKissGPKroneckerProductClassification(unittest.TestCase):

    def test_kissgp_classification_error(self):
        model = GPClassificationModel()
        likelihood = BernoulliLikelihood()
        mll = gpytorch.mlls.VariationalMarginalLogLikelihood(
            likelihood,
            model,
            n_data=len(train_y),
        )

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        optimizer = optim.Adam(model.parameters(), lr=0.15)
        optimizer.n_iter = 0
        for _ in range(20):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        # Set back to eval mode
        model.eval()
        likelihood.eval()

        test_preds = model(train_x).mean().ge(0.5).float().mul(2).sub(1).squeeze()
        mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
        self.assertLess(mean_abs_error.data.squeeze().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
