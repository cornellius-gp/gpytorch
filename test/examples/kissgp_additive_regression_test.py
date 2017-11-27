import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function, but with KISS-GP let's use 100 training examples.
n = 40
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_x = Variable(train_x)
train_y = Variable((torch.sin(train_x.data[:, 0]) + torch.cos(train_x.data[:, 1])) * (2 * math.pi))

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_x = Variable(test_x)
test_y = Variable((torch.sin(test_x.data[:, 0]) + torch.cos(test_x.data[:, 1])) * (2 * math.pi))


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class LatentFunction(gpytorch.AdditiveGridInducingPointModule):
    def __init__(self):
        super(LatentFunction, self).__init__(grid_size=100, grid_bounds=[(0, 1)], n_components=2)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class GPRegressionModel(gpytorch.GPModel):
    def __init__(self):
        super(GPRegressionModel, self).__init__(GaussianLikelihood())
        self.latent_function = LatentFunction()

    def forward(self, x):
        return self.latent_function(x)


def test_kissgp_gp_mean_abs_error():
    gp_model = GPRegressionModel()

    # Optimize the model
    gp_model.train()
    optimizer = optim.Adam(gp_model.parameters(), lr=0.2)
    optimizer.n_iter = 0
    for i in range(20):
        optimizer.zero_grad()
        output = gp_model(train_x)
        loss = -gp_model.marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    gp_model.eval()
    gp_model.condition(train_x, train_y)
    test_preds = gp_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
    assert(mean_abs_error.data.squeeze()[0] < 0.1)
