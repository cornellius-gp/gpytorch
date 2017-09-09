import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function, but with KISS-GP let's use 100 training examples.
train_x = Variable(torch.linspace(0, 1, 1000))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

test_x = Variable(torch.linspace(0, 1, 51))
test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class KissGPModel(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
        super(KissGPModel, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))
        self.grid_covar_module = GridInterpolationKernel(covar_module)
        self.initialize_interpolation_grid(30, grid_bounds=[(0, 1)])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.grid_covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


def test_kissgp_gp_mean_abs_error():
    gp_model = KissGPModel()

    # Optimize the model
    gp_model.train()
    optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
    optimizer.n_iter = 0
    for i in range(25):
        optimizer.zero_grad()
        output = gp_model(train_x)
        loss = -gp_model.marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    gp_model.condition(train_x, train_y)
    gp_model.eval()
    test_preds = gp_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

    assert(mean_abs_error.data.squeeze()[0] < 0.01)
