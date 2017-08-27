import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import Inference
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function, but with KISS-GP let's use 100 training examples.
n = 40
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_x = Variable(train_x)
train_y = Variable(torch.sin(((train_x.data[:, 0] + train_x.data[:, 1]) * (2 * math.pi))))

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_x = Variable(test_x)
test_y = Variable(torch.sin((test_x.data[:, 0] + test_x.data[:, 1]) * (2 * math.pi)))


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class KissGPModel(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
        super(KissGPModel, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))
        self.grid_covar_module = GridInterpolationKernel(covar_module)
        self.initialize_interpolation_grid(20, grid_bounds=[(0, 1), (0, 1)])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.grid_covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


def test_kissgp_gp_mean_abs_error():
    prior_gp_model = KissGPModel()

    # Compute posterior distribution
    infer = Inference(prior_gp_model)
    posterior_gp_model = infer.run(train_x, train_y)

    # Optimize the model
    posterior_gp_model.train()
    optimizer = optim.Adam(posterior_gp_model.parameters(), lr=0.2)
    optimizer.n_iter = 0
    for i in range(20):
        optimizer.zero_grad()
        output = posterior_gp_model(train_x)
        loss = -posterior_gp_model.marginal_log_likelihood(output, train_y, 10)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    posterior_gp_model.eval()
    test_preds = posterior_gp_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
    assert(mean_abs_error.data.squeeze()[0] < 0.1)
