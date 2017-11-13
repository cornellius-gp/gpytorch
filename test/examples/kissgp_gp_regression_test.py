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
def make_data(cuda=False):
    train_x = Variable(torch.linspace(0, 1, 100))
    train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))
    test_x = Variable(torch.linspace(0, 1, 51))
    test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class LatentFunction(gpytorch.GridInducingPointModule):
    def __init__(self):
        super(LatentFunction, self).__init__(grid_size=50, grid_bounds=[(0, 1)])
        self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred


class GPRegressionModel(gpytorch.GPModel):
    def __init__(self):
        super(GPRegressionModel, self).__init__(GaussianLikelihood())
        self.latent_function = LatentFunction()

    def forward(self, x):
        return self.latent_function(x)


def test_kissgp_gp_mean_abs_error():
    train_x, train_y, test_x, test_y = make_data()
    gp_model = GPRegressionModel()

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
    gp_model.eval()
    gp_model.condition(train_x, train_y)
    test_preds = gp_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

    assert(mean_abs_error.data.squeeze()[0] < 0.05)


def test_kissgp_gp_mean_abs_error_cuda():
    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = make_data(cuda=True)
        gp_model = GPRegressionModel().cuda()

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
        gp_model.eval()
        gp_model.condition(train_x, train_y)
        test_preds = gp_model(test_x).mean()
        mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        assert(mean_abs_error.data.squeeze()[0] < 0.02)
