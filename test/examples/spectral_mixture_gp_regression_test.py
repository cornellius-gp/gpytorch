import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 0.75, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

# Spectral mixture kernel should be able to train on data up to x=0.75, but test on data up to x=2
test_x = Variable(torch.linspace(0, 2, 51))
test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = SpectralMixtureKernel(
            n_mixtures=3,
            log_mixture_weight_bounds=(-5, 5),
            log_mixture_mean_bounds=(-5, 5),
            log_mixture_scale_bounds=(-5, 5),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


def test_spectral_mixture_gp_mean_abs_error():
    likelihood = GaussianLikelihood(log_noise_bounds=(-5, 5))
    gp_model = SpectralMixtureGPModel(train_x.data, train_y.data, likelihood)

    # Optimize the model
    gp_model.train()
    likelihood.train()
    optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
    optimizer.n_iter = 0

    gpytorch.functions.fastest = False
    for i in range(50):
        optimizer.zero_grad()
        output = gp_model(train_x)
        loss = -gp_model.marginal_log_likelihood(likelihood, output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    gp_model.eval()
    likelihood.eval()
    test_preds = likelihood(gp_model(test_x)).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

    # The spectral mixture kernel should be trivially able to extrapolate the sine function.
    assert(mean_abs_error.data.squeeze()[0] < 0.05)
