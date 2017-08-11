import math
import torch
import gpytorch
from torch import nn, optim
from torch.autograd import Variable
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import Inference
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 0.75, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

# Spectral mixture kernel should be able to train on data up to x=0.75, but test on data up to x=2
test_x = Variable(torch.linspace(0, 2, 51))
test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))


class SpectralMixtureGPModel(gpytorch.GPModel):
    def __init__(self):
        super(SpectralMixtureGPModel, self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = SpectralMixtureKernel()
        self.register_parameter('log_noise', nn.Parameter(torch.Tensor([-2])), bounds=(-5, 5))
        self.register_parameter('log_mixture_weights', nn.Parameter(torch.zeros(3)), bounds=(-5, 5))
        self.register_parameter('log_mixture_means', nn.Parameter(torch.zeros(3)), bounds=(-5, 5))
        self.register_parameter('log_mixture_scales', nn.Parameter(torch.zeros(3)), bounds=(-5, 5))

    def forward(self, x):
        mean_x = self.mean_module(x, constant=Variable(torch.Tensor([0])))
        covar_x = self.covar_module(x,
                                    log_mixture_weights=self.log_mixture_weights,
                                    log_mixture_means=self.log_mixture_means,
                                    log_mixture_scales=self.log_mixture_scales)

        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred, self.log_noise


prior_gp_model = SpectralMixtureGPModel()


def test_spectral_mixture_gp_mean_abs_error():
    prior_gp_model = SpectralMixtureGPModel()

    # Compute posterior distribution
    infer = Inference(prior_gp_model)
    posterior_gp_model = infer.run(train_x, train_y)

    # Optimize the model
    posterior_gp_model.train()
    optimizer = optim.Adam(posterior_gp_model.parameters(), lr=0.1)
    optimizer.n_iter = 0
    for i in range(50):
        optimizer.zero_grad()
        output = posterior_gp_model(train_x)
        loss = -posterior_gp_model.marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    posterior_gp_model.eval()
    test_preds = posterior_gp_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

    # The spectral mixture kernel should be trivially able to extrapolate the sine function.
    assert(mean_abs_error.data.squeeze()[0] < 0.05)
