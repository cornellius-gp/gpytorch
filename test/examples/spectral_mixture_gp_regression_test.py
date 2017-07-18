import math
import torch
import gpytorch

from torch.autograd import Variable
from gpytorch.parameters import MLEParameterGroup, BoundedParameter
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


class SpectralMixtureGPModel(gpytorch.ObservationModel):
    def __init__(self):
        super(SpectralMixtureGPModel, self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = SpectralMixtureKernel()
        self.params = MLEParameterGroup(
            log_noise=BoundedParameter(torch.Tensor([-2]), -15, 15),
            log_mixture_weights=BoundedParameter(torch.zeros(3), -15, 15),
            log_mixture_means=BoundedParameter(torch.zeros(3), -15, 15),
            log_mixture_scales=BoundedParameter(torch.zeros(3), -15, 15)
        )

    def forward(self, x):
        mean_x = self.mean_module(x, constant=Variable(torch.Tensor([0])))
        covar_x = self.covar_module(x,
                                    log_mixture_weights=self.params.log_mixture_weights,
                                    log_mixture_means=self.params.log_mixture_means,
                                    log_mixture_scales=self.params.log_mixture_scales)

        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred, self.params.log_noise


prior_observation_model = SpectralMixtureGPModel()


def test_spectral_mixture_gp_mean_abs_error():
    prior_observation_model = SpectralMixtureGPModel()

    # Compute posterior distribution
    infer = Inference(prior_observation_model)
    posterior_observation_model = infer.run(train_x, train_y)

    test_preds = posterior_observation_model(test_x).mean()
    mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

    # The spectral mixture kernel should be trivially able to extrapolate the sine function.
    assert(mean_abs_error.data.squeeze()[0] < 0.05)
