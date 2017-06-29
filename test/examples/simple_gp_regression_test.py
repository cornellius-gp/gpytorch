import math
import torch
import gpytorch

from torch.autograd import Variable
from torch.nn import Parameter
from gpytorch.parameters import MLEParameterGroup
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import Inference
from gpytorch import ObservationModel
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

class ExactGPObservationModel(gpytorch.ObservationModel):
    def __init__(self):
        super(ExactGPObservationModel,self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.params = MLEParameterGroup(
            constant_mean=Parameter(torch.Tensor([0])),
            log_noise=Parameter(torch.Tensor([0])),
            log_lengthscale=Parameter(torch.Tensor([0])),
            )


    def forward(self,x):
        mean_x = self.mean_module(x, constant=self.params.constant_mean)
        covar_x = self.covar_module(x, log_lengthscale=self.params.log_lengthscale)

        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred, self.params.log_noise


def test_gp_prior_and_likelihood():
    prior_observation_model = ExactGPObservationModel()
    prior_observation_model.params.log_lengthscale.data.fill_(0) # This shouldn't really do anything now
    prior_observation_model.params.constant_mean.data.fill_(1) # Let's have a mean of 1
    prior_observation_model.params.log_noise.data.fill_(math.log(0.5))

    # Let's see how our model does, not conditioned on any data
    # The GP prior should predict mean of 1, with a variance of 1
    function_predictions = prior_observation_model(train_x)
    assert(torch.norm(function_predictions.mean().data  - 1) < 1e-5)
    assert(torch.norm(function_predictions.var().data - 1.5) < 1e-5)

    # The covariance between the furthest apart points should be 1/e
    least_covar = function_predictions.covar().data[0, -1]
    assert(math.fabs(least_covar - math.exp(-1)) < 1e-6)


def test_posterior_latent_gp_and_likelihood_without_optimization():
    # We're manually going to set the hyperparameters to be ridiculous
    prior_observation_model = ExactGPObservationModel()
    prior_observation_model.params.log_lengthscale.data.fill_(-10) # This shouldn't really do anything now
    prior_observation_model.params.constant_mean.data.fill_(0)
    prior_observation_model.params.log_noise.data.fill_(-10)

    # Compute posterior distribution
    infer = Inference(prior_observation_model)
    posterior_observation_model = infer.run(train_x, train_y, optimize=False)

    # Let's see how our model does, conditioned with weird hyperparams
    # The posterior should fit all the data
    function_predictions = posterior_observation_model(train_x)
    assert(torch.norm(function_predictions.mean().data - train_y.data) < 1e-3)
    assert(torch.norm(function_predictions.var().data) < 1e-3)

    # It shouldn't fit much else though
    test_function_predictions = posterior_observation_model(Variable(torch.Tensor([1.1])))

    assert(torch.norm(test_function_predictions.mean().data - 0) < 1e-4)
    assert(torch.norm(test_function_predictions.var().data - 1) < 1e-4)


def test_posterior_latent_gp_and_likelihood_with_optimization():

    # We're manually going to set the hyperparameters to something they shouldn't be
    prior_observation_model = ExactGPObservationModel()
    prior_observation_model.params.log_lengthscale.data.fill_(1)
    prior_observation_model.params.constant_mean.data.fill_(0)
    prior_observation_model.params.log_noise.data.fill_(1)

    # Compute posterior distribution
    infer = Inference(prior_observation_model)
    posterior_observation_model = infer.run(train_x, train_y, optimize=True)

    # We should learn optimal hyperparmaters
    # bias should be near 0
    constant_value = posterior_observation_model.params.constant_mean.data.squeeze()[0]
    assert(math.fabs(constant_value) < .05)

    # log_lengthscale should be near -1.4
    log_lengthscale_value = posterior_observation_model.params.log_lengthscale.data.squeeze()[0]
    assert(log_lengthscale_value < -1.1)
    assert(log_lengthscale_value > -1.8)

    # log_noise should be very small
    log_noise_value = posterior_observation_model.params.log_noise.data.squeeze()[0]
    assert(log_noise_value < -8)
