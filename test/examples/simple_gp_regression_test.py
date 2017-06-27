import math
import torch
import gpytorch

from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.math.modules import Bias, Identity
from gpytorch.latent_distributions import GPDistribution
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.inference import ExactGPInference
from gpytorch import ObservationModel


# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))


# Create a simple GP regression model - using a RBF kernel
prior = GPDistribution(mean_module=Bias(), covar_module=RBFKernel())
likelihood = GaussianLikelihood()

class ExactGPObservationModel(gpytorch.ObservationModel):
    def forward(self,x):
        latent_pred = self.latent_distribution(x)
        observed_pred = self.observation_model(latent_pred)
        return observed_pred



def test_gp_prior_and_likelihood():
    prior.covar_module.initialize(log_lengthscale=0) # This shouldn't really do anything now
    prior.mean_module.initialize(bias=1) # Let's have a mean of 1
    likelihood.initialize(log_noise=math.log(0.5))

    # Let's see how our model does, not conditioned on any data
    # The GP prior should predict mean of 1, with a variance of 1
    function_predictions = prior(train_x)
    assert(all(function_predictions.mean().data == 1))
    assert(all(function_predictions.var().data == 1))

    # The covariance between the furthest apart points should be 1/e
    least_covar = function_predictions.covar().data[0, -1]
    assert(math.fabs(least_covar - math.exp(-1)) < 1e-6)


def test_posterior_latent_gp_and_likelihood_without_optimization():
    # We're manually going to set the hyperparameters to be ridiculous
    prior.covar_module.initialize(log_lengthscale=-10) # This should fit every point exactly
    prior.mean_module.initialize(bias=0) # Let's have a mean of 0
    likelihood.initialize(log_noise=-10)

    prior_observation_model = ExactGPObservationModel(likelihood,prior)

    # Compute posterior distribution
    infer = ExactGPInference(prior_observation_model)
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
    prior.covar_module.initialize(log_lengthscale=1)
    prior.mean_module.initialize(bias=0)
    likelihood.initialize(log_noise=1)

    prior_observation_model = ExactGPObservationModel(likelihood,prior)

    # Compute posterior distribution
    infer = ExactGPInference(prior_observation_model)
    posterior_observation_model = infer.run(train_x, train_y, optimize=True)

    # We should learn optimal hyperparmaters
    # bias should be near 0
    bias_value = posterior_observation_model.latent_distribution.mean_module.bias.data[0]
    assert(math.fabs(bias_value) < .05)

    # log_lengthscale should be near -1.4
    log_lengthscale_value = posterior_observation_model.latent_distribution.covar_module.log_lengthscale.data.squeeze()[0]
    assert(log_lengthscale_value < -1.1)
    assert(log_lengthscale_value > -1.8)

    # log_noise should be very small
    log_noise_value = posterior_observation_model.observation_model.log_noise.data.squeeze()[0]
    assert(log_noise_value < -8)
