import math
import torch
import gpytorch

from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.math.modules import Bias, Identity
from gpytorch.distributions import GPDistribution
from gpytorch.distributions.likelihoods import GaussianLikelihood


# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))


# Create a simple GP regression model - using a RBF kernel
prior = GPDistribution(mean_module=Bias(), covar_module=RBFKernel())
likelihood = GaussianLikelihood()


def test_gp_prior_and_likelihood():
    prior.covar_module.initialize(log_lengthscale=0) # This shouldn't really do anything now
    prior.mean_module.initialize(bias=1) # Let's have a mean of 1
    likelihood.initialize(noise=0.5)

    # Let's see how our model does, not conditioned on any data
    # The GP prior should predict mean of 1, with a variance of 1
    function_predictions = prior(train_x)
    assert(all(function_predictions.mean().data == 1))
    assert(all(function_predictions.var().data == 1))

    # The covariance between the furthest apart points should be 1/e
    least_covar = function_predictions.covar().data[0, -1]
    assert(math.fabs(least_covar - math.exp(-1)) < 1e-6)
