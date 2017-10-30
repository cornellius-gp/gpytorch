import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

test_x = Variable(torch.linspace(0, 1, 51))
test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))


class ExactGPModel(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
        super(ExactGPModel, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


def test_gp_prior_and_likelihood():
    gp_model = ExactGPModel()
    gp_model.covar_module.initialize(log_lengthscale=0)  # This shouldn't really do anything now
    gp_model.mean_module.initialize(constant=1)  # Let's have a mean of 1
    gp_model.likelihood.initialize(log_noise=math.log(0.5))
    gp_model.eval()

    # Let's see how our model does, not conditioned on any data
    # The GP prior should predict mean of 1, with a variance of 1
    function_predictions = gp_model(train_x)
    assert(torch.norm(function_predictions.mean().data - 1) < 1e-5)
    assert(torch.norm(function_predictions.var().data - 1.5) < 1e-5)

    # The covariance between the furthest apart points should be 1/e
    least_covar = function_predictions.covar().data[0, -1]
    assert(math.fabs(least_covar - math.exp(-1)) < 1e-5)


def test_posterior_latent_gp_and_likelihood_without_optimization():
    fast_pred_var = gpytorch.functions.fast_pred_var
    gpytorch.functions.fast_pred_var = False

    # We're manually going to set the hyperparameters to be ridiculous
    gp_model = ExactGPModel()
    # Update bounds to accomodate extreme parameters
    gp_model.covar_module.set_bounds(log_lengthscale=(-10, 10))
    gp_model.likelihood.set_bounds(log_noise=(-10, 10))
    # Update parameters
    gp_model.covar_module.initialize(log_lengthscale=-10)
    gp_model.mean_module.initialize(constant=0)
    gp_model.likelihood.initialize(log_noise=-10)

    # Compute posterior distribution
    gp_model.condition(train_x, train_y)
    gp_model.eval()

    # Let's see how our model does, conditioned with weird hyperparams
    # The posterior should fit all the data
    function_predictions = gp_model(train_x)

    assert(torch.norm(function_predictions.mean().data - train_y.data) < 1e-3)
    assert(torch.norm(function_predictions.var().data) < 1e-3)

    # It shouldn't fit much else though
    test_function_predictions = gp_model(Variable(torch.Tensor([1.1])))

    assert(torch.norm(test_function_predictions.mean().data - 0) < 1e-4)
    assert(torch.norm(test_function_predictions.var().data - 1) < 1e-4)

    fast_pred_var = gpytorch.functions.fast_pred_var = fast_pred_var


def test_posterior_latent_gp_and_likelihood_with_optimization():
    # We're manually going to set the hyperparameters to something they shouldn't be
    gp_model = ExactGPModel()
    gp_model.covar_module.initialize(log_lengthscale=1)
    gp_model.mean_module.initialize(constant=0)
    gp_model.likelihood.initialize(log_noise=1)

    # Compute posterior distribution
    gp_model.condition(train_x, train_y)

    # Find optimal model hyperparameters
    gp_model.train()
    optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
    optimizer.n_iter = 0
    for i in range(50):
        optimizer.zero_grad()
        output = gp_model(train_x)
        loss = -gp_model.marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Test the model
    gp_model.eval()
    test_function_predictions = gp_model(test_x)
    mean_abs_error = torch.mean(torch.abs(test_y - test_function_predictions.mean()))

    assert(mean_abs_error.data.squeeze()[0] < 0.05)


def test_posterior_latent_gp_and_likelihood_with_optimization_cuda():
    if torch.cuda.is_available():
        # We're manually going to set the hyperparameters to something they shouldn't be
        gp_model = ExactGPModel().cuda()
        gp_model.covar_module.initialize(log_lengthscale=1)
        gp_model.mean_module.initialize(constant=0)
        gp_model.likelihood.initialize(log_noise=1)

        # Compute posterior distribution
        gp_model.condition(train_x.cuda(), train_y.cuda())

        # Find optimal model hyperparameters
        gp_model.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for i in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x.cuda())
            loss = -gp_model.marginal_log_likelihood(output, train_y.cuda())
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        # Test the model
        gp_model.eval()
        test_function_predictions = gp_model(test_x.cuda())
        mean_abs_error = torch.mean(torch.abs(test_y.cuda() - test_function_predictions.mean()))

        assert(mean_abs_error.data.squeeze()[0] < 0.05)
