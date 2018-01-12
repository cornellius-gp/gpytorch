import math
import torch
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable


gpytorch.functions.fast_pred_var = True


# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
train_y = Variable(torch.sin(train_x.data * (2 * math.pi)))

test_x = Variable(torch.linspace(0, 1, 51))
test_y = Variable(torch.sin(test_x.data * (2 * math.pi)))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


def test_posterior_latent_gp_and_likelihood_without_optimization():
    fast_pred_var = gpytorch.functions.fast_pred_var
    gpytorch.functions.fast_pred_var = False

    # We're manually going to set the hyperparameters to be ridiculous
    likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
    gp_model = ExactGPModel(train_x.data, train_y.data, likelihood)
    # Update bounds to accomodate extreme parameters
    gp_model.covar_module.set_bounds(log_lengthscale=(-10, 10))
    likelihood.set_bounds(log_noise=(-10, 10))
    # Update parameters
    gp_model.covar_module.initialize(log_lengthscale=-10)
    gp_model.mean_module.initialize(constant=0)
    likelihood.initialize(log_noise=-10)

    # Compute posterior distribution
    gp_model.eval()
    likelihood.eval()

    # Let's see how our model does, conditioned with weird hyperparams
    # The posterior should fit all the data
    function_predictions = likelihood(gp_model(train_x))

    assert(torch.norm(function_predictions.mean().data - train_y.data) < 1e-3)
    assert(torch.norm(function_predictions.var().data) < 1e-3)

    # It shouldn't fit much else though
    test_function_predictions = gp_model(Variable(torch.Tensor([1.1])))

    assert(torch.norm(test_function_predictions.mean().data - 0) < 1e-4)
    assert(torch.norm(test_function_predictions.var().data - 1) < 1e-4)

    fast_pred_var = gpytorch.functions.fast_pred_var = fast_pred_var


def test_posterior_latent_gp_and_likelihood_with_optimization():
    # We're manually going to set the hyperparameters to something they shouldn't be
    likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
    gp_model = ExactGPModel(train_x.data, train_y.data, likelihood)
    gp_model.covar_module.initialize(log_lengthscale=1)
    gp_model.mean_module.initialize(constant=0)
    likelihood.initialize(log_noise=1)

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()
    optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
    optimizer.n_iter = 0
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
    test_function_predictions = likelihood(gp_model(test_x))
    mean_abs_error = torch.mean(torch.abs(test_y - test_function_predictions.mean()))

    assert(mean_abs_error.data.squeeze()[0] < 0.05)


def test_posterior_latent_gp_and_likelihood_with_optimization_cuda():
    if torch.cuda.is_available():
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3)).cuda()
        gp_model = ExactGPModel(train_x.data.cuda(), train_y.data.cuda(), likelihood).cuda()
        gp_model.covar_module.initialize(log_lengthscale=1)
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(log_noise=1)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for i in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x.cuda())
            loss = -gp_model.marginal_log_likelihood(likelihood, output, train_y.cuda())
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        # Test the model
        gp_model.eval()
        likelihood.eval()
        test_function_predictions = likelihood(gp_model(test_x.cuda()))
        mean_abs_error = torch.mean(torch.abs(test_y.cuda() - test_function_predictions.mean()))

        assert(mean_abs_error.data.squeeze()[0] < 0.05)
