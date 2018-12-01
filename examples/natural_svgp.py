import math
import torch
import gpytorch
import sys

torch.manual_seed(0)

use_nat = (sys.argv[1]=='T')

N_train = 260
train_x = torch.linspace(0, 1, N_train)
train_y = torch.cos(train_x * (2 * math.pi)) + 0.2 * torch.randn(N_train)


from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import NaturalVariationalDistribution
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class SVGPRegressionModel(AbstractVariationalGP):
    def __init__(self, train_x, use_nat):
        if use_nat:
            variational_distribution = NaturalVariationalDistribution(train_x.size(0))
        else:
            variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self,
                                                   train_x,
                                                   variational_distribution,
                                                   learn_inducing_locations=False)
        super(SVGPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



# Initialize model and likelihood
model = SVGPRegressionModel(train_x, use_nat=use_nat)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

from gpytorch.utils.natural_sgd import NaturalSGD
from torch.optim import SGD


variational_params = list(model.variational_strategy.variational_distribution.parameters())
# Get parameters of everything except the VariationalDistribution
other_params = list(model.parameters())
_ = [other_params.remove(p) for p in variational_params]
_ = [other_params.append(p) for p in likelihood.parameters()]

from copy import copy
all_params = copy(variational_params)
for p in other_params:
    all_params.append(p)

if use_nat:
    print("USING NATURAL SGD + ADAM")
    nat_optimizer = NaturalSGD(variational_params, [], model.variational_strategy.variational_distribution, lr=0.5)
    optimizer = torch.optim.Adam(other_params, lr=0.05)
else:
    print("USING ADAM")
    optimizer = torch.optim.Adam(all_params, lr=0.05)

from gpytorch.mlls.variational_elbo import VariationalELBO

# Find optimal model hyperparameters
model.train()
likelihood.train()

mll = VariationalELBO(likelihood, model, train_y.numel())

training_iter = 250
for i in range(training_iter):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    if use_nat:
        nat_optimizer.step(N_train * loss)
        optimizer.step()
    else:
        loss.backward()
        optimizer.step()
    if i % 25==0 or i == training_iter - 1:
        print('Iter %d/%d - Loss: %.4f    noise: %.3f' % (i + 1, training_iter, loss.item(),
              likelihood.noise.item()))
