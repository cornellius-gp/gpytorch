import math
import torch
import gpytorch
from torch import nn, optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.random_variables import GaussianRandomVariable

train_x = Variable(torch.linspace(0, 1, 10))
train_y = Variable(torch.sign(torch.cos(train_x.data * (4 * math.pi))))


class LatentFunction(gpytorch.GridInducingPointModule):
    def __init__(self):
        super(LatentFunction, self).__init__(grid_size=30, grid_bounds=[(0, 1)])
        self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5, 6))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred


class GPClassificationModel(gpytorch.GPModel):
    def __init__(self):
        super(GPClassificationModel, self).__init__(BernoulliLikelihood())
        self.latent_function = LatentFunction()

    def forward(self, x):
        return self.latent_function(x)


def test_kissgp_classification_error():
    model = GPClassificationModel()

    # Find optimal model hyperparameters
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.15)
    optimizer.n_iter = 0
    for i in range(200):
        optimizer.zero_grad()
        output = model.forward(train_x)
        loss = -model.marginal_log_likelihood(output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Set back to eval mode
    model.eval()
    model.condition(train_x, train_y)
    test_preds = model(train_x).mean().ge(0.5).float().mul(2).sub(1).squeeze()
    mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
    assert(mean_abs_error.data.squeeze()[0] < 1e-5)
