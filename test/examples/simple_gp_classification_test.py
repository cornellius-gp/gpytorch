import math
import torch
import gpytorch
from torch import nn, optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.random_variables import GaussianRandomVariable


def train_data(cuda=False):
    train_x = Variable(torch.linspace(0, 1, 10))
    train_y = Variable(torch.sign(torch.cos(train_x.data * (4 * math.pi))))
    if cuda:
        return train_x.cuda(), train_y.cuda()
    else:
        return train_x, train_y


class GPClassificationModel(gpytorch.models.VariationalGP):
    def __init__(self, train_x):
        super(GPClassificationModel, self).__init__(train_x)
        self.mean_module = ConstantMean(constant_bounds=[-1e-5, 1e-5])
        self.covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5, 6))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp().expand_as(covar_x))
        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred


def test_kissgp_classification_error():
    train_x, train_y = train_data()
    likelihood = BernoulliLikelihood()
    model = GPClassificationModel(train_x.data)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    optimizer.n_iter = 0
    for i in range(50):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -model.marginal_log_likelihood(likelihood, output, train_y)
        loss.backward()
        optimizer.n_iter += 1
        optimizer.step()

    # Set back to eval mode
    model.eval()
    likelihood.eval()
    test_preds = likelihood(model(train_x)).mean().ge(0.5).float().mul(2).sub(1).squeeze()
    mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
    assert(mean_abs_error.data.squeeze()[0] < 1e-5)


def test_kissgp_classification_error_cuda():
    if torch.cuda.is_available():
        train_x, train_y = train_data(cuda=True)
        likelihood = BernoulliLikelihood().cuda()
        model = GPClassificationModel(train_x.data).cuda()

        # Find optimal model hyperparameters
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for i in range(50):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -model.marginal_log_likelihood(likelihood, output, train_y)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        # Set back to eval mode
        model.eval()
        test_preds = likelihood(model(train_x)).mean().ge(0.5).float().mul(2).sub(1).squeeze()
        mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
        assert(mean_abs_error.data.squeeze()[0] < 1e-5)
