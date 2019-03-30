import torch
from torch.nn import Linear

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import AbstractVariationalGP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import AbstractDeepGPHiddenLayer, AbstractDeepGP, DeepGaussianLikelihood

import urllib.request
import os.path
from scipy.io import loadmat
from math import floor
import numpy as np

from fancy_backward import fancy_backward


if not os.path.isfile('elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', 'elevators.mat')

data = torch.Tensor(loadmat('elevators.mat')['data'])
X = data[:, :-1]
y = data[:, -1]

N = data.shape[0]
np.random.seed(0)
data = data[np.random.permutation(np.arange(N)),:]

train_n = int(floor(0.8*len(X)))

train_x = X[:train_n, :].contiguous().cuda()
train_y = y[:train_n].contiguous().cuda()

test_x = X[train_n:, :].contiguous().cuda()
test_y = y[train_n:].contiguous().cuda()

mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

mean,std = train_y.mean(),train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


class ToyDeepGPHiddenLayer(AbstractDeepGPHiddenLayer):
    def __init__(self, input_dims, output_dims, num_inducing=512, num_samples=1):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_size=output_dims
        )

        variational_strategy = WhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy,
                                            input_dims,
                                            output_dims,
                                            num_samples=num_samples)

        self.mean_module = ConstantMean(batch_size=output_dims)
        self.covar_module = ScaleKernel(RBFKernel(batch_size=output_dims,
                                                  ard_num_dims=input_dims), batch_size=output_dims,
                                        ard_num_dims=None)

        self.linear_layer = Linear(input_dims, 1)


    def forward(self, x):
        mean_x = self.linear_layer(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



class ToyDeepGP(AbstractDeepGP):
    def __init__(self, input_dims, output_dims, hidden_gp_net, num_samples, num_inducing=256):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_size=output_dims
        )

        variational_strategy = WhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGP, self).__init__(variational_strategy,
                                                  input_dims,
                                                  output_dims,
                                                  num_samples,
                                                  hidden_gp_net)

        self.mean_module = ConstantMean(batch_size=output_dims)
        self.covar_module = ScaleKernel(RBFKernel(batch_size=output_dims,
                                                  ard_num_dims=input_dims), batch_size=output_dims,
                                        ard_num_dims=None)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)




num_samples = 5
hidden_layer_size = train_x.size(-1)

hidden_gp = ToyDeepGPHiddenLayer(input_dims=train_x.size(-1),
                                 output_dims=hidden_layer_size,
                                 num_samples=num_samples).cuda()
hidden_net = torch.nn.Sequential(hidden_gp)
# Uncomment these lines to use a 3 layer deep GP instead of a 2 layer deep GP!
# hidden_gp2 = ToyDeepGPHiddenLayer(input_dims=hidden_layer_size,
#                                 output_dims=hidden_layer_size,
#                                 num_samples=num_samples).cuda()
# hidden_net = torch.nn.Sequential(hidden_gp, hidden_gp2)
model = ToyDeepGP(hidden_layer_size, 1, hidden_gp_net=hidden_net, num_samples=num_samples).cuda()



likelihood = DeepGaussianLikelihood(num_samples=num_samples).cuda()
mll = VariationalELBO(likelihood, model, train_x.size(-2), combine_terms=False)


num_epochs = 60

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

import time

other_params = model.named_hyperparameters()

for i in range(num_epochs):
    # Within each iteration, we will go over each minibatch of data
    for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
        start_time = time.time()
        optimizer.zero_grad()
        with gpytorch.settings.fast_computations(solves=False, log_prob=False):
            output = model(x_batch)
            # Here we handle the fact that the output is actually num_samples Gaussians by expanding the labels.
            y_batch = y_batch.unsqueeze(0).unsqueeze(0).expand(model.output_dims, model.num_samples, -1)
            log_lik, kl_div, _, added_loss = mll(output, y_batch, num_samples=num_samples)

            log_lik_loss = log_lik * num_samples
            kl_div_loss = - kl_div + added_loss.div(mll.num_data)

            kl_div_loss.backward(retain_graph=True)

            q_f_means = [mod._mean_qf for mod in hidden_net]
            q_f_stds = [mod._std_qf for mod in hidden_net]

            fancy_backward(log_lik_loss, q_f_means[0], q_f_stds[0].pow(2.0), other_params=other_params)
            loss = (log_lik_loss + kl_div_loss).item()

            print('Epoch %d [%d/%d] - Loss: %.3f - - Time: %.3f' % (i + 1, minibatch_i, len(train_loader),
                                                                    loss, time.time() - start_time))

            optimizer.step()
