from gpytorch.models.pyro_deep_gp import AbstractPyroHiddenGPLayer, AbstractPyroDeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
import math

import torch
import numpy as np

from scipy.cluster.vq import kmeans2

from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro import optim
import pyro

from torch.utils.data import TensorDataset, DataLoader

###########################################################################
#### THIS FILE REQUIRES
#### pip install git+git://github.com/hughsalimbeni/bayesian_benchmarks.git
###########################################################################

import bayesian_benchmarks
from bayesian_benchmarks.data import get_regression_data



class ToyHiddenGPLayer(AbstractPyroHiddenGPLayer):
    def __init__(self, input_dims, output_dims, name="", inducing_points=50):
        if type(inducing_points) == int:
            inducing_points = torch.randn(output_dims, inducing_points, input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_size=output_dims
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims, name)

        batch_shape = torch.Size([output_dims])

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


#TODO: Double inheritance
class ToyDeepGP(AbstractPyroDeepGP):
    def __init__(self, input_dims, output_dims, total_num_data, hidden_gp_layers, likelihood, name="", inducing_points=50):
        inducing_points = torch.randn(output_dims, inducing_points, input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2),
            batch_size=output_dims
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(
            variational_strategy,
            input_dims,
            output_dims,
            total_num_data,
            hidden_gp_layers,
            likelihood,
            name
        )

        batch_shape = torch.Size([output_dims])

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


dataset='boston'
dataset = get_regression_data(dataset)
N_train = dataset.X_train.shape[0]
N_test = dataset.X_test.shape[0]
D_X = dataset.D + 1
train_x, train_y = torch.tensor(dataset.X_train).float(), torch.tensor(dataset.Y_train[:, 0]).float()
test_x, test_y = torch.tensor(dataset.X_test).float(), torch.tensor(dataset.Y_test[:, 0]).float()
print("N_train = %d   N_test = %d" % (N_train, N_test))

pyro.set_rng_seed(0)
torch.manual_seed(0)

hidden_layer_width = 2
num_inducing = 50
inducing_points = (train_x[torch.randperm(N_train)[0:num_inducing], :])
inducing_points = inducing_points.clone().data.cpu().numpy()
inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(),
			       inducing_points, minit='matrix')[0])
inducing_points = inducing_points.unsqueeze(0).expand((hidden_layer_width,) + inducing_points.shape)

# NOTE
# this is currently unused
likelihood = GaussianLikelihood()

hidden_gp = ToyHiddenGPLayer(train_x.size(-1), hidden_layer_width, name="layer1", inducing_points=inducing_points)
deep_gp = ToyDeepGP(hidden_layer_width, 1, train_x.size(-2), [hidden_gp], likelihood, name="output_layer",
                    inducing_points=num_inducing)

hidden_gp.variational_strategy.variational_distribution.variational_mean.data = \
    0.2 * torch.randn(hidden_gp.variational_strategy.variational_distribution.variational_mean.shape)
deep_gp.variational_strategy.variational_distribution.variational_mean.data = \
    0.2 * torch.randn(deep_gp.variational_strategy.variational_distribution.variational_mean.shape)

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)

optimizer = optim.Adam({"lr": 0.03, "betas": (0.90, 0.999)})

deep_gp.annealing = 0.1
hidden_gp.annealing = 0.1

# different settings for u/f sampling versus f sampling (u marginalized out)
deep_gp.EXACT = hidden_gp.EXACT = False
num_particles = 4 if deep_gp.EXACT else 32
annealing_epoch = 0 if deep_gp.EXACT else 100
n_epochs = 300 if deep_gp.EXACT else 400

elbo = TraceMeanField_ELBO(num_particles=num_particles, vectorize_particles=True, max_plate_nesting=1)
svi = SVI(deep_gp.model, deep_gp.guide, optimizer, elbo)

def ll_rmse(x, y, num_samples=50):
    pred = deep_gp(x, num_samples=num_samples)[:, :, 0]
    log_prob = torch.distributions.Normal(pred, (-0.5 * deep_gp.log_beta).exp()).log_prob(y)
    log_prob = torch.logsumexp(log_prob - math.log(num_samples), dim=0).mean()
    rmse = (pred.mean(0) - y).pow(2.0).mean().sqrt().item()
    return log_prob, rmse

print("Beginning training in EXACT=%s mode with %d particles" % (deep_gp.EXACT, num_particles))

for epoch_i in range(n_epochs):
    epoch_loss = 0
    if epoch_i == annealing_epoch:
        deep_gp.annealing = 1.0
        hidden_gp.annealing = 1.0
        if epoch_i > 0:
            print("Turning off KL annealing...")

    for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
        loss = svi.step(x_batch, y_batch)
        epoch_loss = epoch_loss + loss / len(train_loader)
    if epoch_i % 10 == 0 or epoch_i == (n_epochs - 1):
        train_ll, train_rmse = ll_rmse(train_x, train_y)
        test_ll, test_rmse = ll_rmse(test_x, test_y)
        precision = pyro.param('log_beta').exp().item()
        frmt = "[epoch %03d] loss: %.4f  test_ll: %.3f  train_ll: %.3f  test_rmse: %.3f  train_rmse: %.3f  obs_prec: %.3f"
        print(frmt % (epoch_i, epoch_loss, test_ll, train_ll, test_rmse, train_rmse, precision))
