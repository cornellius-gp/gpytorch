from gpytorch.models.pyro_deep_gp import AbstractPyroHiddenGPLayer, AbstractPyroDeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

import torch


NUM_INDUCING = 32


class ToyHiddenGPLayer(AbstractPyroHiddenGPLayer):
    def __init__(self, input_dims, output_dims, name=""):
        inducing_points = torch.randn(output_dims, NUM_INDUCING, input_dims)

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
    def __init__(self, input_dims, output_dims, total_num_data, hidden_gp_layers, likelihood, name=""):
        inducing_points = torch.randn(output_dims, NUM_INDUCING, input_dims)

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

from scipy.io import loadmat
from math import floor
import numpy as np

data = loadmat(f'examples/elevators.mat')

train_x = torch.Tensor(data['X_tr'])
train_y = torch.Tensor(data['T_tr'])[:, 0]
#print("train_x, train_y: ", train_x.shape, train_y.shape)

mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6
train_x = (train_x - mean) / std

mean,std = train_y.mean(),train_y.std()
train_y = (train_y - mean) / std

likelihood = GaussianLikelihood()

hidden_gp = ToyHiddenGPLayer(train_x.size(-1), 3, name="layer1")
deep_gp = ToyDeepGP(3, 1, train_x.size(-2), [hidden_gp], likelihood, name="output_layer")

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro import optim


optimizer = optim.Adam({"lr": 0.0, "betas": (0.96, 0.999)})

elbo = Trace_ELBO(num_particles=2, vectorize_particles=True, max_plate_nesting=1)
svi = SVI(deep_gp.model, deep_gp.guide, optimizer, elbo)

n_epochs = 1
n_iter = 0

for epoch_i in range(n_epochs):
    epoch_loss = 0
    for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
        if minibatch_i == n_iter:
            break
        loss = svi.step(x_batch, y_batch)
        epoch_loss = epoch_loss + loss / len(train_loader)
        if minibatch_i % 1 == 0:
            print(f'Epoch {epoch_i}, [{minibatch_i} / {len(train_loader)}], loss = {loss:.3f}, last epoch loss = {epoch_loss:.3f}')



from pyro.poutine import trace, replay
from pyro.infer.importance import vectorized_importance_weights


_, model_trace, guide_trace = vectorized_importance_weights(deep_gp.model, deep_gp.guide,
                                                            x_batch, y_batch,
                                                            num_samples=2,
                                                            max_plate_nesting=2,
                                                            normalized=False)
#guide_trace = trace(deep_gp.guide).get_trace(x_batch, y_batch)
#model_trace = trace(replay(deep_gp.model, guide_trace)).get_trace(x_batch, y_batch)

model_trace.compute_log_prob()
guide_trace.compute_log_prob()

for name, site in model_trace.nodes.items():
    if site['type'] == 'sample':
        print("model site", name, 'value', site['value'].shape, 'logprob', site['log_prob'].shape)
print()

for name, site in guide_trace.nodes.items():
    if site['type'] == 'sample':
        print("guide site", name, 'value', site['value'].shape, 'logprob', site['log_prob'].shape)
