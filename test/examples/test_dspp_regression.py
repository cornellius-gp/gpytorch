import gpytorch
import torch
import unittest
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.test.base_test_case import BaseTestCase


train_n = 20
train_x = torch.linspace(0, 1, train_n).unsqueeze(-1)
train_y = torch.sin(2 * train_x[..., 0]) + torch.randn(train_n).mul_(0.01)
test_n = 10
test_x = torch.linspace(-0.2, 1.2, test_n).unsqueeze(-1)
test_y = torch.sin(2 * test_x[..., 0])

num_inducing_pts = 10            # Number of inducing points in each hidden layer
num_epochs = 100                 # Number of epochs to train for
initial_lr = 0.1                 # Initial learning rate
hidden_dim = 3                   # Number of GPs (i.e., the width) in the hidden layer.
num_quadrature_sites = 8         # Number of quadrature sites (see paper for a description of this.


class DSPPHiddenLayer(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if output_dims is None:
            # An output_dims of None implies there is only one GP in this layer
            # (e.g., the last layer for univariate regression).
            inducing_points = torch.randn(num_inducing, input_dims)
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        super().__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ThreeLayerDSPP(DSPP):
    def __init__(self, train_x_shape, num_inducing, hidden_dim=3, Q=3):
        hidden_layer1 = DSPPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            num_inducing=num_inducing,
            mean_type='linear',
            Q=Q,
        )
        hidden_layer2 = DSPPHiddenLayer(
            input_dims=hidden_layer1.output_dims,
            output_dims=hidden_dim,
            mean_type='linear',
            num_inducing=num_inducing,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer2.output_dims,
            output_dims=None,
            mean_type='constant',
            num_inducing=num_inducing,
            Q=Q,
        )

        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer2 = hidden_layer2
        self.hidden_layer1 = hidden_layer1

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer1(inputs, **kwargs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1, **kwargs)
        output = self.last_layer(hidden_rep2, expand_for_quadgrid=False, **kwargs)
        return output


class TestSGPRRegression(unittest.TestCase, BaseTestCase):
    seed = 0

    def test_dspp(self):
        model = ThreeLayerDSPP(
            train_x.shape,
            num_inducing=num_inducing_pts,
            hidden_dim=hidden_dim,
            Q=num_quadrature_sites
        )

        model.train()

        adam = torch.optim.Adam([{'params': model.parameters()}], lr=initial_lr, betas=(0.9, 0.999))
        objective = gpytorch.mlls.DeepPredictiveLogLikelihood(model.likelihood, model, num_data=train_n, beta=0.05)

        epochs_iter = range(num_epochs)
        losses = []

        for i in epochs_iter:
            adam.zero_grad()
            output = model(train_x)
            loss = -objective(output, train_y)
            loss.backward()
            adam.step()
            losses.append(loss.item())

        model.eval()
        preds = model.likelihood(model(test_x))
        self.assertLess((preds.mean - test_y).abs().mean().item(), 0.5)
        self.assertLess(preds.variance.mean().item(), 0.5)
