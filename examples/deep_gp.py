from __future__ import absolute_import, division, print_function

import torch
import math
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import WhitenedVariationalStrategy, VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import AbstractVariationalGP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from torch.nn.functional import softplus

softplus = torch.functional.F.softplus


class NegativeKLDivergence(AddedLossTerm):
    def __init__(self, variational_strategy):
        self.variational_strategy = variational_strategy

    def loss(self):
        return -1 * self.variational_strategy.kl_divergence().sum()


class HiddenGPLayer(AbstractVariationalGP):
    """
    Represents a hidden layer in a deep GP where inference is performed via the doubly stochastic method of
    Salimbeni et al., 2017. Upon calling, instead of returning a variational distribution q(f), returns samples
    from the variational distribution.

    See the documentation for __call__ below for more details below. Note that the behavior of __call__
    will change to be much more elegant with multiple batch dimensions; however, the interface doesn't really
    change.

    Args:
        - input_dims (int): Dimensionality of input data expected by each GP
        - output_dims (int): Number of GPs in this layer, equivalent to output dimensionality.
        - num_inducing (int): Number of inducing points for this hidden layer
        - num_samples (int): Number of samples to draw from q(f) for returning
    """
    def __init__(self, input_dims, output_dims, num_inducing=512, num_samples=20):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_samples = num_samples
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

        super(HiddenGPLayer, self).__init__(variational_strategy)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(batch_size=output_dims, param_transform=softplus,
                                                  ard_num_dims=input_dims), batch_size=output_dims,
                                        param_transform=softplus, ard_num_dims=None)

        self.register_added_loss_term("hidden_kl_divergence")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, inputs):
        """
        Forward data through this hidden GP layer.

        If the input is 2 dimensional, we pass the input through each hidden GP, resulting in a `h x n` batch
        Gaussian distribution. We then draw `s` samples from these Gaussians and reshape the result to be
        `s x n x h` (e.g., h becomes the dimensionality of the output).

        If the input is 3 dimensional, we assume that the input is `s x n x d`, e.g. the batch dimension
        corresponds to the number of samples. We use this as the number of samples to draw, and just propagate
        each sample through the hidden layer. The output will be `s x n x h`.

        If the input is 4 dimensional, we assume that for some reason the user has already reshaped things to be
        `h x s x n x d`. We reshape this internally to be `h x sn x d`, and the output will be `s x n x h`.

        The goal of these last two points is that if you have a tensor `x` that is `n x d`, then:
            >>> hidden_gp2(hidden_gp(x))

        will just work, and return a tensor of size `s x n x h2`, where `h2` is the output dimensionality of
        hidden_gp2. In this way, hidden GP layers are easily composable.
        """
        # TODO: Simplify this logic once multiple batch dimensions is supported

        if inputs.dim() == 2:
            # Assume new input entirely
            inputs = inputs.unsqueeze(0)
            inputs = inputs.expand(self.output_dims, inputs.size(-2), self.input_dims)
        elif inputs.dim() == 3:
            # Assume batch dim is samples, not output_dim
            inputs = inputs.unsqueeze(0)
            inputs = inputs.expand(self.output_dims, inputs.size(1), inputs.size(-2), self.input_dims)

        if inputs.dim() == 4:
            num_samples = inputs.size(-3)
            inputs = inputs.view(self.output_dims, inputs.size(-2) * inputs.size(-3), self.input_dims)
            reshape_output = True
        else:
            reshape_output = False
            num_samples = self.num_samples

        variational_dist_f = super(HiddenGPLayer, self).__call__(inputs)
        mean_qf = variational_dist_f.mean
        std_qf = variational_dist_f.variance.sqrt()

        if reshape_output:
            samples = torch.distributions.Normal(mean_qf, std_qf).rsample()
            samples = samples.view(self.output_dims, num_samples, -1).permute(1, 2, 0)
        else:
            samples = torch.distributions.Normal(mean_qf, std_qf).rsample(torch.Size([num_samples]))
            samples = samples.transpose(-2, -1)

        loss_term = NegativeKLDivergence(self.variational_strategy)
        self.update_added_loss_term("hidden_kl_divergence", loss_term)

        return samples

class DeepGP(AbstractVariationalGP):
    def __init__(self, input_dims, hidden_dims, output_dims, num_inducing=512, num_samples=20):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.num_samples = num_samples

        inducing_points = torch.randn(output_dims, num_inducing, hidden_dims)

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

        super(DeepGP, self).__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_size=output_dims)
        self.covar_module = ScaleKernel(RBFKernel(batch_size=output_dims, param_transform=softplus,
                                                  ard_num_dims=hidden_dims), batch_size=output_dims,
                                        param_transform=softplus, ard_num_dims=None)

        # For more layers, just make more hidden layers to forward through
        self.hidden_gp_layer = HiddenGPLayer(
            input_dims=input_dims,
            output_dims=hidden_dims,
            num_inducing=num_inducing,
            num_samples=num_samples
        )

        # self.second_hidden_gp_layer = HiddenGPLayer(
        #     input_dims=hidden_dims,
        #     output_dims=hidden_dims,
        #     num_inducing=num_inducing,
        #     num_samples=num_samples
        # )



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, inputs):
        """
        The call method of a Deep GP differs from a standard variational GP simply in that we first
        pass the data through any hidden GP layers, reshape the input to account for the fact that
        the batch dimension should be interpreted as a number of samples, and then pass through the final layer.

        In some sense, the DeepGP object itself represents the last layer of the deep GP.
        """
        # Forward through more layers here if they exist
        hidden_inputs = self.hidden_gp_layer(inputs).contiguous()
        # hidden_inputs = self.second_hidden_gp_layer(hidden_inputs)

        # hidden_inputs is num_samples x n_inputs x hidden_dims
        # Combine samples and inputs dimension since we want to mean over those in the likelihood
        last_layer_inputs = hidden_inputs.contiguous().view(-1, self.hidden_dims)
        last_layer_inputs = last_layer_inputs.unsqueeze(0).expand(self.output_dims, self.num_samples * inputs.size(-2), self.hidden_dims)

        return super(DeepGP, self).__call__(last_layer_inputs)

class DeepGaussianLikelihood(GaussianLikelihood):
    def __init__(self, num_samples, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None, **kwargs):
        super(DeepGaussianLikelihood, self).__init__(noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None, **kwargs)
        self.num_samples = num_samples

    def variational_log_probability(self, input, target, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise_covar.noise
        num_outputs = mean.size(0)
        from IPython.core.debugger import set_trace
        mean = mean.view(num_outputs, self.num_samples, -1)
        variance = variance.view(num_outputs, self.num_samples, -1)

        if mean.dim() > target.dim():
            target = target.unsqueeze(-1)

        if variance.ndimension() == 1:
            if settings.debug.on() and noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultivariateNormal distribution.")
            noise = noise.squeeze(0)

        res = -0.5 * ((target - mean) ** 2 + variance) / noise
        res += -0.5 * noise.log() - 0.5 * math.log(2 * math.pi)
        return res.sum(-1).mean(-1)
