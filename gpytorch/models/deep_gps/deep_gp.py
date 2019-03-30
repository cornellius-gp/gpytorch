from __future__ import absolute_import, division, print_function

import torch
import math
from gpytorch.models import AbstractVariationalGP
from gpytorch.mlls import AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import settings
from torch.nn.functional import softplus


class NegativeKLDivergence(AddedLossTerm):
    def __init__(self, variational_strategy):
        self.variational_strategy = variational_strategy

    def loss(self):
        return -1 * self.variational_strategy.kl_divergence().sum()


class AbstractDeepGPHiddenLayer(AbstractVariationalGP):
    def __init__(self, variational_strategy, input_dims, output_dims, num_samples):
        """
        Represents a layer in a deep GP where inference is performed via the doubly stochastic method of
        Salimbeni et al., 2017. Upon calling, instead of returning a variational distribution q(f), returns samples
        from the variational distribution.

        See the documentation for __call__ below for more details below. Note that the behavior of __call__
        will change to be much more elegant with multiple batch dimensions; however, the interface doesn't really
        change.

        Args:
            - variational_strategy (VariationalStrategy): Strategy for changing q(u) -> q(f) (see other VI docs)
            - input_dims (int): Dimensionality of input data expected by each GP
            - output_dims (int): Number of GPs in this layer, equivalent to output dimensionality.
            - last_layer (bool): True if this is to be the last layer in the deep GP, false otherwise.
            - num_samples (int): Number of samples to draw from q(f) for returning
        """
        super(AbstractDeepGPHiddenLayer, self).__init__(variational_strategy)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_samples = num_samples
        self.register_added_loss_term("hidden_kl_divergence")

    def forward(self, x):
        raise NotImplementedError

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
        inputs = inputs.contiguous()
        # Forward samples through the VariationalStrategy and return *samples* from q(f)
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

        variational_dist_f = super(AbstractDeepGPHiddenLayer, self).__call__(inputs)
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


class AbstractDeepGP(AbstractDeepGPHiddenLayer):
    def __init__(self, variational_strategy, input_dims, output_dims, num_samples, hidden_gp_net):
        super(AbstractDeepGP, self).__init__(variational_strategy, input_dims, output_dims, num_samples)

        self.hidden_gp_net = hidden_gp_net

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, inputs):
        # Forward samples through the VariationalStrategy and return q(f)
        inputs = self.hidden_gp_net(inputs)
        last_layer_inputs = inputs.contiguous().view(-1, self.input_dims)
        last_layer_inputs = last_layer_inputs.unsqueeze(0)
        last_layer_inputs = last_layer_inputs.expand(
            self.output_dims,
            self.num_samples * inputs.size(-2),
            self.input_dims
        )

        return AbstractVariationalGP.__call__(self, last_layer_inputs)


class DeepGaussianLikelihood(GaussianLikelihood):
    def __init__(
        self,
        num_samples,
        noise_prior=None,
        batch_size=1,
        param_transform=softplus,
        inv_param_transform=None,
        **kwargs
    ):
        # TODO: Rewrite to be a general DeepLikelihood wrapper once the remainder of GPyTorch supports multiple
        # batch dimensions.
        super(DeepGaussianLikelihood, self).__init__(
            noise_prior=None,
            batch_size=1,
            param_transform=softplus,
            inv_param_transform=None,
            **kwargs)
        self.num_samples = num_samples

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise_covar.noise
        num_outputs = mean.size(0)
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
