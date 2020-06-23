import torch

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood

from ..approximate_gp import ApproximateGP
from .deep_gp import DeepGPLayer as AbstractDeepGPLayer


class AbstractPredictiveDeepGPLayer(AbstractDeepGPLayer):
    def __init__(self, variational_strategy, input_dims, output_dims, num_sample_sites=3, quad_grid=None):
        super().__init__(variational_strategy, input_dims, output_dims)

        self.num_sample_sites = num_sample_sites

        # Pass in previous_layer.quad_grid if you want to share quad_grid across layers.
        if quad_grid is not None:
            self.quad_grid = quad_grid
        else:
            self.quad_grid = torch.nn.Parameter(torch.randn(num_sample_sites, input_dims))

    def __call__(self, inputs, are_samples=False, expand_for_quadgrid=True, **kwargs):
        if isinstance(inputs, MultitaskMultivariateNormal):
            # inputs is definitely in the second layer, and mean is n x t
            mus, sigmas = inputs.mean, inputs.variance.sqrt()

            if expand_for_quadgrid:
                xi_mus = mus.unsqueeze(-3)  # 1 x n x t
                xi_sigmas = sigmas.unsqueeze(-3)  # 1 x n x t
            else:
                xi_mus = mus
                xi_sigmas = sigmas

            # unsqueeze sigmas to 1 x n x t, locations from [q] to Q^T x 1 x T.
            # Broadcasted result will be Q^T x N x T
            qg = self.quad_grid.unsqueeze(-2)
            # qg = qg + torch.randn_like(qg) * 1e-2
            xi_sigmas = xi_sigmas * qg

            inputs = xi_mus + xi_sigmas  # q^t x n x t
        if settings.debug.on():
            if not torch.is_tensor(inputs):
                raise ValueError(
                    "`inputs` should either be a MultitaskMultivariateNormal or a Tensor, got "
                    f"{inputs.__class__.__Name__}"
                )

            if inputs.size(-1) != self.input_dims:
                raise RuntimeError(
                    f"Input shape did not match self.input_dims. Got total feature dims [{inputs.size(-1)}],"
                    f" expected [{self.input_dims}]"
                )

        # Repeat the input for all possible outputs
        if self.output_dims is not None:
            inputs = inputs.unsqueeze(-3)
            inputs = inputs.expand(*inputs.shape[:-3], self.output_dims, *inputs.shape[-2:])
        # Now run samples through the GP
        output = ApproximateGP.__call__(self, inputs, **kwargs)

        if self.num_sample_sites > 0:
            if self.output_dims is not None and not isinstance(output, MultitaskMultivariateNormal):
                mean = output.loc.transpose(-1, -2)
                covar = BlockDiagLazyTensor(output.lazy_covariance_matrix, block_dim=-3)
                output = MultitaskMultivariateNormal(mean, covar, interleaved=False)
        else:
            output = output.loc.transpose(-1, -2)  # this layer provides noiseless kernel interpolation

        return output


class DeepPredictiveGaussianLikelihood(GaussianLikelihood):
    def __init__(self, dims, num_sample_sites=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_sample_sites = num_sample_sites
        self.register_parameter("raw_quad_weight_grid", torch.nn.Parameter(torch.randn(self.num_sample_sites)))

    @property
    def quad_weight_grid(self):
        qwd = self.raw_quad_weight_grid
        return qwd - qwd.logsumexp(dim=-1)

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        # Q^T x N
        base_log_marginal = super().log_marginal(observations, function_dist)
        deep_log_marginal = self.quad_weight_grid.unsqueeze(-1) + base_log_marginal

        deep_log_prob = deep_log_marginal.logsumexp(dim=-2)

        return deep_log_prob

    def forward(self, *args, **kwargs):
        pass


class MultitaskDeepPredictiveGaussianLikelihood(MultitaskGaussianLikelihood):
    def __init__(self, dims, num_sample_sites=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_sample_sites = num_sample_sites
        self.register_parameter("raw_quad_weight_grid", torch.nn.Parameter(torch.randn(self.num_sample_sites)))

    @property
    def quad_weight_grid(self):
        qwd = self.raw_quad_weight_grid
        return qwd - qwd.logsumexp(dim=-1)

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        # Q^T x N
        base_log_marginal = super().log_marginal(observations, function_dist)
        deep_log_marginal = self.quad_weight_grid.unsqueeze(-1) + base_log_marginal

        deep_log_prob = deep_log_marginal.logsumexp(dim=-2)

        return deep_log_prob

    def forward(self, *args, **kwargs):
        pass
