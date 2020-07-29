import torch

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import BlockDiagLazyTensor

from ..approximate_gp import ApproximateGP
from .deep_gp import DeepGP, DeepGPLayer


class DSPPLayer(DeepGPLayer):
    def __init__(self, variational_strategy, input_dims, output_dims, num_sample_sites=3, quad_sites=None):
        super().__init__(variational_strategy, input_dims, output_dims)

        self.num_sample_sites = num_sample_sites

        # Pass in previous_layer.quad_sites if you want to share quad_sites across layers.
        if quad_sites is not None:
            self.quad_sites = quad_sites
        else:
            self.quad_sites = torch.nn.Parameter(torch.randn(num_sample_sites, input_dims))

    def __call__(self, inputs, are_samples=False, expand_for_quadgrid=True, **kwargs):
        if isinstance(inputs, MultitaskMultivariateNormal):
            # inputs is definitely in the second layer, and mean is n x t
            mus, sigmas = inputs.mean, inputs.variance.sqrt()

            if expand_for_quadgrid:
                xi_mus = mus.unsqueeze(0)  # 1 x n x t
                xi_sigmas = sigmas.unsqueeze(0)  # 1 x n x t
            else:
                xi_mus = mus
                xi_sigmas = sigmas

            # unsqueeze sigmas to 1 x n x t, locations from [q] to Q^T x 1 x T.
            # Broadcasted result will be Q^T x N x T
            qg = self.quad_sites.view([self.num_sample_sites] + [1] * (xi_mus.dim() - 2) + [self.input_dims])
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


class DSPP(DeepGP):
    def __init__(self, num_sample_sites):
        super().__init__()
        self.num_sample_sites = num_sample_sites
        self.register_parameter("raw_quad_weights", torch.nn.Parameter(torch.randn(self.num_sample_sites)))

    @property
    def quad_weights(self):
        qwd = self.raw_quad_weights
        return qwd - qwd.logsumexp(dim=-1)
