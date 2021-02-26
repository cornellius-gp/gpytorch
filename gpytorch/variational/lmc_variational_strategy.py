#!/usr/bin/env python3

import torch

from ..distributions import MultitaskMultivariateNormal
from ..lazy import KroneckerProductLazyTensor, MatmulLazyTensor
from ..module import Module
from ._variational_strategy import _VariationalStrategy


class LMCVariationalStrategy(_VariationalStrategy):
    r"""
    LMCVariationalStrategy is an implementation of the "Linear Model of Coregionalization"
    for multitask GPs. This model assumes that there are :math:`Q` latent functions
    :math:`\mathbf g(\cdot) = [g^{(1)}(\cdot), \ldots, g^{(q)}(\cdot)]`,
    each of which is modelled by a GP.
    The output functions (tasks) are linear combination of the latent functions:

    .. math::

        f_{\text{task } i}( \mathbf x) = \sum_{q=1}^Q a_i^{(q)} g^{(q)} ( \mathbf x )

    LMCVariationalStrategy wraps an existing :obj:`~gpytorch.variational.VariationalStrategy`
    to produce a :obj:`~gpytorch.variational.MultitaskMultivariateNormal` distribution.
    The base variational strategy is assumed to operate on a multi-batch of GPs, where one
    of the batch dimensions corresponds to the latent function dimension.

    .. note::

        The batch shape of the base :obj:`~gpytorch.variational.VariationalStrategy` does not
        necessarily have to correspond to the batch shape of the underlying GP objects.

        For example, if the base variational strategy has a batch shape of `[3]` (corresponding
        to 3 latent functions), the GP kernel object could have a batch shape of `[3]` or no
        batch shape. This would correspond to each of the latent functions having different kernels
        or the same kernel, respectivly.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: The total number of tasks (output functions)
    :param int num_latents: The total number of latent functions in each group
    :param latent_dim: (Default: -1) Which batch dimension corresponds to the latent function batch.
        **Must be negative indexed**
    :type latent_dim: `int` < 0

    Example:
        >>> class LMCMultitaskGP(gpytorch.models.ApproximateGP):
        >>>     '''
        >>>     3 latent functions
        >>>     5 output dimensions (tasks)
        >>>     '''
        >>>     def __init__(self):
        >>>         # Each latent function shares the same inducing points
        >>>         # We'll have 32 inducing points, and let's assume the input dimensionality is 2
        >>>         inducing_points = torch.randn(32, 2)
        >>>
        >>>         # The variational parameters have a batch_shape of [3] - for 3 latent functions
        >>>         variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
        >>>             inducing_points.size(-1), batch_shape=torch.Size([3]),
        >>>         )
        >>>         variational_strategy = gpytorch.variational.LMCVariationalStrategy(
        >>>             gpytorch.variational.VariationalStrategy(
        >>>                 inducing_points, variational_distribution, learn_inducing_locations=True,
        >>>             ),
        >>>             num_tasks=5,
        >>>             num_latents=3,
        >>>             latent_dim=0,
        >>>         )
        >>>
        >>>         # Each latent function has its own mean/kernel function
        >>>         super().__init__(variational_strategy)
        >>>         self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3]))
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(
        >>>             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3])),
        >>>             batch_shape=torch.Size([3]),
        >>>         )
        >>>
        >>> # Model output: n x 5
    """

    def __init__(
        self, base_variational_strategy, num_tasks, num_latents=1, latent_dim=-1,
    ):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = num_tasks
        batch_shape = self.base_variational_strategy._variational_distribution.batch_shape

        # Check if no functions
        if latent_dim >= 0:
            raise RuntimeError(f"latent_dim must be a negative indexed batch dimension: got {latent_dim}.")
        if not (batch_shape[latent_dim] == num_latents or batch_shape[latent_dim] == 1):
            raise RuntimeError(
                f"Mismatch in num_latents: got a variational distribution of batch shape {batch_shape}, "
                f"expected the function dim {latent_dim} to be {num_latents}."
            )
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Make the batch_shape
        self.batch_shape = list(batch_shape)
        del self.batch_shape[self.latent_dim]
        self.batch_shape = torch.Size(self.batch_shape)

        # LCM coefficients
        lmc_coefficients = torch.randn(*batch_shape, self.num_tasks)
        self.register_parameter("lmc_coefficients", torch.nn.Parameter(lmc_coefficients))

    @property
    def prior_distribution(self):
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self):
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self):
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=self.latent_dim)

    def __call__(self, x, prior=False, **kwargs):
        function_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        lmc_coefficients = self.lmc_coefficients.expand(*function_dist.batch_shape, self.lmc_coefficients.size(-1))
        num_batch = len(function_dist.batch_shape)
        num_dim = num_batch + len(function_dist.event_shape)
        latent_dim = num_batch + self.latent_dim if self.latent_dim is not None else None

        # Mean
        mean = function_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
        mean = mean @ lmc_coefficients.permute(
            *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
        )

        # Covar
        covar = function_dist.lazy_covariance_matrix
        lmc_factor = MatmulLazyTensor(lmc_coefficients.unsqueeze(-1), lmc_coefficients.unsqueeze(-2))
        covar = KroneckerProductLazyTensor(covar, lmc_factor)
        covar = covar.sum(latent_dim)

        # Add a bit of jitter to make the covar PD
        covar = covar.add_jitter(1e-6)

        # Done!
        function_dist = MultitaskMultivariateNormal(mean, covar)
        return function_dist
