#!/usr/bin/env python3

import torch
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.utils.interpolation import left_interp

from .. import settings
from ..distributions import MultitaskMultivariateNormal, MultivariateNormal
from ..module import Module
from ._variational_strategy import _VariationalStrategy


def _select_lmc_coefficients(lmc_coefficients: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    Given a list of indices for ... x N datapoints,
      select the row from lmc_coefficient that corresponds to each datapoint

    lmc_coefficients: torch.Tensor ... x num_latents x ... x num_tasks
    indices: torch.Tesnor ... x N
    """
    batch_shape = torch.broadcast_shapes(lmc_coefficients.shape[:-1], indices.shape[:-1])

    # We will use the left_interp helper to do the indexing
    lmc_coefficients = lmc_coefficients.expand(*batch_shape, lmc_coefficients.shape[-1])[..., None]
    indices = indices.expand(*batch_shape, indices.shape[-1])[..., None]
    res = left_interp(
        indices,
        torch.ones(indices.shape, dtype=torch.long, device=indices.device),
        lmc_coefficients,
    ).squeeze(-1)
    return res


class LMCVariationalStrategy(_VariationalStrategy):
    r"""
    LMCVariationalStrategy is an implementation of the "Linear Model of Coregionalization"
    for multitask GPs. This model assumes that there are :math:`Q` latent functions
    :math:`\mathbf g(\cdot) = [g^{(1)}(\cdot), \ldots, g^{(q)}(\cdot)]`,
    each of which is modelled by a GP.
    The output functions (tasks) are linear combination of the latent functions:

    .. math::

        f_{\text{task } i}( \mathbf x) = \sum_{q=1}^Q a_i^{(q)} g^{(q)} ( \mathbf x )

    LMCVariationalStrategy wraps an existing :obj:`~gpytorch.variational.VariationalStrategy`.
    The output will either be a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal` distribution
    (if we wish to evaluate all tasks for each input) or a :obj:`~gpytorch.distributions.MultivariateNormal`
    (if we wish to evaluate a single task for each input).

    The base variational strategy is assumed to operate on a multi-batch of GPs, where one
    of the batch dimensions corresponds to the latent function dimension.

    .. note::

        The batch shape of the base :obj:`~gpytorch.variational.VariationalStrategy` does not
        necessarily have to correspond to the batch shape of the underlying GP objects.

        For example, if the base variational strategy has a batch shape of `[3]` (corresponding
        to 3 latent functions), the GP kernel object could have a batch shape of `[3]` or no
        batch shape. This would correspond to each of the latent functions having different kernels
        or the same kernel, respectivly.

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

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: The total number of tasks (output functions)
    :param int num_latents: The total number of latent functions in each group
    :param latent_dim: (Default: -1) Which batch dimension corresponds to the latent function batch.
        **Must be negative indexed**
    :type latent_dim: `int` < 0
    """

    def __init__(
        self,
        base_variational_strategy,
        num_tasks,
        num_latents=1,
        latent_dim=-1,
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

    def __call__(self, x, task_indices=None, prior=False, **kwargs):
        r"""
        Computes the variational (or prior) distribution
        :math:`q( \mathbf f \mid \mathbf X)` (or :math:`p( \mathbf f \mid \mathbf X)`).
        There are two modes:

        1.  Compute **all tasks** for all inputs.
            If this is the case, the task_indices attribute should be None.
            The return type will be a (... x N x num_tasks)
            :class:`~gpytorch.distributions.MultitaskMultivariateNormal`.
        2.  Compute **one task** per inputs.
            If this is the case, the (... x N) task_indices tensor should contain
            the indices of each input's assigned task.
            The return type will be a (... x N)
            :class:`~gpytorch.distributions.MultivariateNormal`.

        :param x: Input locations to evaluate variational strategy
        :type x: torch.Tensor (... x N x D)
        :param task_indices: (Default: None) Task index associated with each input.
            If this **is not** provided, then the returned distribution evaluates every input on every task
            (returns :class:`~gpytorch.distributions.MultitaskMultivariateNormal`).
            If this **is** provided, then the returned distribution evaluates each input only on its assigned task.
            (returns :class:`~gpytorch.distributions.MultivariateNormal`).
        :type task_indices: torch.Tensor (... x N), optional
        :param prior: (Default: False) If False, returns the variational distribution
            :math:`q( \mathbf f \mid \mathbf X)`.
            If True, returns the prior distribution
            :math:`p( \mathbf f \mid \mathbf X)`.
        :type prior: bool
        :return: :math:`q( \mathbf f \mid \mathbf X)` (or the prior),
            either for all tasks (if `task_indices == None`)
            or for a specific task (if `task_indices != None`).
        :rtype: ~gpytorch.distributions.MultitaskMultivariateNormal (... x N x num_tasks)
            or ~gpytorch.distributions.MultivariateNormal (... x N)
        """
        latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        if task_indices is None:
            num_dim = num_batch + len(latent_dist.event_shape)

            # Every data point will get an output for each task
            # Therefore, we will set up the lmc_coefficients shape for a matmul
            lmc_coefficients = self.lmc_coefficients.expand(*latent_dist.batch_shape, self.lmc_coefficients.size(-1))

            # Mean: ... x N x num_tasks
            latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
            mean = latent_mean @ lmc_coefficients.permute(
                *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
            )

            # Covar: ... x (N x num_tasks) x (N x num_tasks)
            latent_covar = latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(settings.cholesky_jitter.value(dtype=mean.dtype))

            # Done!
            function_dist = MultitaskMultivariateNormal(mean, covar)

        else:
            # Each data point will get a single output corresponding to a single task
            # Therefore, we will select the appropriate lmc coefficients for each task
            lmc_coefficients = _select_lmc_coefficients(self.lmc_coefficients, task_indices)

            # Mean: ... x N
            mean = (latent_dist.mean * lmc_coefficients).sum(latent_dim)

            # Covar: ... x N x N
            latent_covar = latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = (latent_covar * lmc_factor).sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(settings.cholesky_jitter.value(dtype=mean.dtype))

            # Done!
            function_dist = MultivariateNormal(mean, covar)

        return function_dist
