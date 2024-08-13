#!/usr/bin/env python3

from typing import Iterable, Optional, Tuple

import torch
from jaxtyping import Float
from linear_operator.operators import InterpolatedLinearOperator
from linear_operator.utils.interpolation import left_interp
from torch import Tensor

from ..distributions import MultivariateNormal
from ..models import ApproximateGP
from ..utils.interpolation import Interpolation
from ..utils.memoize import cached
from ._variational_distribution import _VariationalDistribution
from ._variational_strategy import _VariationalStrategy


class GridInterpolationVariationalStrategy(_VariationalStrategy):
    r"""
    This strategy constrains the inducing points to a grid and applies a deterministic
    relationship between :math:`\mathbf f` and :math:`\mathbf u`.
    It was introduced by `Wilson et al. (2016)`_.

    Here, the inducing points are not learned. Instead, the strategy
    automatically creates inducing points based on a set of grid sizes and grid
    bounds.

    .. _Wilson et al. (2016):
        https://arxiv.org/abs/1611.00336

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param grid_size: Size of the grid
    :param grid_bounds: Bounds of each dimension of the grid (should be a list of (float, float) tuples)
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`

    :ivar grid: The grid of points that the inducing points are based on.
        The grid is stored as a matrix, where each column corresponds to the
        projection of the grid onto one dimension.
    :type grid: torch.Tensor (M x D)
    """

    def __init__(
        self,
        model: ApproximateGP,
        grid_size: int,
        grid_bounds: Iterable[Tuple[float, float]],
        variational_distribution: _VariationalDistribution,
    ):
        grid = torch.zeros(grid_size, len(grid_bounds))
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)

        inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[j * grid_size**i : (j + 1) * grid_size**i, i].fill_(grid[j, i])
                if prev_points is not None:
                    inducing_points[j * grid_size**i : (j + 1) * grid_size**i, :i].copy_(prev_points)
            prev_points = inducing_points[: grid_size ** (i + 1), : (i + 1)]

        super(GridInterpolationVariationalStrategy, self).__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        object.__setattr__(self, "model", model)
        self.register_buffer("grid", grid)

    def _compute_grid(
        self,
        inputs: Float[Tensor, "... N D"],
    ) -> Tuple[Float[Tensor, "... N num_interp"], Float[Tensor, "... N num_interp"]]:
        *batch_shape, n_data, n_dimensions = inputs.shape
        grid = tuple(self.grid[..., i] for i in range(n_dimensions))

        inputs = inputs.reshape(-1, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(grid, inputs)
        interp_indices = interp_indices.view(*batch_shape, n_data, -1)
        interp_values = interp_values.view(*batch_shape, n_data, -1)

        if (interp_indices.dim() - 2) != len(self._variational_distribution.batch_shape):
            batch_shape = torch.broadcast_shapes(interp_indices.shape[:-2], self._variational_distribution.batch_shape)
            interp_indices = interp_indices.expand(*batch_shape, *interp_indices.shape[-2:])
            interp_values = interp_values.expand(*batch_shape, *interp_values.shape[-2:])
        return interp_indices, interp_values

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> Float[MultivariateNormal, "M"]:  # noqa: F821
        out = self.model.forward(self.inducing_points)
        # TODO: investigate why smaller than 1e-3 breaks some tests
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter(1e-3))
        return res

    def forward(
        self,
        x: Float[Tensor, "... N D"],
        inducing_points: Float[Tensor, "... M D"],
        inducing_values: Float[Tensor, "... M"],
        variational_inducing_covar: Optional[Float[Tensor, "... M M"]] = None,
    ) -> Float[MultivariateNormal, "... N"]:
        if variational_inducing_covar is None:
            raise RuntimeError(
                "GridInterpolationVariationalStrategy is only compatible with Gaussian variational "
                f"distributions. Got ({self.variational_distribution.__class__.__name__}."
            )

        variational_distribution = self.variational_distribution

        # Get interpolations
        interp_indices, interp_values = self._compute_grid(x)

        # Compute test mean
        # Left multiply samples by interpolation matrix
        predictive_mean = left_interp(interp_indices, interp_values, inducing_values.unsqueeze(-1))
        predictive_mean = predictive_mean.squeeze(-1)

        # Compute test covar
        predictive_covar = InterpolatedLinearOperator(
            variational_distribution.lazy_covariance_matrix,
            interp_indices,
            interp_values,
            interp_indices,
            interp_values,
        )
        output = MultivariateNormal(predictive_mean, predictive_covar)
        return output
