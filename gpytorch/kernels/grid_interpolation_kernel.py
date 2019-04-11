#!/usr/bin/env python3

import torch
from .grid_kernel import GridKernel
from ..lazy import InterpolatedLazyTensor
from ..models.exact_prediction_strategies import InterpolatedPredictionStrategy
from ..utils.interpolation import Interpolation


class GridInterpolationKernel(GridKernel):
    r"""
    Implements the KISS-GP (or SKI) approximation for a given kernel.
    It was proposed in `Kernel Interpolation for Scalable Structured Gaussian Processes`_,
    and offers extremely fast and accurate Kernel approximations for large datasets.

    Given a base kernel `k`, the covariance :math:`k(\mathbf{x_1}, \mathbf{x_2})` is approximated by
    using a grid of regularly spaced *inducing points*:

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = \mathbf{w_{x_1}}^\top K_{U,U} \mathbf{w_{x_2}}
       \end{equation*}

    where

    * :math:`U` is the set of gridded inducing points

    * :math:`K_{U,U}` is the kernel matrix between the inducing points

    * :math:`\mathbf{w_{x_1}}` and :math:`\mathbf{w_{x_2}}` are sparse vectors based on
      :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` that apply cubic interpolation.

    The user should supply the size of the grid (using the :attr:`grid_size` attribute).
    To choose a reasonable grid value, we highly recommend using the
    :func:`gpytorch.utils.grid.choose_grid_size` helper function.
    The bounds of the grid will automatically be determined by data.

    (Alternatively, you can hard-code bounds using the :attr:`grid_bounds`, which
    will speed up this kernel's computations.)

    .. note::

        `GridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        - :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        - :attr:`grid_size` (int):
            The size of the grid (in each dimension)
        - :attr:`num_dims` (int):
            The dimension of the input data. Required if `grid_bounds=None`
        - :attr:`grid_bounds` (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the number of dimensions.
            The entries represent the min/max values for each dimension.
        - :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Kernel Interpolation for Scalable Structured Gaussian Processes:
        http://proceedings.mlr.press/v37/wilson15.pdf
    """

    def __init__(self, base_kernel, grid_size, num_dims=None, grid_bounds=None, active_dims=None):
        has_initialized_grid = 0
        grid_is_dynamic = True

        # Make some temporary grid bounds, if none exist
        if grid_bounds is None:
            if num_dims is None:
                raise RuntimeError("num_dims must be supplied if grid_bounds is None")
            else:
                # Create some temporary grid bounds - they'll be changed soon
                grid_bounds = tuple((-1.0, 1.0) for _ in range(num_dims))
        else:
            has_initialized_grid = 1
            grid_is_dynamic = False
            if num_dims is None:
                num_dims = len(grid_bounds)
            elif num_dims != len(grid_bounds):
                raise RuntimeError(
                    "num_dims ({}) disagrees with the number of supplied "
                    "grid_bounds ({})".format(num_dims, len(grid_bounds))
                )

        # Initialize values and the grid
        self.grid_is_dynamic = grid_is_dynamic
        self.num_dims = num_dims
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        grid = self._create_grid()

        super(GridInterpolationKernel, self).__init__(
            base_kernel=base_kernel, grid=grid, interpolation_mode=True, active_dims=active_dims
        )
        self.register_buffer("has_initialized_grid", torch.tensor(has_initialized_grid, dtype=torch.uint8))

    def _create_grid(self):
        grid = torch.zeros(self.grid_size, len(self.grid_bounds))
        for i in range(len(self.grid_bounds)):
            grid_diff = float(self.grid_bounds[i][1] - self.grid_bounds[i][0]) / (self.grid_size - 2)
            grid[:, i] = torch.linspace(
                self.grid_bounds[i][0] - grid_diff, self.grid_bounds[i][1] + grid_diff, self.grid_size
            )

        return grid

    @property
    def _tight_grid_bounds(self):
        grid_spacings = tuple((bound[1] - bound[0]) / self.grid_size for bound in self.grid_bounds)
        return tuple(
            (bound[0] + 2.01 * spacing, bound[1] - 2.01 * spacing)
            for bound, spacing in zip(self.grid_bounds, grid_spacings)
        )

    def _compute_grid(self, inputs, last_dim_is_batch=False):
        n_data, n_dimensions = inputs.size(-2), inputs.size(-1)
        if last_dim_is_batch:
            inputs = inputs.transpose(-1, -2).unsqueeze(-1)
            n_dimensions = 1
        batch_shape = inputs.shape[:-2]

        inputs = inputs.contiguous().view(-1, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        interp_indices = interp_indices.view(*batch_shape, n_data, -1)
        interp_values = interp_values.view(*batch_shape, n_data, -1)
        return interp_indices, interp_values

    def _inducing_forward(self, last_dim_is_batch, **params):
        return super().forward(self.grid, self.grid, last_dim_is_batch=last_dim_is_batch, **params)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # See if we need to update the grid or not
        if self.grid_is_dynamic:  # This is true if a grid_bounds wasn't passed in
            if torch.equal(x1, x2):
                x = x1.contiguous().view(-1, self.num_dims)
            else:
                x = torch.cat([x1.contiguous().view(-1, self.num_dims), x2.contiguous().view(-1, self.num_dims)])
            x_maxs = x.max(0)[0].tolist()
            x_mins = x.min(0)[0].tolist()

            # We need to update the grid if
            # 1) it hasn't ever been initialized, or
            # 2) if any of the grid points are "out of bounds"
            update_grid = (not self.has_initialized_grid.item()) or any(
                x_min < bound[0] or x_max > bound[1]
                for x_min, x_max, bound in zip(x_mins, x_maxs, self._tight_grid_bounds)
            )

            # Update the grid if needed
            if update_grid:
                grid_spacings = tuple((x_max - x_min) / (self.grid_size - 4.02) for x_min, x_max in zip(x_mins, x_maxs))
                self.grid_bounds = tuple(
                    (x_min - 2.01 * spacing, x_max + 2.01 * spacing)
                    for x_min, x_max, spacing in zip(x_mins, x_maxs, grid_spacings)
                )
                grid = self._create_grid()
                self.update_grid(grid)

        base_lazy_tsr = self._inducing_forward(last_dim_is_batch=last_dim_is_batch, **params)
        if last_dim_is_batch:
            base_lazy_tsr = base_lazy_tsr.repeat(*x1.shape[:-2], x1.size(-1), 1, 1)
        if x1.dim() > 2:
            base_lazy_tsr = base_lazy_tsr.repeat(*x1.shape[:-2], 1, 1)

        left_interp_indices, left_interp_values = self._compute_grid(x1, last_dim_is_batch)
        if torch.equal(x1, x2):
            right_interp_indices = left_interp_indices
            right_interp_values = left_interp_values
        else:
            right_interp_indices, right_interp_values = self._compute_grid(x2, last_dim_is_batch)

        res = InterpolatedLazyTensor(
            base_lazy_tsr,
            left_interp_indices.detach(),
            left_interp_values,
            right_interp_indices.detach(),
            right_interp_values,
        )

        if diag:
            return res.diag()
        else:
            return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return InterpolatedPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
