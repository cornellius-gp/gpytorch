#!/usr/bin/env python3

import warnings
from typing import Iterable, Union

import torch
from jaxtyping import Float
from linear_operator import to_dense
from linear_operator.operators import KroneckerProductLinearOperator, LinearOperator, ToeplitzLinearOperator
from torch import Tensor

from .. import settings
from ..utils.grid import create_data_from_grid
from .kernel import Kernel


class GridKernel(Kernel):
    r"""
    `GridKernel` wraps a stationary kernel that is computed on a (multidimensional)
    grid that is regularly spaced along each dimension.
    It exploits Toeplitz and Kronecker structure within the covariance matrix
    for massive computational speedups.
    See `Fast kernel learning for multidimensional pattern extrapolation`_ for more info.

    .. note::

        `GridKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    :param base_kernel: The stationary kernel to speed up with grid methods.
    :param grid: A list of tensors where tensor `i` consists of the projections
        of the grid in dimension i.
    :param active_dims:

    :ivar ragged_grid: A concatenation of all grid projections
    :type ragged_grid: Tensor (max(M_i) x D)
    :ivar full_grid: A full representation of the grid
    :type ragged_grid: Tensor (N x D)

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    def __init__(
        self,
        base_kernel: Kernel,
        grid: Iterable[Float[Tensor, "M_i"]],  # noqa F821
        **kwargs,
    ):
        if not base_kernel.is_stationary:
            raise RuntimeError("The base_kernel for GridKernel must be stationary.")
        batch_shapes, num_grid_points = zip(*[(sub_grid.shape[:-1], sub_grid.shape[-1]) for sub_grid in grid])

        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.num_dims = len(grid)
        self.num_grid_points = num_grid_points

        # Store each grid in a buffer
        for i, sub_grid in enumerate(grid):
            assert sub_grid.dim() == 1
            self.register_buffer(f"grid_{i}", sub_grid)

        # Create a buffer to store a concatenation of all grids
        num_grid_points = [sub_grid.size(-1) for sub_grid in grid]
        ragged_grid: Float[Tensor, "M D"] = torch.zeros(
            max(self.num_grid_points), self.num_dims, dtype=grid[0].dtype, device=grid[0].device
        )
        self.register_buffer("ragged_grid", ragged_grid)

        # Update the ragged_grid buffer
        # Also create the full_grid buffer
        self.update_grid(grid)

    @property
    def _lazily_evaluate(self) -> bool:
        # Toeplitz structure is very efficient; no need to lazily evaluate
        return False

    @property
    def is_stationary(self) -> bool:
        return True

    def _clear_cache(self):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat

    def _create_or_update_full_grid(self, grid: Iterable[Float[Tensor, "M_i"]]):  # noqa F821
        full_grid = create_data_from_grid(self.grid)
        if hasattr(self, "full_grid"):
            self.full_grid.reshape(full_grid.shape)
            self.full_grid.copy_(full_grid.type_as(self.full_grid))
        else:
            self.register_buffer("full_grid", full_grid)

    def _validate_inputs(self, x: Float[Tensor, "... N D"]) -> bool:
        return torch.equal(self.full_grid.expand(*x.shape[:-2], *self.full_grid.shape[-2:]), x)

    @property
    def grid(self) -> Float[Tensor, "N D"]:
        return [getattr(self, f"grid_{i}") for i in range(self.num_dims)]

    def update_grid(self, grid: Iterable[Float[Tensor, "M_i"]]):  # noqa F821
        """
        Supply a new `grid` if it ever changes.
        """
        if len(grid) != self.num_dims:
            raise RuntimeError("New grid should have the same number of dimensions as before.")
        num_grid_points = [sub_grid.size(-1) for sub_grid in grid]

        # Update the size of the ragged_grid buffer
        self.ragged_grid.reshape(max(self.num_grid_points), self.num_dims)
        self.num_grid_points = num_grid_points

        # Update the grid and ragged_grid buffers
        for i, (num_grid_point, sub_grid) in enumerate(zip(num_grid_points, grid)):
            assert sub_grid.dim() == 1
            getattr(self, f"grid_{i}").reshape(sub_grid.shape)
            getattr(self, f"grid_{i}").copy_(sub_grid.type_as(self.ragged_grid))
            # Grids aren't necessarily the same size across each dimension
            # Some grids will be padded by zeros, which will be removed after computing kernel rows
            self.ragged_grid[..., :num_grid_point, i] = sub_grid.type_as(self.ragged_grid)

        # Update the full_grid buffer
        self._create_or_update_full_grid(grid)

        # Clear cache
        self._clear_cache()
        return self

    def forward(
        self, x1: Float[Tensor, "... N_1 D"], x2: Float[Tensor, "... N_2 D"], diag: bool = False, **params
    ) -> Union[Float[LinearOperator, "... N_1 N_2"], Float[Tensor, "... N_1"]]:
        if diag:
            return self.base_kernel(x1, x2, diag=True, **params)

        # If this kernel is not called with the grid data, directly call base_kernel
        if not (self._validate_inputs(x1) and self._validate_inputs(x2)):
            warnings.warn("GridKernel was called with non-grid data.", RuntimeWarning)
            return self.base_kernel(x1, x2, diag=False, **params)

        # Default case
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat

        first_grid_points = self.ragged_grid[..., :1, :]

        # Compute the first rows of each univariate kernel on each of the D-dimensions
        # The result will be batched and stored in a D x ... x M matrix
        # Hack:
        # Base kernel expects a d-dimensional input. To compute the kernel on
        # the grid projected ondo dim i, we zero the data in all other dimensions.
        # Since the kernel is stationary, the other dimensions won't contribute to the covariance.
        batch_shape = torch.broadcast_shapes(self.ragged_grid.shape[:-2], self.base_kernel.batch_shape)
        masks = torch.eye(self.num_dims, dtype=self.ragged_grid.dtype, device=self.ragged_grid.device).view(
            self.num_dims, *[1 for _ in batch_shape], 1, self.num_dims
        )  # D x ... x 1 x D
        # This mask will zero out all but the i^th dimension for the i^th batch member
        with settings.lazily_evaluate_kernels(False):
            unidimensional_kernel_first_rows = to_dense(
                self.base_kernel(first_grid_points * masks, self.ragged_grid * masks, **params)
            )  # D x ... x M

        # Convert the first rows of the unidimensional kernels into ToeplitzLinearOperators
        # (Un-pad the kernel first row as necessary)
        unidimensional_kernels = [
            ToeplitzLinearOperator(unidimensional_kernel_first_rows[i, ..., 0, :num_grid_point])
            for i, num_grid_point in enumerate(self.num_grid_points)
        ]  # D x ... x M_i x M_i
        # Due to legacy reasons, KroneckerProductLinearOperator(A, B, C) is actually (C Kron B Kron A)
        covar = KroneckerProductLinearOperator(*unidimensional_kernels[::-1])  # ... x N x N

        if not self.training:
            self._cached_kernel_mat = covar

        return covar

    def num_outputs_per_input(self, x1: Float[Tensor, "... N_1 D"], x2: Float[Tensor, "... N_2 D"]) -> int:
        return self.base_kernel.num_outputs_per_input(x1, x2)
