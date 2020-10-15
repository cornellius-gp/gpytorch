#!/usr/bin/env python3

from typing import List

import torch
from torch import Tensor

from .. import settings
from ..lazy import KroneckerProductLazyTensor, ToeplitzLazyTensor, delazify
from ..utils.grid import convert_legacy_grid, create_data_from_grid
from .kernel import Kernel


class GridKernel(Kernel):
    r"""
    If the input data :math:`X` are regularly spaced on a grid, then
    `GridKernel` can dramatically speed up computatations for stationary kernel.

    GridKernel exploits Toeplitz and Kronecker structure within the covariance matrix.
    See `Fast kernel learning for multidimensional pattern extrapolation`_ for more info.

    .. note::

        `GridKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to speed up with grid methods.
        :attr:`grid` (Tensor):
            A g x d tensor where column i consists of the projections of the
            grid in dimension i.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.
        :attr:`interpolation_mode` (bool):
            Used for GridInterpolationKernel where we want the covariance
            between points in the projections of the grid of each dimension.
            We do this by treating `grid` as d batches of g x 1 tensors by
            calling base_kernel(grid, grid) with last_dim_is_batch to get a d x g x g Tensor
            which we Kronecker product to get a g x g KroneckerProductLazyTensor.

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    is_stationary = True

    def __init__(
        self, base_kernel: Kernel, grid: List[Tensor], interpolation_mode: bool = False, active_dims: bool = None
    ):
        if not base_kernel.is_stationary:
            raise RuntimeError("The base_kernel for GridKernel must be stationary.")

        super().__init__(active_dims=active_dims)
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)
        self.interpolation_mode = interpolation_mode
        self.base_kernel = base_kernel
        self.num_dims = len(grid)
        self.register_buffer_list("grid", grid)
        if not self.interpolation_mode:
            self.register_buffer("full_grid", create_data_from_grid(grid))

    def _clear_cache(self):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat

    def register_buffer_list(self, base_name, tensors):
        """Helper to register several buffers at once under a single base name"""
        for i, tensor in enumerate(tensors):
            self.register_buffer(base_name + "_" + str(i), tensor)

    @property
    def grid(self):
        return [getattr(self, f"grid_{i}") for i in range(self.num_dims)]

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)

        if len(grid) != self.num_dims:
            raise RuntimeError("New grid should have the same number of dimensions as before.")

        for i in range(self.num_dims):
            setattr(self, f"grid_{i}", grid[i])

        if not self.interpolation_mode:
            self.full_grid = create_data_from_grid(self.grid)

        self._clear_cache()
        return self

    @property
    def is_ragged(self):
        return not all(self.grid[0].size() == proj.size() for proj in self.grid)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch and not self.interpolation_mode:
            raise ValueError("last_dim_is_batch is only valid with interpolation model")

        grid = self.grid
        if self.is_ragged:
            # Pad the grid - so that grid is the same size for each dimension
            max_grid_size = max(proj.size(-1) for proj in grid)
            padded_grid = []
            for proj in grid:
                padding_size = max_grid_size - proj.size(-1)
                if padding_size > 0:
                    dtype = proj.dtype
                    device = proj.device
                    padded_grid.append(
                        torch.cat([proj, torch.zeros(*proj.shape[:-1], padding_size, dtype=dtype, device=device)])
                    )
                else:
                    padded_grid.append(proj)
        else:
            padded_grid = grid

        if not self.interpolation_mode:
            if len(x1.shape[:-2]):
                full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])
            else:
                full_grid = self.full_grid

        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat
            # Can exploit Toeplitz structure if grid points in each dimension are equally
            # spaced and using a translation-invariant kernel
            if settings.use_toeplitz.on():
                # Use padded grid for batch mode
                first_grid_point = torch.stack([proj[0].unsqueeze(0) for proj in grid], dim=-1)
                full_grid = torch.stack(padded_grid, dim=-1)
                covars = delazify(self.base_kernel(first_grid_point, full_grid, last_dim_is_batch=True, **params))

                if last_dim_is_batch:
                    # Toeplitz expects batches of columns so we concatenate the
                    # 1 x grid_size[i] tensors together
                    # Note that this requires all the dimensions to have the same number of grid points
                    covar = ToeplitzLazyTensor(covars.squeeze(-2))
                else:
                    # Non-batched ToeplitzLazyTensor expects a 1D tensor, so we squeeze out the row dimension
                    covars = covars.squeeze(-2)  # Get rid of the dimension corresponding to the first point
                    # Un-pad the grid
                    covars = [ToeplitzLazyTensor(covars[..., i, : proj.size(-1)]) for i, proj in enumerate(grid)]
                    # Due to legacy reasons, KroneckerProductLazyTensor(A, B, C) is actually (C Kron B Kron A)
                    covar = KroneckerProductLazyTensor(*covars[::-1])
            else:
                full_grid = torch.stack(padded_grid, dim=-1)
                covars = delazify(self.base_kernel(full_grid, full_grid, last_dim_is_batch=True, **params))
                if last_dim_is_batch:
                    # Note that this requires all the dimensions to have the same number of grid points
                    covar = covars
                else:
                    covars = [covars[..., i, : proj.size(-1), : proj.size(-1)] for i, proj in enumerate(self.grid)]
                    covar = KroneckerProductLazyTensor(*covars[::-1])

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
        else:
            return self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
