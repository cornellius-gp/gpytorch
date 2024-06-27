#!/usr/bin/env python3

import warnings
from functools import reduce
from operator import mul
from typing import List

import torch
from linear_operator.utils.interpolation import left_interp as _left_interp, left_t_interp as _left_t_interp


class Interpolation(object):
    def _cubic_interpolation_kernel(self, scaled_grid_dist):
        """
        Computes the interpolation kernel u() for points X given the scaled
        grid distances:
                                    (X-x_{t})/s
        where s is the distance between neighboring grid points. Note that,
        in this context, the word "kernel" is not used to mean a covariance
        function as in the rest of the package. For more details, see the
        original paper Keys et al., 1989, equation (4).

        scaled_grid_dist should be an n-by-g matrix of distances, where the
        (ij)th element is the distance between the ith data point in X and the
        jth element in the grid.

        Note that, although this method ultimately expects a scaled distance matrix,
        it is only intended to be used on single dimensional data.
        """
        U = scaled_grid_dist.abs()
        res = torch.zeros(U.size(), dtype=U.dtype, device=U.device)

        U_lt_1 = 1 - U.floor().clamp(0, 1)  # U, if U < 1, 0 otherwise
        res = res + (((1.5 * U - 2.5).mul(U)).mul(U) + 1) * U_lt_1

        # u(s) = -0.5|s|^3 + 2.5|s|^2 - 4|s| + 2 when 1 < |s| < 2
        U_ge_1_le_2 = 1 - U_lt_1  # U, if U <= 1 <= 2, 0 otherwise
        res = res + (((-0.5 * U + 2.5).mul(U) - 4).mul(U) + 2) * U_ge_1_le_2
        return res

    def interpolate(self, x_grid: List[torch.Tensor], x_target: torch.Tensor, interp_points=range(-2, 2), eps=1e-10):
        num_target_points = x_target.size(0)
        num_dim = x_target.size(-1)
        assert num_dim == len(x_grid)

        grid_sizes = [len(x_grid[i]) for i in range(num_dim)]
        # Do some boundary checking, # min/max along each dimension
        x_target_max = x_target.max(0)[0]
        x_target_min = x_target.min(0)[0]
        grid_mins = torch.stack([x_grid[i].min() for i in range(num_dim)], dim=0).to(x_target_min)
        grid_maxs = torch.stack([x_grid[i].max() for i in range(num_dim)], dim=0).to(x_target_max)

        lt_min_mask = (x_target_min - grid_mins).lt(-1e-7)
        gt_max_mask = (x_target_max - grid_maxs).gt(1e-7)
        if lt_min_mask.sum().item():
            first_out_of_range = lt_min_mask.nonzero(as_tuple=False).squeeze(1)[0].item()
            raise RuntimeError(
                (
                    "Received data that was out of bounds for the specified grid. "
                    "Grid bounds were ({:.3f}, {:.3f}), but min = {:.3f}, "
                    "max = {:.3f}"
                ).format(
                    grid_mins[first_out_of_range].item(),
                    grid_maxs[first_out_of_range].item(),
                    x_target_min[first_out_of_range].item(),
                    x_target_max[first_out_of_range].item(),
                )
            )
        if gt_max_mask.sum().item():
            first_out_of_range = gt_max_mask.nonzero(as_tuple=False).squeeze(1)[0].item()
            raise RuntimeError(
                (
                    "Received data that was out of bounds for the specified grid. "
                    "Grid bounds were ({:.3f}, {:.3f}), but min = {:.3f}, "
                    "max = {:.3f}"
                ).format(
                    grid_mins[first_out_of_range].item(),
                    grid_maxs[first_out_of_range].item(),
                    x_target_min[first_out_of_range].item(),
                    x_target_max[first_out_of_range].item(),
                )
            )

        # Now do interpolation
        interp_points = torch.tensor(interp_points, dtype=x_grid[0].dtype, device=x_grid[0].device)
        interp_points_flip = interp_points.flip(0)  # [1, 0, -1, -2]

        num_coefficients = len(interp_points)

        interp_values = torch.ones(
            num_target_points, num_coefficients**num_dim, dtype=x_grid[0].dtype, device=x_grid[0].device
        )
        interp_indices = torch.zeros(
            num_target_points, num_coefficients**num_dim, dtype=torch.long, device=x_grid[0].device
        )

        for i in range(num_dim):
            num_grid_points = x_grid[i].size(0)
            grid_delta = (x_grid[i][1] - x_grid[i][0]).clamp_min_(eps)
            # left-bounding grid point in index space
            lower_grid_pt_idxs = torch.floor((x_target[:, i] - x_grid[i][0]) / grid_delta)
            # distance from that left-bounding grid point, again in index space
            lower_pt_rel_dists = (x_target[:, i] - x_grid[i][0]) / grid_delta - lower_grid_pt_idxs
            lower_grid_pt_idxs = lower_grid_pt_idxs - interp_points.max()  # ends up being the left-most (relevant) pt
            lower_grid_pt_idxs.detach_()

            if len(lower_grid_pt_idxs.shape) == 0:
                lower_grid_pt_idxs = lower_grid_pt_idxs.unsqueeze(0)

            # get the interp. coeff. based on distances to interpolating points
            scaled_dist = lower_pt_rel_dists.unsqueeze(-1) + interp_points_flip.unsqueeze(-2)
            dim_interp_values = self._cubic_interpolation_kernel(scaled_dist)

            # Find points who's closest lower grid point is the first grid point
            # This corresponds to a boundary condition that we must fix manually.
            left_boundary_pts = (lower_grid_pt_idxs < 0).nonzero(as_tuple=False)
            num_left = len(left_boundary_pts)

            if num_left > 0:
                left_boundary_pts.squeeze_(1)
                x_grid_first = x_grid[i][:num_coefficients].unsqueeze(1).t().expand(num_left, num_coefficients)

                grid_targets = x_target.select(1, i)[left_boundary_pts].unsqueeze(1).expand(num_left, num_coefficients)
                dists = torch.abs(x_grid_first - grid_targets)
                closest_from_first = torch.min(dists, 1)[1]

                for j in range(num_left):
                    dim_interp_values[left_boundary_pts[j], :] = 0
                    dim_interp_values[left_boundary_pts[j], closest_from_first[j]] = 1
                    lower_grid_pt_idxs[left_boundary_pts[j]] = 0

            right_boundary_pts = (lower_grid_pt_idxs > num_grid_points - num_coefficients).nonzero(as_tuple=False)
            num_right = len(right_boundary_pts)

            if num_right > 0:
                right_boundary_pts.squeeze_(1)
                x_grid_last = x_grid[i][-num_coefficients:].unsqueeze(1).t().expand(num_right, num_coefficients)

                grid_targets = x_target.select(1, i)[right_boundary_pts].unsqueeze(1)
                grid_targets = grid_targets.expand(num_right, num_coefficients)
                dists = torch.abs(x_grid_last - grid_targets)
                closest_from_last = torch.min(dists, 1)[1]

                for j in range(num_right):
                    dim_interp_values[right_boundary_pts[j], :] = 0
                    dim_interp_values[right_boundary_pts[j], closest_from_last[j]] = 1
                    lower_grid_pt_idxs[right_boundary_pts[j]] = num_grid_points - num_coefficients

            offset = (interp_points - interp_points.min()).long().unsqueeze(-2)
            dim_interp_indices = lower_grid_pt_idxs.long().unsqueeze(-1) + offset  # indices of corresponding ind. pts.

            n_inner_repeat = num_coefficients**i
            n_outer_repeat = num_coefficients ** (num_dim - i - 1)
            # index_coeff = num_grid_points ** (num_dim - i - 1)  # TODO: double check
            index_coeff = reduce(mul, grid_sizes[i + 1 :], 1)  # Think this is right...
            dim_interp_indices = dim_interp_indices.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
            dim_interp_values = dim_interp_values.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
            # compute the lexicographical position of the indices in the d-dimensional grid points
            interp_indices = interp_indices.add(dim_interp_indices.view(num_target_points, -1).mul(index_coeff))
            interp_values = interp_values.mul(dim_interp_values.view(num_target_points, -1))

        return interp_indices, interp_values


def left_interp(interp_indices: torch.LongTensor, interp_values: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    warnings.warn(
        "gpytorch.utils.interpolation.left_interp is deprecated in favor of "
        "linear_operator.utils.interpolation.left_interp.",
        DeprecationWarning,
    )
    return _left_interp(interp_indices=interp_indices, interp_values=interp_values, rhs=rhs)


def left_t_interp(
    interp_indices: torch.LongTensor, interp_values: torch.Tensor, rhs: torch.Tensor, output_dim: int
) -> torch.Tensor:
    warnings.warn(
        "gpytorch.utils.interpolation.left_t_interp is deprecated in favor of "
        "linear_operator.utils.interpolation.left_t_interp.",
        DeprecationWarning,
    )
    return _left_t_interp(interp_indices=interp_indices, interp_values=interp_values, rhs=rhs, output_dim=output_dim)
