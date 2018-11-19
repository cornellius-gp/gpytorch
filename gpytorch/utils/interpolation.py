#!/usr/bin/env python3

from copy import deepcopy

import torch


class Interpolation(object):
    """
    """

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

    def interpolate(self, x_grid, x_target, interp_points=range(-2, 2)):
        # Do some boundary checking
        grid_mins = x_grid.min(0)[0]
        grid_maxs = x_grid.max(0)[0]
        x_target_min = x_target.min(0)[0]
        x_target_max = x_target.min(0)[0]
        lt_min_mask = (x_target_min - grid_mins).lt(-1e-7)
        gt_max_mask = (x_target_max - grid_maxs).gt(1e-7)
        if lt_min_mask.sum().item():
            first_out_of_range = lt_min_mask.nonzero().squeeze(1)[0].item()
            raise RuntimeError(
                (
                    "Received data that was out of bounds for the specified grid. "
                    "Grid bounds were ({0:.3f}, {0:.3f}), but min = {0:.3f}, "
                    "max = {0:.3f}"
                ).format(
                    grid_mins[first_out_of_range].item(),
                    grid_maxs[first_out_of_range].item(),
                    x_target_min[first_out_of_range].item(),
                    x_target_max[first_out_of_range].item(),
                )
            )
        if gt_max_mask.sum().item():
            first_out_of_range = gt_max_mask.nonzero().squeeze(1)[0].item()
            raise RuntimeError(
                (
                    "Received data that was out of bounds for the specified grid. "
                    "Grid bounds were ({0:.3f}, {0:.3f}), but min = {0:.3f}, "
                    "max = {0:.3f}"
                ).format(
                    grid_mins[first_out_of_range].item(),
                    grid_maxs[first_out_of_range].item(),
                    x_target_min[first_out_of_range].item(),
                    x_target_max[first_out_of_range].item(),
                )
            )

        # Now do interpolation
        interp_points = torch.tensor(interp_points, dtype=x_grid.dtype, device=x_grid.device)
        interp_points_flip = interp_points.flip(0)

        num_grid_points = x_grid.size(0)
        num_target_points = x_target.size(0)
        num_dim = x_target.size(-1)
        num_coefficients = len(interp_points)

        interp_values = torch.ones(
            num_target_points, num_coefficients ** num_dim, dtype=x_grid.dtype, device=x_grid.device
        )
        interp_indices = torch.zeros(
            num_target_points, num_coefficients ** num_dim, dtype=torch.long, device=x_grid.device
        )

        for i in range(num_dim):
            grid_delta = x_grid[1, i] - x_grid[0, i]
            lower_grid_pt_idxs = torch.floor((x_target[:, i] - x_grid[0, i]) / grid_delta).squeeze()
            lower_pt_rel_dists = (x_target[:, i] - x_grid[0, i]) / grid_delta - lower_grid_pt_idxs
            lower_grid_pt_idxs = lower_grid_pt_idxs - interp_points.max()
            lower_grid_pt_idxs.detach_()

            if len(lower_grid_pt_idxs.shape) == 0:
                lower_grid_pt_idxs = lower_grid_pt_idxs.unsqueeze(0)

            scaled_dist = lower_pt_rel_dists.unsqueeze(-1) + interp_points_flip.unsqueeze(-2)
            dim_interp_values = self._cubic_interpolation_kernel(scaled_dist)

            # Find points who's closest lower grid point is the first grid point
            # This corresponds to a boundary condition that we must fix manually.
            left_boundary_pts = torch.nonzero(lower_grid_pt_idxs < 1)
            num_left = len(left_boundary_pts)

            if num_left > 0:
                left_boundary_pts.squeeze_(1)
                x_grid_first = x_grid[:num_coefficients, i].unsqueeze(1).t().expand(num_left, num_coefficients)

                grid_targets = x_target.select(1, i)[left_boundary_pts].unsqueeze(1).expand(num_left, num_coefficients)
                dists = torch.abs(x_grid_first - grid_targets)
                closest_from_first = torch.min(dists, 1)[1]

                for j in range(num_left):
                    dim_interp_values[left_boundary_pts[j], :] = 0
                    dim_interp_values[left_boundary_pts[j], closest_from_first[j]] = 1
                    lower_grid_pt_idxs[left_boundary_pts[j]] = 0

            right_boundary_pts = torch.nonzero(lower_grid_pt_idxs > num_grid_points - num_coefficients)
            num_right = len(right_boundary_pts)

            if num_right > 0:
                right_boundary_pts.squeeze_(1)
                x_grid_last = x_grid[-num_coefficients:, i].unsqueeze(1).t().expand(num_right, num_coefficients)

                grid_targets = x_target.select(1, i)[right_boundary_pts].unsqueeze(1)
                grid_targets = grid_targets.expand(num_right, num_coefficients)
                dists = torch.abs(x_grid_last - grid_targets)
                closest_from_last = torch.min(dists, 1)[1]

                for j in range(num_right):
                    dim_interp_values[right_boundary_pts[j], :] = 0
                    dim_interp_values[right_boundary_pts[j], closest_from_last[j]] = 1
                    lower_grid_pt_idxs[right_boundary_pts[j]] = num_grid_points - num_coefficients

            offset = (interp_points - interp_points.min()).long().unsqueeze(-2)
            dim_interp_indices = lower_grid_pt_idxs.long().unsqueeze(-1) + offset

            n_inner_repeat = num_coefficients ** i
            n_outer_repeat = num_coefficients ** (num_dim - i - 1)
            index_coeff = num_grid_points ** (num_dim - i - 1)
            dim_interp_indices = dim_interp_indices.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
            dim_interp_values = dim_interp_values.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
            interp_indices = interp_indices.add(dim_interp_indices.view(num_target_points, -1).mul(index_coeff))
            interp_values = interp_values.mul(dim_interp_values.view(num_target_points, -1))

        return interp_indices, interp_values


def left_interp(interp_indices, interp_values, rhs):
    """
    """
    is_vector = rhs.ndimension() == 1

    if is_vector:
        res = rhs.index_select(0, interp_indices.view(-1)).view(*interp_values.size())
        res = res.mul(interp_values)
        res = res.sum(-1)
        return res

    else:
        # Special cuda version -- this is faster on the GPU for some reason
        if interp_indices.is_cuda:
            if interp_indices.ndimension() == 3:
                num_batch, n_data, n_interp = interp_indices.size()
                interp_indices = interp_indices.contiguous().view(-1)
                interp_values = interp_values.contiguous().view(-1, 1)

                if rhs.ndimension() == 3:
                    if rhs.size(0) == 1 and interp_indices.size(0) > 1:
                        rhs = rhs.expand(interp_indices.size(0), rhs.size(1), rhs.size(2))
                    batch_indices = torch.arange(0, num_batch, dtype=torch.long, device=rhs.device).unsqueeze_(1)
                    batch_indices = batch_indices.repeat(1, n_data * n_interp).view(-1)
                    res = rhs[batch_indices, interp_indices, :] * interp_values
                else:
                    res = rhs[interp_indices, :].unsqueeze(0) * interp_values
                res = res.view(num_batch, n_data, n_interp, -1)
                res = res.sum(-2)
                return res
            else:
                n_data, n_interp = interp_indices.size()
                interp_indices = interp_indices.contiguous().view(-1)
                interp_values = interp_values.contiguous().view(-1, 1)
                if rhs.ndimension() == 3:
                    num_batch, _, num_cols = rhs.size()
                    rhs = rhs.transpose(0, 1).contiguous().view(-1, num_batch * num_cols)
                    res = rhs[interp_indices, :] * interp_values
                    res = res.view(n_data, n_interp, num_batch, num_cols)
                    res = res.sum(-2).transpose(0, 1).contiguous()
                else:
                    res = rhs[interp_indices, :] * interp_values
                    res = res.view(n_data, n_interp, -1)
                    res = res.sum(-2)
                return res

        # Special non-cuda version -- this is faster on the CPU
        else:
            interp_size = list(interp_indices.size()) + [rhs.size(-1)]
            rhs_size = deepcopy(interp_size)
            rhs_size[-3] = rhs.size()[-2]
            interp_indices_expanded = interp_indices.unsqueeze(-1).expand(*interp_size)
            res = rhs.unsqueeze(-2).expand(*rhs_size).gather(-3, interp_indices_expanded)
            res = res.mul(interp_values.unsqueeze(-1).expand(interp_size))
            return res.sum(-2)


def left_t_interp(interp_indices, interp_values, rhs, output_dim):
    """
    """
    from .. import dsmm

    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    is_batch = rhs.ndimension() == 3
    if not is_batch:
        rhs = rhs.unsqueeze(0)
    if not interp_indices.ndimension() == 3:
        interp_indices = interp_indices.unsqueeze(0)
        interp_values = interp_values.unsqueeze(0)

    batch_size, n_data, n_interp = interp_values.size()
    _, _, num_cols = rhs.size()

    values = (rhs.unsqueeze(-2) * interp_values.unsqueeze(-1)).view(batch_size, n_data * n_interp, num_cols)

    flat_interp_indices = interp_indices.contiguous().view(1, -1)
    batch_indices = torch.arange(0, batch_size, dtype=torch.long, device=values.device).unsqueeze_(1)
    batch_indices = batch_indices.repeat(1, n_data * n_interp).view(1, -1)
    column_indices = torch.arange(0, n_data * n_interp, dtype=torch.long, device=values.device).unsqueeze_(1)
    column_indices = column_indices.repeat(batch_size, 1).view(1, -1)

    summing_matrix_indices = torch.cat([batch_indices, flat_interp_indices, column_indices])
    summing_matrix_values = torch.ones(
        batch_size * n_data * n_interp, dtype=interp_values.dtype, device=interp_values.device
    )
    size = torch.Size((batch_size, output_dim, n_data * n_interp))

    type_name = summing_matrix_values.type().split(".")[-1]  # e.g. FloatTensor
    if interp_values.is_cuda:
        cls = getattr(torch.cuda.sparse, type_name)
    else:
        cls = getattr(torch.sparse, type_name)
    summing_matrix = cls(summing_matrix_indices, summing_matrix_values, size)

    res = dsmm(summing_matrix, values)

    if not is_batch:
        res = res.squeeze(0)
    if is_vector:
        res = res.squeeze(-1)
    return res
