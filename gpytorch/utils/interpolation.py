from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable


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
        res = Variable(U.data.new(U.size()).zero_())

        U_lt_1 = (1 - U.floor().clamp(0, 1))  # U, if U < 1, 0 otherwise
        res = res + (((1.5 * U - 2.5).mul(U)).mul(U) + 1) * U_lt_1

        # u(s) = -0.5|s|^3 + 2.5|s|^2 - 4|s| + 2 when 1 < |s| < 2
        U_ge_1_le_2 = 1 - U_lt_1  # U, if U <= 1 <= 2, 0 otherwise
        res = res + (((-0.5 * U + 2.5).mul(U) - 4).mul(U) + 2) * U_ge_1_le_2
        return res

    def interpolate(self, x_grid, x_target, interp_points=range(-2, 2)):
        # Do some boundary checking
        grid_mins = x_grid.min(1)[0]
        grid_maxs = x_grid.max(1)[0]
        x_target_min = x_target.min(0)[0]
        x_target_max = x_target.min(0)[0]
        lt_min_mask = ((x_target_min - grid_mins).lt(-1e-7))
        gt_max_mask = ((x_target_max - grid_maxs).gt(1e-7))
        if lt_min_mask.data.sum():
            first_out_of_range = lt_min_mask.nonzero().squeeze(1)[0]
            raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                Grid bounds were ({}, {}), but min = {}, \
                                max = {}'.format(self.grid_mins[first_out_of_range],
                                                 self.grid_maxs[first_out_of_range],
                                                 x_target_min[first_out_of_range],
                                                 x_target_max[first_out_of_range]))
        if gt_max_mask.data.sum():
            first_out_of_range = gt_max_mask.nonzero().squeeze(1)[0]
            raise RuntimeError('Received data that was out of bounds for the specified grid. \
                                Grid bounds were ({}, {}), but min = {}, \
                                max = {}'.format(self.grid_mins[first_out_of_range],
                                                 self.grid_maxs[first_out_of_range],
                                                 x_target_min[first_out_of_range],
                                                 x_target_max[first_out_of_range]))

        # Now do interpolation
        interp_points_flip = Variable(x_grid.data.new(interp_points[::-1]))
        interp_points = Variable(x_grid.data.new(interp_points))

        num_grid_points = x_grid.size(1)
        num_target_points = x_target.size(0)
        num_dim = x_target.size(-1)
        num_coefficients = len(interp_points)

        interp_values = Variable(x_target.data.new(num_target_points, num_coefficients ** num_dim).fill_(1))
        interp_indices = Variable(x_grid.data.new(num_target_points, num_coefficients ** num_dim).long().zero_())

        for i in range(num_dim):
            grid_delta = x_grid[i, 1] - x_grid[i, 0]
            lower_grid_pt_idxs = torch.floor((x_target[:, i] - x_grid[i, 0]) / grid_delta).squeeze()
            lower_pt_rel_dists = (x_target[:, i] - x_grid[i, 0]) / grid_delta - lower_grid_pt_idxs
            lower_grid_pt_idxs = lower_grid_pt_idxs - interp_points.max()

            scaled_dist = lower_pt_rel_dists.unsqueeze(-1) + interp_points_flip.unsqueeze(-2)
            dim_interp_values = self._cubic_interpolation_kernel(scaled_dist)

            # Find points who's closest lower grid point is the first grid point
            # This corresponds to a boundary condition that we must fix manually.
            left_boundary_pts = torch.nonzero(lower_grid_pt_idxs < 1)
            num_left = len(left_boundary_pts)

            if num_left > 0:
                left_boundary_pts.squeeze_(1)
                x_grid_first = x_grid[i, :num_coefficients].unsqueeze(1).t().expand(num_left, num_coefficients)

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
                x_grid_last = x_grid[i, -num_coefficients:].unsqueeze(1).t().expand(num_right, num_coefficients)

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
