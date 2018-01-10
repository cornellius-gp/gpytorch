import torch


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

        first_case = U[U <= 1]
        # u(s) = 1.5|s|^3 - 2.5|s|^2 + 1 when 0 < |s| < 1
        U[U <= 1] = ((1.5 * first_case - 2.5).mul(first_case)).mul(first_case) + 1

        # u(s) = -0.5|s|^3 + 2.5|s|^2 - 4|s| + 2 when 1 < |s| < 2
        second_case = U[(1 < U) & (U <= 2)]
        U[(1 < U) & (U <= 2)] = ((-0.5 * second_case + 2.5).mul(second_case) - 4).mul(second_case) + 2
        return U

    def interpolate(self, x_grid, x_target, interp_points=range(-2, 2)):
        interp_points_flip = x_grid.new(interp_points[::-1])
        interp_points = x_grid.new(interp_points)

        num_grid_points = x_grid.size(1)
        num_target_points = x_target.size(0)
        num_dim = x_target.size(-1)
        num_coefficients = len(interp_points)

        interp_values = x_target.new(num_dim, num_target_points, num_coefficients).zero_()
        interp_indices = x_grid.new(num_dim, num_target_points, num_coefficients).long().zero_()

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

            interp_indices[i].copy_(dim_interp_indices)
            interp_values[i].copy_(dim_interp_values)

        return interp_indices, interp_values
