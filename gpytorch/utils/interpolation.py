import pdb
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

    def interpolate(self, x_grid, x_target):
        interp_points = range(-2,2)
        num_grid_points = len(x_grid)
        num_target_points = len(x_target)
        num_coefficients = len(interp_points)

        grid_delta = x_grid[1] - x_grid[0]

        lower_grid_pt_idxs = torch.floor((x_target - x_grid[0]) / grid_delta)
        lower_pt_rel_dists = (x_target - x_grid[0]) / grid_delta - lower_grid_pt_idxs
        lower_grid_pt_idxs = lower_grid_pt_idxs - interp_points[-1]
        C = torch.zeros(num_target_points,num_coefficients)

        for i in range(num_coefficients):
            scaled_dist = lower_pt_rel_dists + interp_points[-i-1]
            C[:,i] = self._cubic_interpolation_kernel(scaled_dist)

        # Find points who's closest lower grid point is the first grid point
        # This corresponds to a boundary condition that we must fix manually.
        left_boundary_pts = torch.LongTensor([i for i in range(len(lower_grid_pt_idxs)) if (lower_grid_pt_idxs < 1)[i] == 1])
        num_left = len(left_boundary_pts)

        if num_left > 0:
            x_grid_first = x_grid[:num_coefficients].unsqueeze(1).t().expand(num_left,num_coefficients)

            dists = torch.abs(x_grid_first - x_target[left_boundary_pts].unsqueeze(1).expand(num_left,num_coefficients))
            closest_from_first = torch.min(dists,1)[1].squeeze()

            for i in range(num_left):
                C[left_boundary_pts[i],:] = 0
                C[left_boundary_pts[i],closest_from_first[i]] = 1
                lower_grid_pt_idxs[left_boundary_pts[i]] = 0

        right_boundary_pts = torch.LongTensor([i for i in range(len(lower_grid_pt_idxs)) if (lower_grid_pt_idxs > num_grid_points - num_coefficients)[i] == 1])
        num_right = len(right_boundary_pts)

        if num_right > 0:
            x_grid_last = x_grid[-num_coefficients:].unsqueeze(1).t().expand(num_right,num_coefficients)

            dists = torch.abs(x_grid_last - x_target[right_boundary_pts].unsqueeze(1).expand(num_right,num_coefficients))
            closest_from_last = torch.min(dists,1)[1].squeeze()

            for i in range(num_right):
                C[right_boundary_pts[i],:] = 0
                C[right_boundary_pts[i],closest_from_last[i]] = 1
                lower_grid_pt_idxs[right_boundary_pts[i]] = num_grid_points - num_coefficients

        J = torch.zeros(num_target_points,num_coefficients)
        for i in range(num_coefficients):
            J[:,i] = lower_grid_pt_idxs + i

        J = J.long()
        J_list = [[],[]]
        value_list = []
        for i in range(num_target_points):
            for j in range(num_coefficients):
                if C[i,j] == 0:
                    continue
                J_list[0].append(i)
                J_list[1].append(J[i,j])
                value_list.append(C[i,j])

        index_tensor = torch.LongTensor(J_list)
        value_tensor = torch.FloatTensor(value_list)
        W = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([num_target_points,num_grid_points]))

        return W
