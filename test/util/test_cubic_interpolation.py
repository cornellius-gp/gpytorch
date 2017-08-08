import torch
from gpytorch.utils.interpolation import Interpolation
from gpytorch import utils


def test_interpolation():
    x = torch.linspace(0.01, 1, 100)
    grid = torch.linspace(-0.05, 1.05, 50)
    J, C = Interpolation().interpolate(grid, x)
    W = utils.toeplitz.index_coef_to_sparse(J, C, len(grid))
    test_func_grid = grid.pow(2)
    test_func_x = x.pow(2)

    interp_func_x = torch.dsmm(W, test_func_grid.unsqueeze(1)).squeeze()

    assert all(torch.abs(interp_func_x - test_func_x) / (test_func_x + 1e-10) < 1e-5)
