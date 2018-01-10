import torch
from gpytorch.utils.interpolation import Interpolation
from gpytorch import utils


def test_interpolation():
    x = torch.linspace(0.01, 1, 100).unsqueeze(1)
    grid = torch.linspace(-0.05, 1.05, 50).unsqueeze(0)
    indices, values = Interpolation().interpolate(grid, x)
    indices.squeeze_(0)
    values.squeeze_(0)
    test_func_grid = grid.squeeze(0).pow(2)
    test_func_x = x.pow(2).squeeze(-1)

    interp_func_x = utils.left_interp(indices, values, test_func_grid.unsqueeze(1)).squeeze()

    assert utils.approx_equal(interp_func_x, test_func_x)
