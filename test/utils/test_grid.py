from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
import unittest


class TestGrid(unittest.TestCase):
    def test_scale_to_bounds(self):
        """
        """
        x = torch.randn(100) * 50
        res = gpytorch.utils.grid.scale_to_bounds(x, -1, 1)
        self.assertGreater(res.min().item(), -1)
        self.assertLess(res.max().item(), 1)

    def test_choose_grid_size(self):
        """
        """
        x = torch.randn(100)
        grid_size = gpytorch.utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 200)

        x = torch.randn(100, 4)
        grid_size = gpytorch.utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 200)

        x = torch.randn(16, 100, 4)
        grid_size = gpytorch.utils.grid.choose_grid_size(x, ratio=2.0)
        self.assertEqual(grid_size, 200)
