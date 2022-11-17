import unittest

import torch
from gpytorch.vecchia.old_blocking import Block

square_grid_data_2d = torch.cartesian_prod(torch.linspace(1, 10, 10), torch.linspace(1, 10, 10))

class Test2DSquareGridNBlocks0Neighbors(unittest.TestCase):
    def test_something(self):
        self.assertEqual(square_grid_data_2d.shape, square_grid_data_2d.shape)  # add assertion here


class Test2DSquareGridNBlocks3Neighbors(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


class Test2DSquareGrid5Blocks0Neighbors(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


class Test2DSquareGrid5Blocks1Neighbors(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
