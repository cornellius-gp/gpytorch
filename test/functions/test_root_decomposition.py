from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import NonLazyVariable
from gpytorch.utils import approx_equal


class TestRootDecomposition(unittest.TestCase):

    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        mat = torch.Tensor([
            [5.0212, 0.5504, -0.1810, 1.5414, 2.9611],
            [0.5504, 2.8000, 1.9944, 0.6208, -0.8902],
            [-0.1810, 1.9944, 3.0505, 1.0790, -1.1774],
            [1.5414, 0.6208, 1.0790, 2.9430, 0.4170],
            [2.9611, -0.8902, -1.1774, 0.4170, 3.3208],
        ])
        self.mat_var = Variable(mat, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_root_decomposition(self):
        # Forward
        root = NonLazyVariable(self.mat_var).root_decomposition()
        res = root.matmul(root.transpose(-1, -2))
        self.assertTrue(approx_equal(res.data, self.mat_var.data))

        # Backward
        res.trace().backward()
        self.mat_var_clone.trace().backward()
        self.assertTrue(approx_equal(self.mat_var.grad.data, self.mat_var_clone.grad.data))

    def test_root_inv_decomposition(self):
        # Forward
        root = NonLazyVariable(self.mat_var).root_inv_decomposition()
        res = root.matmul(root.transpose(-1, -2))
        actual = self.mat_var_clone.inverse()
        self.assertTrue(approx_equal(res.data, actual.data))

        # Backward
        res.trace().backward()
        actual.trace().backward()
        self.assertTrue(approx_equal(self.mat_var.grad.data, self.mat_var_clone.grad.data))


if __name__ == '__main__':
    unittest.main()
