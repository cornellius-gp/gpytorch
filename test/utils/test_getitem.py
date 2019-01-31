#!/usr/bin/env python3

import torch
import unittest
from itertools import product
from gpytorch.utils.getitem import _compute_getitem_size


class TestGetitem(unittest.TestCase):
    def test_compute_getitem_size(self):
        a = torch.tensor(0.).expand(5, 5, 5, 5, 5)

        for indices in product(
            [torch.tensor([0, 1, 1, 0]), slice(None, None, None), 1, slice(0, 2, None)], repeat=5
        ):
            res = _compute_getitem_size(a, indices)
            actual = a[indices].shape
            self.assertEqual(res, actual)
