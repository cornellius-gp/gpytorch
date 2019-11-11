#!/usr/bin/env python3

import unittest
from itertools import product

import torch

from gpytorch.utils.getitem import _compute_getitem_size, _convert_indices_to_tensors


class TestGetitem(unittest.TestCase):
    def test_compute_getitem_size(self):
        a = torch.tensor(0.0).expand(5, 5, 5, 5, 5)

        for indices in product([torch.tensor([0, 1, 1, 0]), slice(None, None, None), 1, slice(0, 2, None)], repeat=5):
            res = _compute_getitem_size(a, indices)
            actual = a[indices].shape
            self.assertEqual(res, actual)

    def test_convert_indices_to_tensors(self):
        a = torch.randn(5, 5, 5, 5, 5)

        for indices in product([torch.tensor([0, 1, 1, 0]), slice(None, None, None), 1, slice(0, 2, None)], repeat=5):
            if not any(torch.is_tensor(index) for index in indices):
                continue
            new_indices = _convert_indices_to_tensors(a, indices)
            self.assertTrue(all(torch.is_tensor(index) for index in new_indices))
            self.assertTrue(torch.equal(a[indices], a[new_indices]))
