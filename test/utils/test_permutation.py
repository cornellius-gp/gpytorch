#!/usr/bin/env python3

import unittest

import torch

from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.utils.permutation import apply_permutation, inverse_permutation


class TestPermutationHelpers(BaseTestCase, unittest.TestCase):
    seed = 0

    def _gen_test_psd(self):
        return torch.tensor([[[0.25, -0.75], [-0.75, 2.25]], [[1.0, 1.2], [1.2, 0.5]]])

    def test_apply_permutation_left_and_right(self):
        A = self._gen_test_psd()
        left_permutation = torch.tensor([[0, 1], [1, 0]])
        right_permutation = torch.tensor([1, 0])
        res = apply_permutation(A, left_permutation, right_permutation)
        self.assertAllClose(res, torch.tensor([[[-0.75, 0.25], [2.25, -0.75]], [[0.5, 1.2], [1.2, 1.0]]]))

    def test_apply_permutation_left_partial_and_right(self):
        A = self._gen_test_psd()
        left_permutation = torch.tensor([[0], [1]])
        right_permutation = torch.tensor([1, 0])
        res = apply_permutation(A, left_permutation, right_permutation)
        self.assertAllClose(res, torch.tensor([[[-0.75, 0.25]], [[0.5, 1.2]]]))

    def test_apply_permutation_left_only(self):
        A = self._gen_test_psd()
        left_permutation = torch.tensor([[0, 1], [1, 0]])
        res = apply_permutation(A, left_permutation=left_permutation)
        self.assertAllClose(res, torch.tensor([[[0.25, -0.75], [-0.75, 2.25]], [[1.2, 0.5], [1.0, 1.2]]]))

    def test_apply_permutation_right_only(self):
        A = self._gen_test_psd()
        right_permutation = torch.tensor([1, 0])
        res = apply_permutation(A, right_permutation=right_permutation)
        self.assertAllClose(res, torch.tensor([[[-0.75, 0.25], [2.25, -0.75]], [[1.2, 1.0], [0.5, 1.2]]]))

    def test_inverse_permutation(self):
        permutation = torch.tensor([[2, 3, 4, 0, 1], [4, 3, 0, 2, 1]])
        res = inverse_permutation(permutation)
        self.assertAllClose(res, torch.tensor([[3, 4, 0, 1, 2], [2, 4, 3, 1, 0]]))


if __name__ == "__main__":
    unittest.main()
