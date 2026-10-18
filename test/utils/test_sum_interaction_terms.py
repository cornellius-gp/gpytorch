#!/usr/bin/env python3

import unittest
from functools import reduce
from itertools import combinations
from operator import mul

import torch
from linear_operator import to_dense, to_linear_operator

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase


def prod(iterable):
    return reduce(mul, iterable, 1)


class TestSumInteractionTerms(BaseTestCase, unittest.TestCase):
    def test_sum_interaction_terms(self):
        batch_shape = torch.Size([2, 1])
        D = 5
        M = 4
        N = 20

        base_kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([D]))
        x = torch.randn(*batch_shape, D, N, 1)
        with torch.no_grad(), gpytorch.settings.lazily_evaluate_kernels(False):
            covars = base_kernel(x)

        actual = torch.zeros(*batch_shape, N, N)
        for degree in range(1, M + 1):
            for interaction_term_indices in combinations(range(D), degree):
                actual = actual + prod([to_dense(covars[..., i, :, :]) for i in interaction_term_indices])

        res = gpytorch.utils.sum_interaction_terms(covars, max_degree=M)
        self.assertAllClose(res, actual)

    def test_sum_interaction_terms_default_degree(self) -> None:
        covars = torch.tensor(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[1.0, 0.25], [0.25, 1.0]],
                [[1.0, 0.75], [0.75, 1.0]],
            ],
            dtype=torch.float64,
        )
        covars = torch.stack((covars, covars * 0.5))
        expected = (1 + covars).prod(dim=-3) - 1

        for dim in (-3, -4):
            for linear_operator in (False, True):
                with self.subTest(dim=dim, linear_operator=linear_operator):
                    inputs = covars.movedim(-3, dim)
                    if linear_operator:
                        inputs = to_linear_operator(inputs)
                    self.assertAllClose(
                        gpytorch.utils.sum_interaction_terms(inputs, dim=dim), expected, rtol=1e-12, atol=1e-12
                    )
                    self.assertAllClose(
                        gpytorch.utils.sum_interaction_terms(inputs, max_degree=None, dim=dim),
                        expected,
                        rtol=1e-12,
                        atol=1e-12,
                    )
