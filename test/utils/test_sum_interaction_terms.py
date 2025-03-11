#!/usr/bin/env python3

import unittest
from functools import reduce
from itertools import combinations
from operator import mul

import torch
from linear_operator import to_dense

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
