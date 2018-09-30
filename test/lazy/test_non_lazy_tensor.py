from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
import os
import random
from gpytorch.lazy import NonLazyTensor
from test._utils import approx_equal


class TestNonLazyTensor(unittest.TestCase):
    """
    This test is mostly here for testing the default LazyTensor methods
    """

    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        self.mat = torch.randn(5, 5)
        self.mat = self.mat.t().matmul(self.mat)
        self.mat.requires_grad = True

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_add_diag_single_element(self):
        diag = torch.tensor(1.5)
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + torch.eye(self.mat.size(-1)).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + torch.eye(self.mat.size(-1)).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

    def test_add_diag_different_elements_on_diagonal(self):
        diag = torch.tensor([1.5, 1.3, 1.2, 1.1, 2.])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + diag.diag()
        self.assertTrue(approx_equal(res, actual))


class TestNonLazyTensorBatch(unittest.TestCase):
    """
    This test is mostly here for testing the default LazyTensor methods
    """

    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        self.mat = torch.randn(3, 5, 5)
        self.mat = self.mat.transpose(-1, -2).matmul(self.mat)
        self.mat.requires_grad = True

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_add_diag_single_element(self):
        diag = torch.tensor(1.5)
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + torch.eye(self.mat.size(-1)).unsqueeze(0).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([1.5])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + torch.eye(self.mat.size(-1)).unsqueeze(0).mul(1.5)
        self.assertTrue(approx_equal(res, actual))

    def test_add_diag_different_elements_on_diagonal(self):
        diag = torch.tensor([1.5, 1.3, 1.2, 1.1, 2.])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + diag.diag().unsqueeze(0)
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([[1.5, 1.3, 1.2, 1.1, 2.]])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + diag[0].diag().unsqueeze(0)
        self.assertTrue(approx_equal(res, actual))

    def test_add_diag_different_batch(self):
        diag = torch.tensor([[1.5, 1.3, 1.2, 1.1, 2.], [0.1, 0.2, 0.3, 0.4, 0.], [0., 0.1, 1.3, 1.4, 0.]])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + torch.cat(
            [diag[0].diag().unsqueeze(0), diag[1].diag().unsqueeze(0), diag[2].diag().unsqueeze(0)]
        )
        self.assertTrue(approx_equal(res, actual))

        diag = torch.tensor([[1.5], [1.3], [0.1]])
        res = NonLazyTensor(self.mat).add_diag(diag).evaluate()
        actual = self.mat + torch.eye(5).unsqueeze(0) * diag.unsqueeze(-1)
        self.assertTrue(approx_equal(res, actual))
