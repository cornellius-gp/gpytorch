from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
from gpytorch.lazy import ToeplitzLazyTensor


def make_sum_lazy_var():
    c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
    t1 = ToeplitzLazyTensor(c1)
    c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
    t2 = ToeplitzLazyTensor(c2)
    return t1 + t2


class TestSumLazyTensor(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        self.t1, self.t2 = make_sum_lazy_var().lazy_vars
        self.t1_eval = self.t1.evaluate()
        self.t2_eval = self.t2.evaluate()

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_add_diag(self):
        diag = torch.tensor(4.)
        lazy_var = make_sum_lazy_var().add_diag(diag)
        self.assertTrue(torch.equal(lazy_var.evaluate(), (self.t1_eval + self.t2_eval + torch.eye(4) * 4)))

    def test_add_jitter(self):
        lazy_var = make_sum_lazy_var().add_jitter()
        self.assertLess(torch.max(torch.abs(lazy_var.evaluate() - (self.t1_eval + self.t2_eval))), 1e-1)

    def test_inv_matmul(self):
        mat = torch.randn(4, 4)
        res = make_sum_lazy_var().inv_matmul(mat)
        self.assertLess(torch.norm(res - (self.t1_eval + self.t2_eval).inverse().matmul(mat)), 1e-3)

    def test_getitem(self):
        res = make_sum_lazy_var()[1, 1]
        self.assertLess(torch.norm(res - (self.t1_eval + self.t2_eval)[1, 1]), 1e-3)


if __name__ == "__main__":
    unittest.main()
