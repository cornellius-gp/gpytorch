from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import gpytorch
import unittest
from gpytorch.lazy import ToeplitzLazyTensor


def make_sum_lazy_var():
    c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
    t1 = ToeplitzLazyTensor(c1)
    c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
    t2 = ToeplitzLazyTensor(c2)
    return t1 + t2


class TestPsdSumLazyTensor(unittest.TestCase):
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

    def test_sample(self):
        res = make_sum_lazy_var()
        actual = res.evaluate()

        with gpytorch.settings.max_root_decomposition_size(1000):
            samples = res.zero_mean_mvn_samples(10000)
            sample_covar = samples.unsqueeze(-1).matmul(samples.unsqueeze(-2)).mean(0)
        self.assertLess(((sample_covar - actual).abs() / actual.abs().clamp(1, 1e5)).max().item(), 4e-1)


if __name__ == "__main__":
    unittest.main()
