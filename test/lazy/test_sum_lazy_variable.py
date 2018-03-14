import os
import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable


def make_sum_lazy_var():
    c1 = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    t1 = ToeplitzLazyVariable(c1)
    c2 = Variable(torch.Tensor([6, 0, 1, -1]), requires_grad=True)
    t2 = ToeplitzLazyVariable(c2)
    return t1 + t2


class TestSumLazyVariable(unittest.TestCase):
    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        self.t1, self.t2 = make_sum_lazy_var().lazy_vars
        self.t1_eval = self.t1.evaluate().data
        self.t2_eval = self.t2.evaluate().data

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def test_add_diag(self):
        diag = Variable(torch.Tensor([4]))
        lazy_var = make_sum_lazy_var().add_diag(diag)
        self.assertTrue(torch.equal(
            lazy_var.evaluate().data,
            (self.t1_eval + self.t2_eval + torch.eye(4) * 4),
        ))

    def test_add_jitter(self):
        lazy_var = make_sum_lazy_var().add_jitter()
        self.assertLess(
            torch.max(torch.abs(
                lazy_var.evaluate().data - (self.t1_eval + self.t2_eval)
            )),
            1e-1,
        )

    def test_inv_matmul(self):
        mat = torch.randn(4, 4)
        res = make_sum_lazy_var().inv_matmul(Variable(mat))
        self.assertLess(
            torch.norm(
                res.data - (self.t1_eval + self.t2_eval).inverse().matmul(mat)
            ),
            1e-3,
        )

    def test_getitem(self):
        res = make_sum_lazy_var()[1, 1]
        self.assertLess(
            torch.norm(res.data - (self.t1_eval + self.t2_eval)[1, 1]),
            1e-3,
        )


if __name__ == '__main__':
    unittest.main()
