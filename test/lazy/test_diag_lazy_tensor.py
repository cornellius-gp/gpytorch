from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import DiagLazyTensor


diag = torch.Tensor([1, 2, 3])


class TestDiagLazyTensor(unittest.TestCase):
    def test_evaluate(self):
        diag_lv = DiagLazyTensor(Variable(diag))
        self.assertTrue(torch.equal(diag_lv.evaluate().data, diag.diag()))

    def test_function_factory(self):
        # 1d
        diag_var1 = Variable(diag, requires_grad=True)
        diag_var2 = Variable(diag, requires_grad=True)
        test_mat = torch.Tensor([3, 4, 5])

        diag_lv = DiagLazyTensor(diag_var1)
        diag_ev = DiagLazyTensor(diag_var2).evaluate()

        # Forward
        res = diag_lv.matmul(Variable(test_mat))
        actual = torch.matmul(diag_ev, Variable(test_mat))
        self.assertLess(torch.norm(res.data - actual.data), 1e-4)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.norm(diag_var1.grad.data - diag_var2.grad.data), 1e-3)

        # 2d
        diag_var1 = Variable(diag, requires_grad=True)
        diag_var2 = Variable(diag, requires_grad=True)
        test_mat = torch.eye(3)

        diag_lv = DiagLazyTensor(diag_var1)
        diag_ev = DiagLazyTensor(diag_var2).evaluate()

        # Forward
        res = diag_lv.matmul(Variable(test_mat))
        actual = torch.matmul(diag_ev, Variable(test_mat))
        self.assertLess(torch.norm(res.data - actual.data), 1e-4)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.norm(diag_var1.grad.data - diag_var2.grad.data), 1e-3)

    def test_batch_function_factory(self):
        # 2d
        diag_var1 = Variable(diag.repeat(5, 1), requires_grad=True)
        diag_var2 = Variable(diag.repeat(5, 1), requires_grad=True)
        test_mat = torch.eye(3).repeat(5, 1, 1)

        diag_lv = DiagLazyTensor(diag_var1)
        diag_ev = DiagLazyTensor(diag_var2).evaluate()

        # Forward
        res = diag_lv.matmul(Variable(test_mat))
        actual = torch.matmul(diag_ev, Variable(test_mat))
        self.assertLess(torch.norm(res.data - actual.data), 1e-4)

        # Backward
        res.sum().backward()
        actual.sum().backward()
        self.assertLess(torch.norm(diag_var1.grad.data - diag_var2.grad.data), 1e-3)

    def test_get_item(self):
        diag_lv = DiagLazyTensor(Variable(diag))
        diag_ev = diag_lv.evaluate()
        self.assertTrue(torch.equal(diag_lv[0:2].evaluate().data, diag_ev[0:2].data))


if __name__ == "__main__":
    unittest.main()
