from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import KroneckerProductLazyVariable, NonLazyVariable
from gpytorch.utils import approx_equal


a = torch.Tensor([[4, 0, 2], [0, 1, -1], [2, -1, 3]])
b = torch.Tensor([[2, 1], [1, 2]])
c = torch.Tensor([[4, 0, 1, 0], [0, 1, -1, 0], [1, -1, 3, 0], [0, 0, 0, 1]])


def kron(a, b):
    res = []
    if b.ndimension() == 2:
        for i in range(b.size(0)):
            row_res = []
            for j in range(b.size(1)):
                row_res.append(a * b[i, j])
            res.append(torch.cat(row_res, 1))
        return torch.cat(res, 0)
    else:
        for i in range(b.size(1)):
            row_res = []
            for j in range(b.size(2)):
                row_res.append(a * b[:, i, j].unsqueeze(1).unsqueeze(2))
            res.append(torch.cat(row_res, 2))
        return torch.cat(res, 1)


class TestKroneckerProductLazyVariable(unittest.TestCase):
    def test_matmul_vec(self):
        avar = Variable(a, requires_grad=True)
        bvar = Variable(b, requires_grad=True)
        cvar = Variable(c, requires_grad=True)
        vec = Variable(torch.randn(24), requires_grad=True)
        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(avar), NonLazyVariable(bvar), NonLazyVariable(cvar))
        res = kp_lazy_var.matmul(vec)

        avar_copy = Variable(a, requires_grad=True)
        bvar_copy = Variable(b, requires_grad=True)
        cvar_copy = Variable(c, requires_grad=True)
        vec_copy = Variable(vec.data.clone(), requires_grad=True)
        actual = kron(kron(avar_copy, bvar_copy), cvar_copy).matmul(vec_copy)

        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(avar_copy.grad.data, avar.grad.data))
        self.assertTrue(approx_equal(bvar_copy.grad.data, bvar.grad.data))
        self.assertTrue(approx_equal(cvar_copy.grad.data, cvar.grad.data))
        self.assertTrue(approx_equal(vec_copy.grad.data, vec.grad.data))

    def test_matmul_vec_new(self):
        ax = torch.randn(4, 2, 3)
        bx = torch.randn(4, 5, 2)
        cx = torch.randn(4, 6, 4)
        rhsx = torch.randn(4, 3 * 2 * 4, 1)
        rhsx = rhsx / torch.norm(rhsx)
        ax_copy = Variable(ax, requires_grad=True)
        bx_copy = bx.clone()
        cx_copy = cx.clone()
        rhsx_copy = rhsx.clone()

        ax.requires_grad = True
        bx.requires_grad = True
        cx.requires_grad = True
        ax_copy.requires_grad = True
        bx_copy.requires_grad = True
        cx_copy.requires_grad = True
        rhsx.requires_grad = True
        rhsx_copy.requires_grad = True

        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(ax), NonLazyVariable(bx), NonLazyVariable(cx))
        res = kp_lazy_var.matmul(rhsx)

        actual_mat = kron(kron(ax_copy, bx_copy), cx_copy)
        actual_eval = kp_lazy_var.evaluate()
        actual = actual_mat.matmul(rhsx_copy)

        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(ax_copy.grad.data, ax.grad.data))
        self.assertTrue(approx_equal(bx_copy.grad.data, bx.grad.data))
        self.assertTrue(approx_equal(cx_copy.grad.data, cx.grad.data))
        self.assertTrue(approx_equal(rhsx_copy.grad.data, rhsx.grad.data))
    #
    # def test_matmul_mat_new(self):
    #     a = torch.randn(3, 3)
    #     b = torch.randn(2, 2)
    #     c = torch.randn(4, 4)
    #     a_copy = torch.tensor(a)
    #     b_copy = b.clone().detach()
    #     c_copy = c.clone().detach()
    #     rhs = torch.randn(3 * 2 * 4, 2)
    #     rhs_copy = rhs.clone().detach()
    #
    #     a.requires_grad = True
    #     b.requires_grad = True
    #     c.requires_grad = True
    #     a_copy.requires_grad = True
    #     b_copy.requires_grad = True
    #     c_copy.requires_grad = True
    #     rhs.requires_grad = True
    #     rhs_copy.requires_grad = True
    #
    #     actual = kron(kron(a_copy, b_copy), c_copy).matmul(rhs_copy)
    #     kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(a), NonLazyVariable(b), NonLazyVariable(c))
    #     res = kp_lazy_var.matmul(rhs)
    #
    #     self.assertTrue(approx_equal(res.data, actual.data))
    #
    #     actual.sum().backward()
    #     res.sum().backward()
    #     print(a_copy.grad, a.grad)
    #     self.assertTrue(approx_equal(a_copy.grad.data, a.grad.data))
    #     self.assertTrue(approx_equal(b_copy.grad.data, b.grad.data))
    #     self.assertTrue(approx_equal(c_copy.grad.data, c.grad.data))
    #     self.assertTrue(approx_equal(rhs_copy.grad.data, rhs.grad.data))

    def test_matmul_batch_mat(self):
        avar = Variable(a.repeat(3, 1, 1), requires_grad=True)
        bvar = Variable(b.repeat(3, 1, 1), requires_grad=True)
        cvar = Variable(c.repeat(3, 1, 1), requires_grad=True)
        mat = Variable(torch.randn(3, 24, 5), requires_grad=True)
        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(avar), NonLazyVariable(bvar), NonLazyVariable(cvar))
        res = kp_lazy_var.matmul(mat)

        avar_copy = Variable(a.repeat(3, 1, 1), requires_grad=True)
        bvar_copy = Variable(b.repeat(3, 1, 1), requires_grad=True)
        cvar_copy = Variable(c.repeat(3, 1, 1), requires_grad=True)
        mat_copy = Variable(mat.data.clone(), requires_grad=True)
        actual = kron(kron(avar_copy, bvar_copy), cvar_copy).matmul(mat_copy)
        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(avar_copy.grad.data, avar.grad.data))
        self.assertTrue(approx_equal(bvar_copy.grad.data, bvar.grad.data))
        self.assertTrue(approx_equal(cvar_copy.grad.data, cvar.grad.data))
        self.assertTrue(approx_equal(mat_copy.grad.data, mat.grad.data))

    def test_evaluate(self):
        avar = Variable(a)
        bvar = Variable(b)
        cvar = Variable(c)
        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(avar), NonLazyVariable(bvar), NonLazyVariable(cvar))
        res = kp_lazy_var.evaluate()
        actual = kron(kron(avar, bvar), cvar)
        self.assertTrue(approx_equal(res.data, actual.data))

        avar = Variable(a.repeat(3, 1, 1))
        bvar = Variable(b.repeat(3, 1, 1))
        cvar = Variable(c.repeat(3, 1, 1))
        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(avar), NonLazyVariable(bvar), NonLazyVariable(cvar))
        res = kp_lazy_var.evaluate()
        actual = kron(kron(avar, bvar), cvar)
        self.assertTrue(approx_equal(res.data, actual.data))

    def test_diag(self):
        avar = Variable(a)
        bvar = Variable(b)
        cvar = Variable(c)
        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(avar), NonLazyVariable(bvar), NonLazyVariable(cvar))
        res = kp_lazy_var.diag()
        actual = kron(kron(avar, bvar), cvar).diag()
        self.assertTrue(approx_equal(res.data, actual.data))

        avar = Variable(a.repeat(3, 1, 1))
        bvar = Variable(b.repeat(3, 1, 1))
        cvar = Variable(c.repeat(3, 1, 1))
        kp_lazy_var = KroneckerProductLazyVariable(NonLazyVariable(avar), NonLazyVariable(bvar), NonLazyVariable(cvar))
        res = kp_lazy_var.diag()
        actual_mat = kron(kron(avar, bvar), cvar)
        actual = torch.stack([actual_mat[0].diag(), actual_mat[1].diag(), actual_mat[2].diag()])
        self.assertTrue(approx_equal(res.data, actual.data))


if __name__ == "__main__":
    unittest.main()
