from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import KroneckerProductLazyTensor, NonLazyTensor
from gpytorch.utils import approx_equal


a = torch.tensor([[4, 0, 2], [0, 1, -1], [2, -1, 3]], dtype=torch.float)
b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
c = torch.tensor([[4, 0, 1, 0], [0, 1, -1, 0], [1, -1, 3, 0], [0, 0, 0, 1]], dtype=torch.float)


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


class TestKroneckerProductLazyTensor(unittest.TestCase):
    def test_matmul_vec(self):
        avar = torch.tensor(a, requires_grad=True)
        bvar = torch.tensor(b, requires_grad=True)
        cvar = torch.tensor(c, requires_grad=True)
        vec = torch.tensor(torch.randn(24), requires_grad=True)
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(avar), NonLazyTensor(bvar), NonLazyTensor(cvar))
        res = kp_lazy_var.matmul(vec)

        avar_copy = torch.tensor(a, requires_grad=True)
        bvar_copy = torch.tensor(b, requires_grad=True)
        cvar_copy = torch.tensor(c, requires_grad=True)
        vec_copy = torch.tensor(vec.detach(), requires_grad=True)
        actual = kron(kron(avar_copy, bvar_copy), cvar_copy).matmul(vec_copy)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(avar_copy.grad, avar.grad))
        self.assertTrue(approx_equal(bvar_copy.grad, bvar.grad))
        self.assertTrue(approx_equal(cvar_copy.grad, cvar.grad))
        self.assertTrue(approx_equal(vec_copy.grad, vec.grad))

    def test_matmul_vec_random_rectangular_nonbatch(self):
        ax = torch.randn(2, 3)
        bx = torch.randn(5, 2)
        cx = torch.randn(6, 4)
        rhsx = torch.randn(3 * 2 * 4, 1)
        rhsx = rhsx / torch.norm(rhsx)
        ax_copy = torch.tensor(ax, requires_grad=True)
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

        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(ax), NonLazyTensor(bx), NonLazyTensor(cx))
        res = kp_lazy_var.matmul(rhsx)

        actual_mat = kron(kron(ax_copy, bx_copy), cx_copy)
        actual = actual_mat.matmul(rhsx_copy)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(ax_copy.grad, ax.grad))
        self.assertTrue(approx_equal(bx_copy.grad, bx.grad))
        self.assertTrue(approx_equal(cx_copy.grad, cx.grad))
        self.assertTrue(approx_equal(rhsx_copy.grad, rhsx.grad))

    def test_matmul_vec_random_rectangular(self):
        ax = torch.randn(4, 2, 3)
        bx = torch.randn(4, 5, 2)
        cx = torch.randn(4, 6, 4)
        rhsx = torch.randn(4, 3 * 2 * 4, 1)
        rhsx = rhsx / torch.norm(rhsx)
        ax_copy = torch.tensor(ax, requires_grad=True)
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

        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(ax), NonLazyTensor(bx), NonLazyTensor(cx))
        res = kp_lazy_var.matmul(rhsx)

        actual_mat = kron(kron(ax_copy, bx_copy), cx_copy)
        actual = actual_mat.matmul(rhsx_copy)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(ax_copy.grad, ax.grad))
        self.assertTrue(approx_equal(bx_copy.grad, bx.grad))
        self.assertTrue(approx_equal(cx_copy.grad, cx.grad))
        self.assertTrue(approx_equal(rhsx_copy.grad, rhsx.grad))

    def test_matmul_mat_random_rectangular_nobatch(self):
        a = torch.randn(2, 3)
        b = torch.randn(5, 2)
        c = torch.randn(6, 4)
        rhs = torch.randn(3 * 2 * 4, 2)
        a_copy = torch.tensor(a)
        b_copy = b.clone()
        c_copy = c.clone()
        rhs_copy = rhs.clone()

        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        a_copy.requires_grad = True
        b_copy.requires_grad = True
        c_copy.requires_grad = True
        rhs.requires_grad = True
        rhs_copy.requires_grad = True

        actual = kron(kron(a_copy, b_copy), c_copy).matmul(rhs_copy)
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        res = kp_lazy_var.matmul(rhs)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(a_copy.grad, a.grad))
        self.assertTrue(approx_equal(b_copy.grad, b.grad))
        self.assertTrue(approx_equal(c_copy.grad, c.grad))
        self.assertTrue(approx_equal(rhs_copy.grad, rhs.grad))

    def test_matmul_mat_random_rectangular(self):
        a = torch.randn(4, 2, 3)
        b = torch.randn(4, 5, 2)
        c = torch.randn(4, 6, 4)
        rhs = torch.randn(4, 3 * 2 * 4, 2)
        a_copy = torch.tensor(a)
        b_copy = b.clone()
        c_copy = c.clone()
        rhs_copy = rhs.clone()

        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        a_copy.requires_grad = True
        b_copy.requires_grad = True
        c_copy.requires_grad = True
        rhs.requires_grad = True
        rhs_copy.requires_grad = True

        actual = kron(kron(a_copy, b_copy), c_copy).matmul(rhs_copy)
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        res = kp_lazy_var.matmul(rhs)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(a_copy.grad, a.grad))
        self.assertTrue(approx_equal(b_copy.grad, b.grad))
        self.assertTrue(approx_equal(c_copy.grad, c.grad))
        self.assertTrue(approx_equal(rhs_copy.grad, rhs.grad))

    def test_matmul_batch_mat(self):
        avar = torch.tensor(a.repeat(3, 1, 1), requires_grad=True)
        bvar = torch.tensor(b.repeat(3, 1, 1), requires_grad=True)
        cvar = torch.tensor(c.repeat(3, 1, 1), requires_grad=True)
        mat = torch.tensor(torch.randn(3, 24, 5), requires_grad=True)
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(avar), NonLazyTensor(bvar), NonLazyTensor(cvar))
        res = kp_lazy_var.matmul(mat)

        avar_copy = torch.tensor(a.repeat(3, 1, 1), requires_grad=True)
        bvar_copy = torch.tensor(b.repeat(3, 1, 1), requires_grad=True)
        cvar_copy = torch.tensor(c.repeat(3, 1, 1), requires_grad=True)
        mat_copy = torch.tensor(mat.detach(), requires_grad=True)
        actual = kron(kron(avar_copy, bvar_copy), cvar_copy).matmul(mat_copy)
        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(avar_copy.grad, avar.grad))
        self.assertTrue(approx_equal(bvar_copy.grad, bvar.grad))
        self.assertTrue(approx_equal(cvar_copy.grad, cvar.grad))
        self.assertTrue(approx_equal(mat_copy.grad, mat.grad))

    def test_evaluate(self):
        avar = a
        bvar = b
        cvar = c
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(avar), NonLazyTensor(bvar), NonLazyTensor(cvar))
        res = kp_lazy_var.evaluate()
        actual = kron(kron(avar, bvar), cvar)
        self.assertTrue(approx_equal(res, actual))

        avar = a.repeat(3, 1, 1)
        bvar = b.repeat(3, 1, 1)
        cvar = c.repeat(3, 1, 1)
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(avar), NonLazyTensor(bvar), NonLazyTensor(cvar))
        res = kp_lazy_var.evaluate()
        actual = kron(kron(avar, bvar), cvar)
        self.assertTrue(approx_equal(res, actual))

    def test_diag(self):
        avar = a
        bvar = b
        cvar = c
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(avar), NonLazyTensor(bvar), NonLazyTensor(cvar))
        res = kp_lazy_var.diag()
        actual = kron(kron(avar, bvar), cvar).diag()
        self.assertTrue(approx_equal(res, actual))

        avar = a.repeat(3, 1, 1)
        bvar = b.repeat(3, 1, 1)
        cvar = c.repeat(3, 1, 1)
        kp_lazy_var = KroneckerProductLazyTensor(NonLazyTensor(avar), NonLazyTensor(bvar), NonLazyTensor(cvar))
        res = kp_lazy_var.diag()
        actual_mat = kron(kron(avar, bvar), cvar)
        actual = torch.stack([actual_mat[0].diag(), actual_mat[1].diag(), actual_mat[2].diag()])
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
