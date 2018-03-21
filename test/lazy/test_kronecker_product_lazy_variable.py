from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import KroneckerProductLazyVariable, NonLazyVariable
from gpytorch.utils import approx_equal


a = torch.Tensor([
    [4, 0, 2],
    [0, 1, -1],
    [2, -1, 3],
])
b = torch.Tensor([
    [2, 1],
    [1, 2],
])
c = torch.Tensor([
    [4, 0, 1, 0],
    [0, 1, -1, 0],
    [1, -1, 3, 0],
    [0, 0, 0, 1],
])


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
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
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

    def test_matmul_mat(self):
        avar = Variable(a, requires_grad=True)
        bvar = Variable(b, requires_grad=True)
        cvar = Variable(c, requires_grad=True)
        mat = Variable(torch.randn(24, 5), requires_grad=True)
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
        res = kp_lazy_var.matmul(mat)

        avar_copy = Variable(a, requires_grad=True)
        bvar_copy = Variable(b, requires_grad=True)
        cvar_copy = Variable(c, requires_grad=True)
        mat_copy = Variable(mat.data.clone(), requires_grad=True)
        actual = kron(kron(avar_copy, bvar_copy), cvar_copy).matmul(mat_copy)
        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()
        res.sum().backward()
        self.assertTrue(approx_equal(avar_copy.grad.data, avar.grad.data))
        self.assertTrue(approx_equal(bvar_copy.grad.data, bvar.grad.data))
        self.assertTrue(approx_equal(cvar_copy.grad.data, cvar.grad.data))
        self.assertTrue(approx_equal(mat_copy.grad.data, mat.grad.data))

    def test_matmul_batch_mat(self):
        avar = Variable(a.repeat(3, 1, 1), requires_grad=True)
        bvar = Variable(b.repeat(3, 1, 1), requires_grad=True)
        cvar = Variable(c.repeat(3, 1, 1), requires_grad=True)
        mat = Variable(torch.randn(3, 24, 5), requires_grad=True)
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
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
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
        res = kp_lazy_var.evaluate()
        actual = kron(kron(avar, bvar), cvar)
        self.assertTrue(approx_equal(res.data, actual.data))

        avar = Variable(a.repeat(3, 1, 1))
        bvar = Variable(b.repeat(3, 1, 1))
        cvar = Variable(c.repeat(3, 1, 1))
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
        res = kp_lazy_var.evaluate()
        actual = kron(kron(avar, bvar), cvar)
        self.assertTrue(approx_equal(res.data, actual.data))

    def test_diag(self):
        avar = Variable(a)
        bvar = Variable(b)
        cvar = Variable(c)
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
        res = kp_lazy_var.diag()
        actual = kron(kron(avar, bvar), cvar).diag()
        self.assertTrue(approx_equal(res.data, actual.data))

        avar = Variable(a.repeat(3, 1, 1))
        bvar = Variable(b.repeat(3, 1, 1))
        cvar = Variable(c.repeat(3, 1, 1))
        kp_lazy_var = KroneckerProductLazyVariable(
            NonLazyVariable(avar),
            NonLazyVariable(bvar),
            NonLazyVariable(cvar),
        )
        res = kp_lazy_var.diag()
        actual_mat = kron(kron(avar, bvar), cvar)
        actual = torch.stack(
            [actual_mat[0].diag(), actual_mat[1].diag(), actual_mat[2].diag()]
        )
        self.assertTrue(approx_equal(res.data, actual.data))


if __name__ == '__main__':
    unittest.main()
