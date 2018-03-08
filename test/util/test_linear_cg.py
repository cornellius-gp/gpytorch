import torch
import unittest
from gpytorch.utils import approx_equal
from gpytorch.utils.linear_cg import linear_cg


class TestLinearCG(unittest.TestCase):
    def test_cg(self):
        size = 100
        matrix = torch.DoubleTensor(size, size).normal_()
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.DoubleTensor(matrix.size(-1)).fill_(1e-1).diag())

        rhs = torch.DoubleTensor(size, 50).normal_()
        solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size)

        # Check cg
        matrix_chol = matrix.potrf()
        actual = torch.potrs(rhs, matrix_chol)
        self.assertTrue(approx_equal(solves, actual))

    def test_cg_with_tridiag(self):
        size = 10
        matrix = torch.DoubleTensor(size, size).normal_()
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.DoubleTensor(matrix.size(-1)).fill_(1e-1).diag())

        rhs = torch.DoubleTensor(size, 50).normal_()
        solves, t_mats = linear_cg(
            matrix.matmul,
            rhs=rhs,
            n_tridiag=5,
            max_iter=size,
            tolerance=0,
        )

        # Check cg
        matrix_chol = matrix.potrf()
        actual = torch.potrs(rhs, matrix_chol)
        self.assertTrue(approx_equal(solves, actual))

        # Check tridiag
        eigs = matrix.symeig()[0]
        for i in range(5):
            approx_eigs = t_mats[i].symeig()[0]
            self.assertTrue(approx_equal(eigs, approx_eigs))

    def test_batch_cg(self):
        batch = 5
        size = 100
        matrix = torch.DoubleTensor(batch, size, size).normal_()
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.DoubleTensor(matrix.size(-1)).fill_(1e-1).diag())

        rhs = torch.DoubleTensor(batch, size, 50).normal_()
        solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size)

        # Check cg
        matrix_chol = torch.cat([matrix[i].potrf().unsqueeze(0) for i in range(5)])
        actual = torch.cat([
            torch.potrs(rhs[i], matrix_chol[i]).unsqueeze(0) for i in range(5)
        ])
        self.assertTrue(approx_equal(solves, actual))

    def test_batch_cg_with_tridiag(self):
        batch = 5
        size = 10
        matrix = torch.DoubleTensor(batch, size, size).normal_()
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.DoubleTensor(matrix.size(-1)).fill_(1e-1).diag())

        rhs = torch.DoubleTensor(batch, size, 50).normal_()
        solves, t_mats = linear_cg(
            matrix.matmul,
            rhs=rhs,
            n_tridiag=8,
            max_iter=size,
            tolerance=0,
        )

        # Check cg
        matrix_chol = torch.cat([matrix[i].potrf().unsqueeze(0) for i in range(5)])
        actual = torch.cat([
            torch.potrs(rhs[i], matrix_chol[i]).unsqueeze(0) for i in range(5)
        ])
        self.assertTrue(approx_equal(solves, actual))

        # Check tridiag
        for i in range(5):
            eigs = matrix[i].symeig()[0]
            for j in range(8):
                approx_eigs = t_mats[j, i].symeig()[0]
                self.assertLess(
                    torch.mean(torch.abs((eigs - approx_eigs) / eigs)),
                    0.05,
                )


if __name__ == '__main__':
    unittest.main()
