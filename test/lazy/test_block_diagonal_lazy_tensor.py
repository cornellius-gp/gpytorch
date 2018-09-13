from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import BlockDiagonalLazyTensor, NonLazyTensor
from gpytorch.utils import approx_equal


blocks = torch.randn(8, 4, 4)
blocks = blocks.transpose(-1, -2).matmul(blocks)


class TestBlockDiagonalLazyTensor(unittest.TestCase):
    def test_matmul(self):
        rhs = torch.randn(4 * 8, 4)
        rhs_var = Variable(rhs, requires_grad=True)
        rhs_var_copy = Variable(rhs, requires_grad=True)

        block_var = Variable(blocks, requires_grad=True)
        block_var_copy = Variable(blocks, requires_grad=True)

        actual_block_diagonal = Variable(torch.zeros(32, 32))
        for i in range(8):
            actual_block_diagonal[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = block_var_copy[i]

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var)).matmul(rhs_var)
        actual = actual_block_diagonal.matmul(rhs_var_copy)

        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()
        res.sum().backward()

        self.assertTrue(approx_equal(rhs_var.grad.data, rhs_var_copy.grad.data))
        self.assertTrue(approx_equal(block_var.grad.data, block_var_copy.grad.data))

    def test_batch_matmul(self):
        rhs = torch.randn(2, 4 * 4, 4)
        rhs_var = Variable(rhs, requires_grad=True)
        rhs_var_copy = Variable(rhs, requires_grad=True)

        block_var = Variable(blocks, requires_grad=True)
        block_var_copy = Variable(blocks, requires_grad=True)

        actual_block_diagonal = Variable(torch.zeros(2, 16, 16))
        for i in range(2):
            for j in range(4):
                actual_block_diagonal[i, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = block_var_copy[i * 4 + j]

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var), n_blocks=4).matmul(rhs_var)
        actual = actual_block_diagonal.matmul(rhs_var_copy)

        self.assertTrue(approx_equal(res.data, actual.data))

        actual.sum().backward()
        res.sum().backward()

        self.assertTrue(approx_equal(rhs_var.grad.data, rhs_var_copy.grad.data))
        self.assertTrue(approx_equal(block_var.grad.data, block_var_copy.grad.data))

    def test_diag(self):
        block_var = Variable(blocks, requires_grad=True)
        actual_block_diagonal = Variable(torch.zeros(32, 32))
        for i in range(8):
            actual_block_diagonal[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = block_var[i]

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var)).diag()
        actual = actual_block_diagonal.diag()
        self.assertTrue(approx_equal(actual.data, res.data))

    def test_batch_diag(self):
        block_var = Variable(blocks, requires_grad=True)
        actual_block_diagonal = Variable(torch.zeros(2, 16, 16))
        for i in range(2):
            for j in range(4):
                actual_block_diagonal[i, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = block_var[i * 4 + j]

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var), n_blocks=4).diag()
        actual = torch.cat([actual_block_diagonal[0].diag().unsqueeze(0), actual_block_diagonal[1].diag().unsqueeze(0)])
        self.assertTrue(approx_equal(actual.data, res.data))

    def test_getitem(self):
        block_var = Variable(blocks, requires_grad=True)
        actual_block_diagonal = Variable(torch.zeros(32, 32))
        for i in range(8):
            actual_block_diagonal[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = block_var[i]

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var))[:5, 2]
        actual = actual_block_diagonal[:5, 2]
        self.assertTrue(approx_equal(actual.data, res.data))

    def test_getitem_batch(self):
        block_var = Variable(blocks, requires_grad=True)
        actual_block_diagonal = Variable(torch.zeros(2, 16, 16))
        for i in range(2):
            for j in range(4):
                actual_block_diagonal[i, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = block_var[i * 4 + j]

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var), n_blocks=4)[0].evaluate()
        actual = actual_block_diagonal[0]
        self.assertTrue(approx_equal(actual.data, res.data))

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var), n_blocks=4)[0, :5].evaluate()
        actual = actual_block_diagonal[0, :5]
        self.assertTrue(approx_equal(actual.data, res.data))

        res = BlockDiagonalLazyTensor(NonLazyTensor(block_var), n_blocks=4)[1:, :5, 2]
        actual = actual_block_diagonal[1:, :5, 2]
        self.assertTrue(approx_equal(actual.data, res.data))


if __name__ == "__main__":
    unittest.main()
