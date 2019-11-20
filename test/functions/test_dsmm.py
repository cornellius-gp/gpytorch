#!/usr/bin/env python3

import unittest

import torch

import gpytorch


class TestDSMM(unittest.TestCase):
    def test_forward(self):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        dense = torch.randn(3, 3)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.mm(sparse.to_dense(), dense)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_forward_batch(self):
        i = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = torch.randn(2, 3, 3)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.matmul(sparse.to_dense(), dense)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_forward_multi_batch(self):
        i = torch.tensor(
            [[0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long
        )
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 2, 3]))
        dense = torch.randn(2, 2, 3, 3)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.matmul(sparse.to_dense(), dense)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_backward(self):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        dense = torch.randn(3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)
        grad_output = torch.randn(2, 4)

        res = gpytorch.dsmm(sparse, dense)
        res.backward(grad_output)
        actual = torch.mm(sparse.to_dense(), dense_copy)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)

    def test_backward_batch(self):
        i = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = torch.randn(2, 3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)
        grad_output = torch.randn(2, 2, 4)

        res = gpytorch.dsmm(sparse, dense)
        res.backward(grad_output)
        actual = torch.matmul(sparse.to_dense(), dense_copy)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)

    def test_backward_multi_batch(self):
        i = torch.tensor(
            [[0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long
        )
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 2, 3]))
        dense = torch.randn(2, 2, 3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)
        grad_output = torch.randn(2, 2, 2, 4)

        res = gpytorch.dsmm(sparse, dense)
        res.backward(grad_output)
        actual = torch.matmul(sparse.to_dense(), dense_copy)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)

    def test_broadcast_rhs(self):
        i = torch.tensor([[0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        dense = torch.randn(4, 2, 3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.matmul(sparse.to_dense(), dense_copy)
        self.assertLess(torch.norm(res - actual), 1e-5)

        grad_output = torch.randn(4, 2, 2, 4)
        res.backward(grad_output)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)

        i = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = torch.randn(4, 2, 3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.matmul(sparse.to_dense(), dense_copy)
        self.assertLess(torch.norm(res - actual), 1e-5)

        grad_output = torch.randn(4, 2, 2, 4)
        res.backward(grad_output)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)

    def test_broadcast_sparse(self):
        i = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = torch.randn(3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.matmul(sparse.to_dense(), dense_copy)
        self.assertLess(torch.norm(res - actual), 1e-5)

        grad_output = torch.randn(2, 2, 4)
        res.backward(grad_output)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)

    def test_broadcast_singleton(self):
        i = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1], [2, 0, 2, 2, 0, 2]], dtype=torch.long)
        v = torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.float)
        sparse = torch.sparse.FloatTensor(i, v, torch.Size([2, 2, 3]))
        dense = torch.randn(1, 3, 4, requires_grad=True)
        dense_copy = dense.clone().detach().requires_grad_(True)

        res = gpytorch.dsmm(sparse, dense)
        actual = torch.matmul(sparse.to_dense(), dense_copy)
        self.assertLess(torch.norm(res - actual), 1e-5)

        grad_output = torch.randn(2, 2, 4)
        res.backward(grad_output)
        actual.backward(grad_output)
        self.assertLess(torch.norm(dense.grad - dense_copy.grad).item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
