from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import InvQuadLazyVariable


class TestInvQuadLazyVariable(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        base_mat_root = torch.randn(5, 5)
        base_mat = base_mat_root.t().matmul(base_mat_root)
        base_mat += torch.Tensor(5).fill_(1e-2).diag()
        left_mat = torch.randn(7, 5)
        right_mat = torch.randn(7, 5)

        self.base_mat_var = Variable(base_mat, requires_grad=True)
        self.left_mat_var = Variable(left_mat, requires_grad=True)
        self.right_mat_var = Variable(right_mat, requires_grad=True)

        self.base_mat_var_copy = Variable(base_mat, requires_grad=True)
        self.left_mat_var_copy = Variable(left_mat, requires_grad=True)
        self.right_mat_var_copy = Variable(right_mat, requires_grad=True)

    def test_matmul_vec(self):
        vec = torch.randn(7)
        vec_var = Variable(vec)
        vec_var_copy = Variable(vec)

        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
        ).matmul(vec_var)
        actual = (
            self.left_mat_var_copy.
            matmul(self.base_mat_var_copy.inverse()).
            matmul(self.right_mat_var_copy.transpose(-1, -2))
        ).matmul(vec_var_copy)

        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

        res.sum().backward()
        actual.sum().backward()

        assert((
            (self.base_mat_var_copy.grad - self.base_mat_var.grad).norm() / self.base_mat_var_copy.norm()
        ).item() < .05)
        assert((
            (self.left_mat_var_copy.grad - self.left_mat_var.grad).norm() / self.left_mat_var_copy.norm()
        ).item() < .05)
        assert((
            (self.right_mat_var_copy.grad - self.right_mat_var.grad).norm() / self.right_mat_var_copy.norm()
        ).item() < .05)

    def test_matmul(self):
        mat = torch.randn(7, 6)
        mat_var = Variable(mat)
        mat_var_copy = Variable(mat)

        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
        ).matmul(mat_var)
        actual = (
            self.left_mat_var_copy.
            matmul(self.base_mat_var_copy.inverse()).
            matmul(self.right_mat_var_copy.transpose(-1, -2))
        ).matmul(mat_var_copy)

        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

        res.sum().backward()
        actual.sum().backward()

        assert((
            (self.base_mat_var_copy.grad - self.base_mat_var.grad).norm() / self.base_mat_var_copy.norm()
        ).item() < .05)
        assert((
            (self.left_mat_var_copy.grad - self.left_mat_var.grad).norm() / self.left_mat_var_copy.norm()
        ).item() < .05)
        assert((
            (self.right_mat_var_copy.grad - self.right_mat_var.grad).norm() / self.right_mat_var_copy.norm()
        ).item() < .05)

    def test_getitem(self):
        pass


class TestInvQuadLazyVariableBatch(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'rng_state'):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv('UNLOCK_SEED') is None or os.getenv('UNLOCK_SEED').lower() == 'false':
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

        base_mat_root = torch.randn(2, 3, 3)
        base_mat = base_mat_root.transpose(-1, -2).matmul(base_mat_root)
        base_mat += torch.Tensor(3).fill_(1e-2).diag().unsqueeze(0).expand(2, 3, 3)
        left_mat = torch.randn(2, 4, 3)
        right_mat = torch.randn(2, 4, 3)

        self.base_mat_var = Variable(base_mat, requires_grad=True)
        self.left_mat_var = Variable(left_mat, requires_grad=True)
        self.right_mat_var = Variable(right_mat, requires_grad=True)

        self.base_mat_var_copy = Variable(base_mat, requires_grad=True)
        self.left_mat_var_copy = Variable(left_mat, requires_grad=True)
        self.right_mat_var_copy = Variable(right_mat, requires_grad=True)

    def test_matmul(self):
        mat = torch.randn(2, 4, 6)
        mat_var = Variable(mat)
        mat_var_copy = Variable(mat)

        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
        ).matmul(mat_var)
        actual = (
            self.left_mat_var_copy.
            matmul(torch.cat([
                self.base_mat_var_copy[0].inverse().unsqueeze(0),
                self.base_mat_var_copy[1].inverse().unsqueeze(0),
            ])).matmul(self.right_mat_var_copy.transpose(-1, -2))
        ).matmul(mat_var_copy)

        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

        res.sum().backward()
        actual.sum().backward()

        assert((
            (self.base_mat_var_copy.grad - self.base_mat_var.grad).norm() / self.base_mat_var_copy.norm()
        ).item() < .05)
        assert((
            (self.left_mat_var_copy.grad - self.left_mat_var.grad).norm() / self.left_mat_var_copy.norm()
        ).item() < .05)
        assert((
            (self.right_mat_var_copy.grad - self.right_mat_var.grad).norm() / self.right_mat_var_copy.norm()
        ).item() < .05)

    def test_getitem(self):
        pass
