from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
import torch
import gpytorch
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
        diag = torch.randn(7).abs_()

        self.base_mat_var = Variable(base_mat, requires_grad=True)
        self.left_mat_var = Variable(left_mat, requires_grad=True)
        self.right_mat_var = Variable(right_mat, requires_grad=True)
        self.diag_var = Variable(diag, requires_grad=True)

        self.base_mat_var_copy = Variable(base_mat, requires_grad=True)
        self.left_mat_var_copy = Variable(left_mat, requires_grad=True)
        self.right_mat_var_copy = Variable(right_mat, requires_grad=True)
        self.diag_var_copy = Variable(diag, requires_grad=True)

    def test_matmul_vec(self):
        vec = torch.randn(7)
        vec_var = Variable(vec)
        vec_var_copy = Variable(vec)

        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
            self.diag_var,
        ).matmul(vec_var)
        actual = (
            self.left_mat_var_copy.
            matmul(self.base_mat_var_copy.inverse()).
            matmul(self.right_mat_var_copy.transpose(-1, -2)).
            add(self.diag_var_copy.diag())
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
            self.diag_var,
        ).matmul(mat_var)
        actual = (
            self.left_mat_var_copy.
            matmul(self.base_mat_var_copy.inverse()).
            matmul(self.right_mat_var_copy.transpose(-1, -2)).
            add(self.diag_var_copy.diag())
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

    def test_add_diag(self):
        lv = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
            self.diag_var,
        )
        ev = lv.evaluate()

        res = lv.add_diag(torch.Tensor([0.5])).evaluate()
        actual = gpytorch.add_diag(ev, torch.Tensor([0.5]))
        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

    def test_diag(self):
        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
            self.diag_var,
        ).diag()
        actual = (
            self.left_mat_var_copy.
            matmul(self.base_mat_var_copy.inverse()).
            matmul(self.right_mat_var_copy.transpose(-1, -2)).
            add(self.diag_var_copy.diag())
        ).diag()

        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

    def test_inv_quad_log_det(self):
        mat = torch.randn(7, 6)
        mat_var = Variable(mat)
        mat_var_copy = Variable(mat)

        with gpytorch.settings.num_trace_samples(1000):
            actual_inv_quad, actual_log_det = gpytorch.inv_quad_log_det(
                (
                    self.left_mat_var_copy.
                    matmul(self.base_mat_var_copy.inverse()).
                    matmul(self.left_mat_var_copy.transpose(-1, -2)).
                    add(self.diag_var_copy.diag())
                ),
                inv_quad_rhs=mat_var_copy,
                log_det=True,
            )
            res_inv_quad, res_log_det = InvQuadLazyVariable(
                self.base_mat_var,
                self.left_mat_var,
                self.left_mat_var,
                self.diag_var,
            ).inv_quad_log_det(inv_quad_rhs=mat_var, log_det=True)

        assert(torch.abs(((res_inv_quad - actual_inv_quad) / actual_inv_quad)).item() < 1e-1)
        assert(torch.abs(((res_log_det - actual_log_det) / actual_log_det)).item() < 1e-1)


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
        diag = torch.randn(2, 4).abs_().add_(1)

        self.base_mat_var = Variable(base_mat, requires_grad=True)
        self.left_mat_var = Variable(left_mat, requires_grad=True)
        self.right_mat_var = Variable(right_mat, requires_grad=True)
        self.diag_var = Variable(diag, requires_grad=True)

        self.base_mat_var_copy = Variable(base_mat, requires_grad=True)
        self.left_mat_var_copy = Variable(left_mat, requires_grad=True)
        self.right_mat_var_copy = Variable(right_mat, requires_grad=True)
        self.diag_var_copy = Variable(diag, requires_grad=True)

    def test_matmul(self):
        mat = torch.randn(2, 4, 6)
        mat_var = Variable(mat)
        mat_var_copy = Variable(mat)

        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
            self.diag_var,
        ).matmul(mat_var)
        actual = (
            self.left_mat_var_copy.
            matmul(torch.cat([
                self.base_mat_var_copy[0].inverse().unsqueeze(0),
                self.base_mat_var_copy[1].inverse().unsqueeze(0),
            ])).matmul(self.right_mat_var_copy.transpose(-1, -2)).
            add(torch.cat([
                self.diag_var_copy[0].diag().unsqueeze(0),
                self.diag_var_copy[1].diag().unsqueeze(0),
            ]))
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

    def test_add_diag(self):
        lv = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
            self.diag_var,
        )
        ev = lv.evaluate()

        res = lv.add_diag(torch.Tensor([1])).evaluate()
        actual = gpytorch.add_diag(ev, torch.Tensor([1]))
        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

    def test_diag(self):
        res = InvQuadLazyVariable(
            self.base_mat_var,
            self.left_mat_var,
            self.right_mat_var,
            self.diag_var,
        ).diag()
        actual = torch.cat([
            (
                self.left_mat_var_copy[0].
                matmul(self.base_mat_var_copy[0].inverse()).
                matmul(self.right_mat_var_copy[0].transpose(-1, -2)).
                add(self.diag_var_copy[0].diag())
            ).diag().unsqueeze(0),
            (
                self.left_mat_var_copy[1].
                matmul(self.base_mat_var_copy[1].inverse()).
                matmul(self.right_mat_var_copy[1].transpose(-1, -2)).
                add(self.diag_var_copy[1].diag())
            ).diag().unsqueeze(0),
        ])

        assert(((res - actual).norm() / actual.norm()).item() < 1e-3)

    def test_inv_quad_log_det(self):
        mat = torch.randn(2, 4, 6)
        mat_var = Variable(mat)
        mat_var_copy = Variable(mat)

        with gpytorch.settings.num_trace_samples(1000):
            res_inv_quad, res_log_det = InvQuadLazyVariable(
                self.base_mat_var,
                self.left_mat_var,
                self.left_mat_var,
                self.diag_var,
            ).inv_quad_log_det(inv_quad_rhs=mat_var, log_det=True)
            actual_inv_quad, actual_log_det = gpytorch.inv_quad_log_det(
                (
                    self.left_mat_var_copy.
                    matmul(torch.cat([
                        self.base_mat_var_copy[0].inverse().unsqueeze(0),
                        self.base_mat_var_copy[1].inverse().unsqueeze(0),
                    ])).matmul(self.left_mat_var_copy.transpose(-1, -2)).
                    add(torch.cat([
                        self.diag_var_copy[0].diag().unsqueeze(0),
                        self.diag_var_copy[1].diag().unsqueeze(0),
                    ]))
                ),
                inv_quad_rhs=mat_var_copy,
                log_det=True,
            )

        assert(((res_inv_quad - actual_inv_quad).norm() / actual_inv_quad.norm()) < 1e-1)
        assert(((res_log_det - actual_log_det).norm() / actual_log_det.norm()) < 1e-1)
