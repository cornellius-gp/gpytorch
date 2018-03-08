import math
import torch
import unittest
import gpytorch
import numpy as np
from torch.autograd import Variable
from gpytorch.utils import approx_equal, function_factory
from gpytorch.lazy import NonLazyVariable


_exact_gp_mll_class = function_factory.exact_gp_mll_factory()


class TestFunctionFactory(unittest.TestCase):
    def test_forward_inv_mm(self):
        for n_cols in [2, 3, 4]:
            a = torch.Tensor([
                [5, -3, 0],
                [-3, 5, 0],
                [0, 0, 2],
            ])
            b = torch.randn(3, n_cols)
            actual = a.inverse().mm(b)

            a_var = Variable(a)
            b_var = Variable(b)
            out_var = gpytorch.inv_matmul(a_var, b_var)
            res = out_var.data

            self.assertLess(torch.norm(actual - res), 1e-4)

    def test_backward_inv_mm(self):
        for n_cols in [2, 3, 4]:
            a = torch.Tensor([
                [5, -3, 0],
                [-3, 5, 0],
                [0, 0, 2],
            ])
            b = torch.ones(3, 3).fill_(2)
            c = torch.randn(3, n_cols)
            actual_a_grad = -torch.mm(
                a.inverse().mul_(0.5).mm(torch.eye(3, n_cols)),
                a.inverse().mul_(0.5).mm(c).t()
            ) * 2 * 2
            actual_c_grad = (a.inverse() / 2).t().mm(torch.eye(3, n_cols)) * 2

            a_var = Variable(a, requires_grad=True)
            c_var = Variable(c, requires_grad=True)
            out_var = a_var.mul(Variable(b))
            out_var = gpytorch.inv_matmul(out_var, c_var)
            out_var = out_var.mul(Variable(torch.eye(3, n_cols))).sum() * 2
            out_var.backward()
            a_res = a_var.grad.data
            c_res = c_var.grad.data

            self.assertLess(torch.norm(actual_a_grad - a_res), 1e-4)
            self.assertLess(torch.norm(actual_c_grad - c_res), 1e-4)

    def test_forward_inv_mv(self):
        a = torch.Tensor([
            [5, -3, 0],
            [-3, 5, 0],
            [0, 0, 2],
        ])
        b = torch.randn(3)
        actual = a.inverse().mv(b)

        a_var = Variable(a)
        b_var = Variable(b)
        out_var = gpytorch.inv_matmul(a_var, b_var)
        res = out_var.data

        self.assertLess(torch.norm(actual - res), 1e-4)

    def test_backward_inv_mv(self):
        a = torch.Tensor([
            [5, -3, 0],
            [-3, 5, 0],
            [0, 0, 2],
        ])
        b = torch.ones(3, 3).fill_(2)
        c = torch.randn(3)
        actual_a_grad = -(
            torch.ger(
                a.inverse().mul_(0.5).mv(torch.ones(3)),
                a.inverse().mul_(0.5).mv(c)
            ) * 2 * 2
        )
        actual_c_grad = (a.inverse() / 2).t().mv(torch.ones(3)) * 2

        a_var = Variable(a, requires_grad=True)
        c_var = Variable(c, requires_grad=True)
        out_var = a_var.mul(Variable(b))
        out_var = gpytorch.inv_matmul(out_var, c_var)
        out_var = out_var.sum() * 2
        out_var.backward()
        a_res = a_var.grad.data
        c_res = c_var.grad.data

        self.assertLess(torch.norm(actual_a_grad - a_res), 1e-4)
        self.assertLess(torch.norm(actual_c_grad - c_res), 1e-4)

    def test_normal_gp_mll_forward(self):
        covar = torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ])
        y = torch.randn(3)

        actual = y.dot(covar.inverse().mv(y))
        actual += math.log(np.linalg.det(covar.numpy()))
        actual += math.log(2 * math.pi) * len(y)
        actual *= -0.5

        covarvar = Variable(covar)
        yvar = Variable(y)

        res = _exact_gp_mll_class()(covarvar, yvar)
        for d in torch.abs(actual - res.data).div(res.data):
            self.assertLess(d, 0.1)

    def test_normal_gp_mll_backward(self):
        covar = torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ])
        y = torch.randn(3)

        covarvar = Variable(covar, requires_grad=True)
        yvar = Variable(y, requires_grad=True)
        actual_mat_grad = torch.ger(covar.inverse().mv(y), covar.inverse().mv(y))
        actual_mat_grad -= covar.inverse()
        actual_mat_grad *= 0.5
        actual_mat_grad *= 3  # For grad output

        actual_y_grad = -covar.inverse().mv(y)
        actual_y_grad *= 3  # For grad output

        covarvar = Variable(covar, requires_grad=True)
        yvar = Variable(y, requires_grad=True)
        with gpytorch.settings.num_trace_samples(1000):
            output = _exact_gp_mll_class()(covarvar, yvar) * 3
            output.backward()

        self.assertLess(torch.norm(actual_mat_grad - covarvar.grad.data), 1e-1)
        self.assertLess(torch.norm(actual_y_grad - yvar.grad.data), 1e-4)

        with gpytorch.settings.num_trace_samples(0):
            covarvar = Variable(covar, requires_grad=True)
            yvar = Variable(y, requires_grad=True)
            with gpytorch.settings.num_trace_samples(1000):
                output = _exact_gp_mll_class()(covarvar, yvar) * 3
                output.backward()

        self.assertLess(torch.norm(actual_mat_grad - covarvar.grad.data), 1e-1)
        self.assertLess(torch.norm(actual_y_grad - yvar.grad.data), 1e-4)

    def test_normal_trace_log_det_quad_form_forward(self):
        covar = torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ])
        mu_diffs = torch.Tensor([0, -1, 1])
        chol_covar = torch.Tensor([
            [1, -2, 0],
            [0, 1, -2],
            [0, 0, 1],
        ])

        actual = mu_diffs.dot(covar.inverse().matmul(mu_diffs))
        actual += math.log(np.linalg.det(covar.numpy()))
        actual += (covar.inverse().matmul(chol_covar.t().matmul(chol_covar))).trace()

        covarvar = Variable(covar)
        chol_covarvar = Variable(chol_covar)
        mu_diffsvar = Variable(mu_diffs)

        res = gpytorch.trace_logdet_quad_form(mu_diffsvar, chol_covarvar, covarvar)
        self.assertTrue((torch.abs(actual - res.data).div(res.data) < 0.1).all())

    def test_normal_trace_log_det_quad_form_backward(self):
        covar = Variable(torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ]), requires_grad=True)
        mu_diffs = Variable(torch.Tensor([0, -1, 1]), requires_grad=True)
        chol_covar = Variable(torch.Tensor([
            [1, -2, 0],
            [0, 1, -2],
            [0, 0, 1],
        ]), requires_grad=True)

        actual = mu_diffs.dot(covar.inverse().matmul(mu_diffs))
        actual += (covar.inverse().matmul(chol_covar.t().matmul(chol_covar))).trace()
        actual.backward()

        actual_covar_grad = covar.grad.data.clone() + covar.data.inverse()
        actual_mu_diffs_grad = mu_diffs.grad.data.clone()
        actual_chol_covar_grad = chol_covar.grad.data.clone()

        covar = Variable(torch.Tensor([
            [3, -1, 0],
            [-1, 3, 0],
            [0, 0, 3],
        ]), requires_grad=True)
        mu_diffs = Variable(torch.Tensor([0, -1, 1]), requires_grad=True)
        chol_covar = Variable(torch.Tensor([
            [1, -2, 0],
            [0, 1, -2],
            [0, 0, 1],
        ]), requires_grad=True)

        with gpytorch.settings.num_trace_samples(1000):
            res = gpytorch.trace_logdet_quad_form(mu_diffs, chol_covar, covar)
            res.backward()

        res_covar_grad = covar.grad.data
        res_mu_diffs_grad = mu_diffs.grad.data
        res_chol_covar_grad = chol_covar.grad.data

        self.assertLess(
            torch.norm(actual_covar_grad - res_covar_grad),
            1e-1,
        )
        self.assertLess(
            torch.norm(actual_mu_diffs_grad - res_mu_diffs_grad),
            1e-1,
        )
        self.assertLess(
            torch.norm(actual_chol_covar_grad - res_chol_covar_grad),
            1e-1,
        )

    def test_batch_trace_log_det_quad_form_forward(self):
        covar = torch.Tensor([
            [
                [3, -1, 0],
                [-1, 3, 0],
                [0, 0, 3],
            ], [
                [10, -2, 1],
                [-2, 10, 0],
                [1, 0, 10],
            ]
        ])
        mu_diffs = torch.Tensor([
            [0, -1, 1],
            [1, 2, 3]
        ])
        chol_covar = torch.Tensor([
            [
                [1, -2, 0],
                [0, 1, -2],
                [0, 0, 1],
            ], [
                [2, -4, 0],
                [0, 2, -4],
                [0, 0, 2],
            ]
        ])

        actual = mu_diffs[0].dot(covar[0].inverse().matmul(mu_diffs[0]))
        actual += math.log(np.linalg.det(covar[0].numpy()))
        actual += (
            covar[0].inverse().matmul(chol_covar[0].t().matmul(chol_covar[0]))
        ).trace()
        actual += mu_diffs[1].dot(covar[1].inverse().matmul(mu_diffs[1]))
        actual += math.log(np.linalg.det(covar[1].numpy()))
        actual += (
            covar[1].inverse().matmul(chol_covar[1].t().matmul(chol_covar[1]))
        ).trace()

        covarvar = Variable(covar)
        chol_covarvar = Variable(chol_covar)
        mu_diffsvar = Variable(mu_diffs)

        res = gpytorch.trace_logdet_quad_form(mu_diffsvar, chol_covarvar, covarvar)
        self.assertTrue((torch.abs(actual - res.data).div(res.data) < 0.1).all())

    def test_batch_trace_log_det_quad_form_backward(self):
        covar = Variable(torch.Tensor([
            [
                [3, -1, 0],
                [-1, 3, 0],
                [0, 0, 3],
            ], [
                [10, -2, 1],
                [-2, 10, 0],
                [1, 0, 10],
            ]
        ]), requires_grad=True)
        mu_diffs = Variable(torch.Tensor([
            [0, -1, 1],
            [1, 2, 3]
        ]), requires_grad=True)
        chol_covar = Variable(torch.Tensor([
            [
                [1, -2, 0],
                [0, 1, -2],
                [0, 0, 1],
            ], [
                [2, -4, 0],
                [0, 2, -4],
                [0, 0, 2],
            ]
        ]), requires_grad=True)

        actual = mu_diffs[0].dot(covar[0].inverse().matmul(mu_diffs[0]))
        actual += (
            covar[0].inverse().matmul(chol_covar[0].t().matmul(chol_covar[0]))
        ).trace()
        actual += mu_diffs[1].dot(covar[1].inverse().matmul(mu_diffs[1]))
        actual += (
            covar[1].inverse().matmul(chol_covar[1].t().matmul(chol_covar[1]))
        ).trace()
        actual.backward()

        actual_covar_grad = (
            covar.grad.data.clone() +
            torch.cat([
                covar[0].data.inverse().unsqueeze(0),
                covar[1].data.inverse().unsqueeze(0)]
            )
        )
        actual_mu_diffs_grad = mu_diffs.grad.data.clone()
        actual_chol_covar_grad = chol_covar.grad.data.clone()

        covar.grad.data.fill_(0)
        mu_diffs.grad.data.fill_(0)
        chol_covar.grad.data.fill_(0)
        with gpytorch.settings.num_trace_samples(1000):
            res = gpytorch.trace_logdet_quad_form(mu_diffs, chol_covar, covar)
            res.backward()

        res_covar_grad = covar.grad.data
        res_mu_diffs_grad = mu_diffs.grad.data
        res_chol_covar_grad = chol_covar.grad.data

        self.assertLess(torch.norm(actual_covar_grad - res_covar_grad), 1e-1)
        self.assertLess(torch.norm(actual_mu_diffs_grad - res_mu_diffs_grad), 1e-1)
        self.assertLess(torch.norm(actual_chol_covar_grad - res_chol_covar_grad), 1e-1)

    def test_root_decomposition_forward(self):
        a = torch.randn(5, 5)
        a = torch.matmul(a, a.t())

        a_lv = NonLazyVariable(Variable(a, requires_grad=True))
        a_root = a_lv.root_decomposition()

        self.assertLess(
            torch.max(((a_root.matmul(a_root.transpose(-1, -2)).data - a)).abs()),
            1e-2,
        )

    def test_root_decomposition_backward(self):
        a = torch.Tensor([
            [5.0212, 0.5504, -0.1810, 1.5414, 2.9611],
            [0.5504, 2.8000, 1.9944, 0.6208, -0.8902],
            [-0.1810, 1.9944, 3.0505, 1.0790, -1.1774],
            [1.5414, 0.6208, 1.0790, 2.9430, 0.4170],
            [2.9611, -0.8902, -1.1774, 0.4170, 3.3208],
        ])

        a_var = Variable(a, requires_grad=True)
        a_lv = NonLazyVariable(a_var)
        a_root = a_lv.root_decomposition()
        res = a_root.matmul(a_root.transpose(-1, -2))
        res.trace().backward()

        a_var_copy = Variable(a, requires_grad=True)
        a_var_copy.trace().backward()

        self.assertTrue(approx_equal(a_var.grad.data, a_var_copy.grad.data))

    def test_root_decomposition_inv_forward(self):
        a = torch.randn(5, 5)
        a = torch.matmul(a, a.t())

        a_lv = NonLazyVariable(Variable(a, requires_grad=True))
        a_root = a_lv.root_inv_decomposition()

        actual = a.inverse()
        diff = (a_root.matmul(a_root.transpose(-1, -2)).data - actual).abs()
        self.assertLess(torch.max(diff / actual), 1e-2)

    def test_root_decomposition_inv_backward(self):
        a = torch.Tensor([
            [5.0212, 0.5504, -0.1810, 1.5414, 2.9611],
            [0.5504, 2.8000, 1.9944, 0.6208, -0.8902],
            [-0.1810, 1.9944, 3.0505, 1.0790, -1.1774],
            [1.5414, 0.6208, 1.0790, 2.9430, 0.4170],
            [2.9611, -0.8902, -1.1774, 0.4170, 3.3208],
        ])

        a_var = Variable(a, requires_grad=True)
        a_lv = NonLazyVariable(a_var)
        a_root = a_lv.root_inv_decomposition()
        res = a_root.matmul(a_root.transpose(-1, -2))
        res.trace().backward()

        a_var_copy = Variable(a, requires_grad=True)
        a_var_copy.inverse().trace().backward()

        self.assertTrue(approx_equal(a_var.grad.data, a_var_copy.grad.data))


if __name__ == '__main__':
    unittest.main()
