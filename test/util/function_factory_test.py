import math
import torch
import gpytorch
import numpy as np
from torch.autograd import Variable
from gpytorch.utils import approx_equal


def test_normal_gp_mll_forward():
    covar = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ])
    y = torch.randn(3)

    actual = y.dot(covar.inverse().mv(y))
    actual += math.log(np.linalg.det(covar.numpy()))
    actual += math.log(2 * math.pi) * len(y)
    actual *= -0.5

    covarvar = Variable(covar)
    yvar = Variable(y)

    res = gpytorch.exact_gp_marginal_log_likelihood(covarvar, yvar)
    assert(all(torch.abs(actual - res.data).div(res.data) < 0.1))


def test_normal_gp_mll_backward():
    covar = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
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
    gpytorch.functions.num_trace_samples = 1000
    output = gpytorch.exact_gp_marginal_log_likelihood(covarvar, yvar) * 3
    output.backward()

    assert(torch.norm(actual_mat_grad - covarvar.grad.data) < 1e-1)
    assert(torch.norm(actual_y_grad - yvar.grad.data) < 1e-4)

    gpytorch.functions.fastest = False
    covarvar = Variable(covar, requires_grad=True)
    yvar = Variable(y, requires_grad=True)
    output = gpytorch.exact_gp_marginal_log_likelihood(covarvar, yvar) * 3
    output.backward()

    assert(torch.norm(actual_mat_grad - covarvar.grad.data) < 1e-1)
    assert(torch.norm(actual_y_grad - yvar.grad.data) < 1e-4)


def test_normal_trace_log_det_quad_form_forward():
    covar = torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
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
    assert(all(torch.abs(actual - res.data).div(res.data) < 0.1))


def test_normal_trace_log_det_quad_form_backward():
    covar = Variable(torch.Tensor([
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
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
        [5, -3, 0],
        [-3, 5, 0],
        [0, 0, 2],
    ]), requires_grad=True)
    mu_diffs = Variable(torch.Tensor([0, -1, 1]), requires_grad=True)
    chol_covar = Variable(torch.Tensor([
        [1, -2, 0],
        [0, 1, -2],
        [0, 0, 1],
    ]), requires_grad=True)

    res = gpytorch.trace_logdet_quad_form(mu_diffs, chol_covar, covar)
    res.backward()

    res_covar_grad = covar.grad.data
    res_mu_diffs_grad = mu_diffs.grad.data
    res_chol_covar_grad = chol_covar.grad.data

    assert approx_equal(actual_covar_grad, res_covar_grad)
    assert approx_equal(actual_mu_diffs_grad, res_mu_diffs_grad)
    assert approx_equal(actual_chol_covar_grad, res_chol_covar_grad)
