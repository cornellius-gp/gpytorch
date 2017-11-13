import math
import torch
import gpytorch
import numpy as np
from torch.autograd import Variable


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
