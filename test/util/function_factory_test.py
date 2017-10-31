import math
import torch
import gpytorch
import numpy as np
from torch.autograd import Variable
from gpytorch.utils.kronecker_product import sym_toeplitz_derivative_quadratic_form
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from gpytorch.utils.function_factory import trace_logdet_quad_form_factory
from gpytorch.utils.toeplitz import sym_toeplitz_matmul


def test_trace_logdet_quad_form_factory():
    x = Variable(torch.linspace(0, 1, 10))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, grid_size=4, grid_bounds=[(0, 1)])
    covar_module.eval()
    c = Variable(covar_module.forward(x.unsqueeze(1), x.unsqueeze(1)).base_lazy_variable.column.data,
                 requires_grad=True)

    T = Variable(torch.zeros(4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = utils.toeplitz.toeplitz_getitem(c, c, i, j)

    U = torch.randn(4, 4).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    mu_diff = Variable(torch.randn(4), requires_grad=True)

    actual = _det(T).log() + mu_diff.dot(T.inverse().mv(mu_diff)) + T.inverse().mm(U.t().mm(U)).trace()
    actual.backward()

    actual_c_grad = c.grad.data.clone()
    actual_mu_diff_grad = mu_diff.grad.data.clone()
    actual_U_grad = U.grad.data.clone()

    c.grad.data.fill_(0)
    mu_diff.grad.data.fill_(0)
    U.grad.data.fill_(0)

    def _matmul_closure_factory(*args):
        c, = args
        return lambda mat2: sym_toeplitz_matmul(c, mat2)

    def _derivative_quadratic_form_factory(*args):
        return lambda left_vector, right_vector: (sym_toeplitz_derivative_quadratic_form(left_vector, right_vector),)

    covar_args = (c,)

    gpytorch.functions.num_trace_samples = 1000
    res = trace_logdet_quad_form_factory(_matmul_closure_factory,
                                         _derivative_quadratic_form_factory)()(mu_diff, U, *covar_args)
    res.backward()

    res_c_grad = c.grad.data
    res_mu_diff_grad = mu_diff.grad.data
    res_U_grad = U.grad.data

    assert (res.data - actual.data).norm() / actual.data.norm() < 0.15
    assert (res_c_grad - actual_c_grad).norm() / actual_c_grad.norm() < 0.15
    assert (res_mu_diff_grad - actual_mu_diff_grad).norm() / actual_mu_diff_grad.norm() < 1e-3
    assert (res_U_grad - actual_U_grad).norm() / actual_U_grad.norm() < 1e-3

    c.grad.data.fill_(0)
    mu_diff.grad.data.fill_(0)
    U.grad.data.fill_(0)

    covar_args = (c,)

    gpytorch.functions.fastest = False
    res = trace_logdet_quad_form_factory(_matmul_closure_factory,
                                         _derivative_quadratic_form_factory)()(mu_diff, U, *covar_args)
    res.backward()

    res_c_grad = c.grad.data
    res_mu_diff_grad = mu_diff.grad.data
    res_U_grad = U.grad.data

    assert (res.data - actual.data).norm() / actual.data.norm() < 1e-3
    assert (res_c_grad - actual_c_grad).norm() / actual_c_grad.norm() < 1e-3
    assert (res_mu_diff_grad - actual_mu_diff_grad).norm() / actual_mu_diff_grad.norm() < 1e-3
    assert (res_U_grad - actual_U_grad).norm() / actual_U_grad.norm() < 1e-3


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


def test_kp_toeplitz_gp_marginal_log_likelihood_forward():
    x = torch.cat([Variable(torch.linspace(0, 1, 2)).unsqueeze(1)] * 3, 1)
    y = torch.randn(2)
    rbf_module = RBFKernel()
    rbf_module.initialize(log_lengthscale=-2)
    covar_module = GridInterpolationKernel(rbf_module, grid_size=5, grid_bounds=[(0, 1), (0, 1), (0, 1)])
    covar_module.eval()

    kronecker_var = covar_module.forward(x, x)
    kronecker_var_eval = kronecker_var.evaluate()
    res = kronecker_var.exact_gp_marginal_log_likelihood(Variable(y)).data
    actual = gpytorch.exact_gp_marginal_log_likelihood(kronecker_var_eval, Variable(y)).data
    assert all(torch.abs((res - actual) / actual) < 0.05)


def _det(A):
    n = len(A)
    if n == 1:
        return A[0, 0]

    det = A[0, 0] * _det(A[1:, 1:])
    det += math.pow(-1, n - 1) * (A[0, -1] * _det(A[1:, :-1]))
    for i in range(1, n - 1):
        const = A[0, i]
        lower_left = A[1:, :i]
        lower_right = A[1:, i + 1:]
        matrix = torch.cat((lower_left, lower_right), 1)
        det += const * math.pow(-1, i) * _det(matrix)

    return det
