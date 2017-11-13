import math
import torch
import gpytorch
import numpy as np
from torch.autograd import Variable
from gpytorch.utils.kronecker_product import sym_toeplitz_derivative_quadratic_form
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from gpytorch.utils.function_factory import trace_logdet_quad_form_factory
from gpytorch.utils.toeplitz import index_coef_to_sparse, sym_toeplitz_matmul
from gpytorch.utils.kronecker_product import kronecker_product, list_of_indices_and_values_to_sparse


def test_trace_logdet_quad_form_factory():
    x = Variable(torch.linspace(0, 1, 10))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, grid_size=4, grid_bounds=[(0, 1)])
    covar_module.eval()
    c = Variable(covar_module.forward(x.unsqueeze(1), x.unsqueeze(1)).c.data, requires_grad=True)

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


def test_interpolated_toeplitz_gp_marginal_log_likelihood_forward():
    x = Variable(torch.linspace(0, 1, 5))
    y = torch.randn(5)
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, grid_size=10, grid_bounds=[(0, 1)])
    covar_module.eval()
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))
    c = covar_x.c.data
    T = utils.toeplitz.sym_toeplitz(c)

    W_left = index_coef_to_sparse(covar_x.J_left, covar_x.C_left, len(c))
    W_right = index_coef_to_sparse(covar_x.J_right, covar_x.C_right, len(c))

    W_left_dense = W_left.to_dense()
    W_right_dense = W_right.to_dense()

    WTW = W_left_dense.matmul(T.matmul(W_right_dense.t())) + torch.eye(len(x)) * 1e-4

    quad_form_actual = y.dot(WTW.inverse().matmul(y))
    chol_T = torch.potrf(WTW)
    log_det_actual = chol_T.diag().log().sum() * 2

    actual = -0.5 * (log_det_actual + quad_form_actual + math.log(2 * math.pi) * len(y))

    res = covar_x.exact_gp_marginal_log_likelihood(Variable(y))
    assert all(torch.abs((res.data - actual) / actual) < 0.05)


def test_interpolated_toeplitz_gp_marginal_log_likelihood_backward():
    x = Variable(torch.linspace(0, 1, 5))
    y = Variable(torch.randn(5), requires_grad=True)
    noise = Variable(torch.Tensor([1e-4]), requires_grad=True)

    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, grid_size=10, grid_bounds=[(0, 1)])
    covar_module.eval()
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))

    c = Variable(covar_x.c.data, requires_grad=True)

    W_left = index_coef_to_sparse(covar_x.J_left, covar_x.C_left, len(c))
    W_right = index_coef_to_sparse(covar_x.J_right, covar_x.C_right, len(c))

    W_left_dense = Variable(W_left.to_dense())
    W_right_dense = Variable(W_right.to_dense())

    T = Variable(torch.zeros(len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            T[i, j] = utils.toeplitz.sym_toeplitz_getitem(c, i, j)

    WTW = W_left_dense.matmul(T.matmul(W_right_dense.t())) + Variable(torch.eye(len(x))) * noise

    quad_form_actual = y.dot(WTW.inverse().matmul(y))
    log_det_actual = _det(WTW).log()

    actual_nll = -0.5 * (log_det_actual + quad_form_actual + math.log(2 * math.pi) * len(y))
    actual_nll.backward()

    actual_c_grad = c.grad.data.clone()
    actual_y_grad = y.grad.data.clone()
    actual_noise_grad = noise.grad.data.clone()

    c.grad.data.fill_(0)
    y.grad.data.fill_(0)
    noise.grad.data.fill_(0)

    covar_x = gpytorch.lazy.ToeplitzLazyVariable(c,
                                                 covar_x.J_left,
                                                 covar_x.C_left,
                                                 covar_x.J_right,
                                                 covar_x.C_right,
                                                 noise)
    res = covar_x.exact_gp_marginal_log_likelihood(y)
    res.backward()

    res_c_grad = covar_x.c.grad.data
    res_y_grad = y.grad.data
    res_noise_grad = noise.grad.data

    assert (actual_c_grad - res_c_grad).norm() / res_c_grad.norm() < 0.05
    assert (actual_y_grad - res_y_grad).norm() / res_y_grad.norm() < 1e-3
    assert (actual_noise_grad - res_noise_grad).norm() / res_noise_grad.norm() < 1e-3


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


def foo_kp_toeplitz_gp_marginal_log_likelihood_backward():
    x = torch.cat([Variable(torch.linspace(0, 1, 2)).unsqueeze(1)] * 3, 1)
    y = Variable(torch.randn(2), requires_grad=True)
    rbf_module = RBFKernel()
    rbf_module.initialize(log_lengthscale=-2)
    covar_module = GridInterpolationKernel(rbf_module, grid_size=5, grid_bounds=[(0, 1), (0, 1), (0, 1)])
    covar_module.eval()

    kronecker_var = covar_module.forward(x, x)

    cs = Variable(torch.zeros(3, 5), requires_grad=True)
    J_lefts = []
    C_lefts = []
    J_rights = []
    C_rights = []
    Ts = []
    for i in range(3):
        covar_x = covar_module.forward(x[:, i].unsqueeze(1), x[:, i].unsqueeze(1))
        cs.data[i] = covar_x.c.data
        J_lefts.append(covar_x.J_left)
        C_lefts.append(covar_x.C_left)
        J_rights.append(covar_x.J_right)
        C_rights.append(covar_x.C_right)
        T = Variable(torch.zeros(len(cs[i].data), len(cs[i].data)))
        for k in range(len(cs[i].data)):
            for j in range(len(cs[i].data)):
                T[k, j] = utils.toeplitz.toeplitz_getitem(cs[i], cs[i], k, j)
        Ts.append(T)

    W_left = list_of_indices_and_values_to_sparse(J_lefts, C_lefts, cs)
    W_right = list_of_indices_and_values_to_sparse(J_rights, C_rights, cs)
    W_left_dense = Variable(W_left.to_dense())
    W_right_dense = Variable(W_right.to_dense())
    K = kronecker_product(Ts)
    WKW = W_left_dense.matmul(K.matmul(W_right_dense.t()))
    quad_form_actual = y.dot(WKW.inverse().matmul(y))
    log_det_actual = _det(WKW).log()

    actual_nll = -0.5 * (log_det_actual + quad_form_actual + math.log(2 * math.pi) * len(y))
    actual_nll.backward()
    actual_cs_grad = cs.grad.data.clone()
    actual_y_grad = y.grad.data.clone()

    y.grad.data.fill_(0)
    cs.grad.data.fill_(0)

    kronecker_var = gpytorch.lazy.kroneckerProductLazyVariable(cs,
                                                               kronecker_var.J_lefts,
                                                               kronecker_var.C_lefts,
                                                               kronecker_var.J_rights,
                                                               kronecker_var.C_rights)
    gpytorch.functions.num_trace_samples = 100
    res = kronecker_var.exact_gp_marginal_log_likelihood(y)
    res.backward()

    res_cs_grad = covar_x.cs.grad.data
    res_y_grad = y.grad.data

    assert (actual_cs_grad - res_cs_grad).norm() / res_cs_grad.norm() < 0.05
    assert (actual_y_grad - res_y_grad).norm() / res_y_grad.norm() < 1e-3

    y.grad.data.fill_(0)
    cs.grad.data.fill_(0)

    gpytorch.functions.fastest = False
    res = kronecker_var.exact_gp_marginal_log_likelihood(y)
    res.backward()

    res_cs_grad = covar_x.cs.grad.data
    res_y_grad = y.grad.data

    assert (actual_cs_grad - res_cs_grad).norm() / res_cs_grad.norm() < 1e-3
    assert (actual_y_grad - res_y_grad).norm() / res_y_grad.norm() < 1e-3


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
