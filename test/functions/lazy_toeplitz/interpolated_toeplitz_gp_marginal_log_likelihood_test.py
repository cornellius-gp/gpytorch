import torch
import math
from gpytorch.functions.lazy_toeplitz import InterpolatedToeplitzGPMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from gpytorch.utils.toeplitz import index_coef_to_sparse
from torch.autograd import Variable


def test_toeplitz_gp_marginal_log_likelihood_forward():
    x = Variable(torch.linspace(0, 1, 5))
    y = torch.randn(5)
    noise = torch.Tensor([1e-4])
    covar_module = GridInterpolationKernel(RBFKernel().initialize(log_lengthscale=-4), 10)
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

    res = InterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right, num_samples=1000)(Variable(c),
                                                                                         Variable(y),
                                                                                         Variable(noise)).data
    assert all(torch.abs((res - actual) / actual) < 0.05)


def test_toeplitz_gp_marginal_log_likelihood_backward():
    x = Variable(torch.linspace(0, 1, 5))
    y = Variable(torch.randn(5), requires_grad=True)
    noise = Variable(torch.Tensor([1e-4]), requires_grad=True)

    covar_module = GridInterpolationKernel(RBFKernel().initialize(log_lengthscale=-4), 10)
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))

    c = Variable(covar_x.c.data, requires_grad=True)

    W_left = index_coef_to_sparse(covar_x.J_left, covar_x.C_left, len(c))
    W_right = index_coef_to_sparse(covar_x.J_right, covar_x.C_right, len(c))

    W_left_dense = Variable(W_left.to_dense())
    W_right_dense = Variable(W_right.to_dense())

    T = Variable(torch.zeros(len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            T[i, j] = utils.toeplitz.toeplitz_getitem(c, c, i, j)

    WTW = W_left_dense.matmul(T.matmul(W_right_dense.t())) + Variable(torch.eye(len(x))) * noise

    quad_form_actual = y.dot(WTW.inverse().matmul(y))
    log_det_actual = _det(WTW).log()

    actual_nll = -0.5 * (log_det_actual + quad_form_actual + math.log(2 * math.pi) * len(y))
    actual_nll.backward()

    actual_c_grad = c.grad.data
    actual_y_grad = y.grad.data
    actual_noise_grad = noise.grad.data

    c.grad.data.fill_(0)
    y.grad.data.fill_(0)
    noise.grad.data.fill_(0)

    res = InterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right, num_samples=1000)(c, y, noise)
    res.backward()

    res_c_grad = c.grad.data
    res_y_grad = y.grad.data
    res_noise_grad = noise.grad.data

    assert utils.approx_equal(actual_c_grad, res_c_grad)
    assert utils.approx_equal(actual_y_grad, res_y_grad)
    assert utils.approx_equal(actual_noise_grad, res_noise_grad)


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
