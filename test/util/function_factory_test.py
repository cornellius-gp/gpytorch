from gpytorch.utils.kronecker_product import sym_toeplitz_derivative_quadratic_form
from torch.autograd import Variable
import torch
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from gpytorch.utils.function_factory import trace_logdet_quad_form_factory
import math


def test_trace_logdet_quad_form_factory():
    x = Variable(torch.linspace(0, 1, 10))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar)
    covar_module.initialize_interpolation_grid(4)
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

    actual_c_grad = c.grad.data
    actual_mu_diff_grad = mu_diff.grad.data
    actual_U_grad = U.grad.data

    c.grad.data.fill_(0)
    mu_diff.grad.data.fill_(0)
    U.grad.data.fill_(0)

    def _mm_closure_factory(*args):
        c, = args
        return lambda mat2: utils.toeplitz.sym_toeplitz_mm(c, mat2)

    def _derivative_quadratic_form_factory(*args):
        return lambda left_vector, right_vector: (sym_toeplitz_derivative_quadratic_form(left_vector, right_vector),)

    covar_args = (c,)

    res = trace_logdet_quad_form_factory(_mm_closure_factory,
                                         _derivative_quadratic_form_factory)(num_samples=1000)(mu_diff, U, *covar_args)
    res.backward()

    res_c_grad = c.grad.data
    res_mu_diff_grad = mu_diff.grad.data
    res_U_grad = U.grad.data

    assert all(torch.abs((res.data - actual.data) / actual.data) < 0.15)
    assert utils.approx_equal(res_c_grad, actual_c_grad)
    assert utils.approx_equal(res_mu_diff_grad, actual_mu_diff_grad)
    assert utils.approx_equal(res_U_grad, actual_U_grad)


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
