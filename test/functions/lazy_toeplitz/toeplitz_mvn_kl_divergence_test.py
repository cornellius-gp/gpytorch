import torch
import gpytorch
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from torch.autograd import Variable
import math


def test_toeplitz_mvn_kl_divergence_forward():
    x = Variable(torch.linspace(0, 1, 5))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar)
    covar_module.initialize_interpolation_grid(10)
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))

    c = Variable(covar_x.c.data, requires_grad=True)
    mu1 = Variable(torch.randn(10), requires_grad=True)
    mu2 = Variable(torch.randn(10), requires_grad=True)

    T = Variable(torch.zeros(len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            T[i, j] = utils.toeplitz.toeplitz_getitem(c, c, i, j)

    U = torch.randn(10, 10).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    actual = gpytorch.mvn_kl_divergence(mu1, U, mu2, T, num_samples=1000)

    res = gpytorch.mvn_kl_divergence(mu1, U, mu2, covar_x, num_samples=1000)

    assert all(torch.abs((res.data - actual.data) / actual.data) < 0.15)


def test_toeplitz_mvn_kl_divergence_backward():
    x = Variable(torch.linspace(0, 1, 5))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar)
    covar_module.initialize_interpolation_grid(4)
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))
    covar_x.c = Variable(covar_x.c.data, requires_grad=True)

    c = covar_x.c
    mu1 = Variable(torch.randn(4), requires_grad=True)
    mu2 = Variable(torch.randn(4), requires_grad=True)

    mu_diff = mu2 - mu1

    T = Variable(torch.zeros(len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            T[i, j] = utils.toeplitz.toeplitz_getitem(c, c, i, j)

    U = torch.randn(4, 4).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    actual = 0.5 * (_det(T).log() +
                    mu_diff.dot(T.inverse().mv(mu_diff)) +
                    T.inverse().mm(U.t().mm(U)).trace() -
                    U.diag().log().sum(0) * 2 - len(mu_diff))
    actual.backward()

    actual_c_grad = c.grad.data.clone()
    actual_mu1_grad = mu1.grad.data.clone()
    actual_mu2_grad = mu2.grad.data.clone()
    actual_U_grad = U.grad.data.clone()

    c.grad.data.fill_(0)
    mu1.grad.data.fill_(0)
    mu2.grad.data.fill_(0)
    U.grad.data.fill_(0)

    res = gpytorch.mvn_kl_divergence(mu1, U, mu2, covar_x, num_samples=1000)
    res.backward()

    res_c_grad = c.grad.data
    res_mu1_grad = mu1.grad.data
    res_mu2_grad = mu2.grad.data
    res_U_grad = U.grad.data

    assert torch.abs((res_c_grad - actual_c_grad)).sum() / actual_c_grad.abs().sum() < 1e-1
    assert torch.abs((res_mu1_grad - actual_mu1_grad)).sum() / actual_mu1_grad.abs().sum() < 1e-5
    assert torch.abs((res_mu2_grad - actual_mu2_grad)).sum() / actual_mu2_grad.abs().sum() < 1e-5
    assert torch.abs((res_U_grad - actual_U_grad)).sum() / actual_U_grad.abs().sum() < 1e-2


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
