import torch
import gpytorch
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from torch.autograd import Variable


def test_toeplitz_mvn_kl_divergence_forward():
    x = Variable(torch.linspace(0, 1, 5))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, 10)
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))

    c = Variable(covar_x.c.data, requires_grad=True)
    mu1 = Variable(torch.randn(12), requires_grad=True)
    mu2 = Variable(torch.randn(12), requires_grad=True)

    T = Variable(torch.zeros(len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            T[i, j] = utils.toeplitz.toeplitz_getitem(c, c, i, j)

    U = torch.randn(12, 12).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    actual = gpytorch.mvn_kl_divergence(mu1, U, mu2, T, num_samples=1000)

    res = gpytorch.mvn_kl_divergence(mu1, U, mu2, covar_x, num_samples=1000)

    assert all(torch.abs((res.data - actual.data) / actual.data) < 0.15)


def test_toeplitz_mvn_kl_divergence_backward():
    x = Variable(torch.linspace(0, 1, 5))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, 10)
    covar_x = covar_module.forward(x.unsqueeze(1), x.unsqueeze(1))
    covar_x.c = Variable(covar_x.c.data, requires_grad=True)

    c = covar_x.c
    mu1 = Variable(torch.randn(12), requires_grad=True)
    mu2 = Variable(torch.randn(12), requires_grad=True)

    T = Variable(torch.zeros(len(c), len(c)))
    for i in range(len(c)):
        for j in range(len(c)):
            T[i, j] = utils.toeplitz.toeplitz_getitem(c, c, i, j)

    U = torch.randn(12, 12).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    actual = gpytorch.mvn_kl_divergence(mu1, U, mu2, T)
    actual.backward()

    actual_c_grad = c.grad.data
    actual_mu1_grad = mu1.grad.data
    actual_mu2_grad = mu2.grad.data
    actual_U_grad = U.grad.data

    c.grad.data.fill_(0)
    mu1.grad.data.fill_(0)
    mu2.grad.data.fill_(0)
    U.grad.data.fill_(0)

    res = gpytorch.mvn_kl_divergence(mu1, U, mu2, covar_x)
    res.backward()

    res_c_grad = c.grad.data
    res_mu1_grad = mu1.grad.data
    res_mu2_grad = mu2.grad.data
    res_U_grad = U.grad.data

    assert utils.approx_equal(res_c_grad, actual_c_grad)
    assert utils.approx_equal(res_mu1_grad, actual_mu1_grad)
    assert utils.approx_equal(res_mu2_grad, actual_mu2_grad)
    assert utils.approx_equal(res_U_grad, actual_U_grad)
