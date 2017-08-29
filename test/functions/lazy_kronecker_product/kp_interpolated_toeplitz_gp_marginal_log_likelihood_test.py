import torch
import math
from gpytorch.functions.lazy_kronecker_product import KPInterpolatedToeplitzGPMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.utils.kronecker_product import kronecker_product, list_of_indices_and_values_to_sparse


def test_kp_toeplitz_gp_marginal_log_likelihood_forward():
    x = []
    x.append(Variable(torch.linspace(0, 1, 2)))
    x.append(Variable(torch.linspace(0, 1, 2)))
    x.append(Variable(torch.linspace(0, 1, 2)))
    y = torch.randn(2)
    noise = torch.Tensor([1e-4])
    rbf_module = RBFKernel()
    rbf_module.initialize(log_lengthscale=-2)
    covar_module = GridInterpolationKernel(rbf_module)
    covar_module.initialize_interpolation_grid(5, [(0, 1), (0, 1), (0, 1)])

    covar_xs = []
    cs = torch.zeros(3, covar_module.forward(x[0].unsqueeze(1), x[0].unsqueeze(1)).c.data.size()[0])
    J_lefts = []
    C_lefts = []
    J_rights = []
    C_rights = []
    Ts = []
    for i in range(3):
        covar_xs.append(covar_module.forward(x[i].unsqueeze(1), x[i].unsqueeze(1)))
        cs[i] = covar_xs[i].c.data
        J_lefts.append(covar_xs[i].J_left)
        C_lefts.append(covar_xs[i].C_left)
        J_rights.append(covar_xs[i].J_right)
        C_rights.append(covar_xs[i].C_right)
        Ts.append(utils.toeplitz.sym_toeplitz(cs[i]))

    W_left = list_of_indices_and_values_to_sparse(J_lefts, C_lefts, cs)
    W_right = list_of_indices_and_values_to_sparse(J_rights, C_rights, cs)

    W_left_dense = W_left.to_dense()
    W_right_dense = W_right.to_dense()

    K = kronecker_product(Ts)

    WKW = W_left_dense.matmul(K.matmul(W_right_dense.t())) + torch.eye(len(x[0])) * 1e-4

    quad_form_actual = y.dot(WKW.inverse().matmul(y))
    chol_K = torch.potrf(WKW)
    log_det_actual = chol_K.diag().log().sum() * 2

    actual = -0.5 * (log_det_actual + quad_form_actual + math.log(2 * math.pi) * len(y))

    res = KPInterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right, num_samples=1000)(Variable(cs),
                                                                                           Variable(y),
                                                                                           Variable(noise)).data
    assert all(torch.abs((res - actual) / actual) < 0.05)


def test_kp_toeplitz_gp_marginal_log_likelihood_backward():
    x = []
    x.append(Variable(torch.linspace(0, 1, 2)))
    x.append(Variable(torch.linspace(0, 1, 2)))
    y = Variable(torch.randn(2), requires_grad=True)

    rbf_module = RBFKernel()
    rbf_module.initialize(log_lengthscale=-2)
    covar_module = GridInterpolationKernel(rbf_module)
    covar_module.initialize_interpolation_grid(5, [(0, 1), (0, 1)])
    noise = Variable(torch.Tensor([1e-4]), requires_grad=True)

    covar_xs = []
    m = covar_module.forward(x[0].unsqueeze(1), x[0].unsqueeze(1)).c.data.size()[0]
    cs = Variable(torch.zeros(2, m), requires_grad=True)

    J_lefts = []
    C_lefts = []
    J_rights = []
    C_rights = []
    Ts = []
    for i in range(2):
        covar_xs.append(covar_module.forward(x[i].unsqueeze(1), x[i].unsqueeze(1)))
        cs.data[i] = covar_xs[i].c.data
        J_lefts.append(covar_xs[i].J_left)
        C_lefts.append(covar_xs[i].C_left)
        J_rights.append(covar_xs[i].J_right)
        C_rights.append(covar_xs[i].C_right)
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

    WKW = W_left_dense.matmul(K.matmul(W_right_dense.t())) + Variable(torch.eye(len(y))) * noise

    quad_form_actual = y.dot(WKW.inverse().matmul(y))
    log_det_actual = _det(WKW).log()

    actual_nll = -0.5 * (log_det_actual + quad_form_actual + math.log(2 * math.pi) * len(y))
    actual_nll.backward()

    actual_c_grad = cs.grad.data.clone()
    actual_y_grad = y.grad.data.clone()
    actual_noise_grad = noise.grad.data.clone()

    cs.grad.data.fill_(0)
    y.grad.data.fill_(0)
    noise.grad.data.fill_(0)

    res = KPInterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right, num_samples=1000)(cs, y, noise)
    res.backward()

    res_c_grad = cs.grad.data
    res_y_grad = y.grad.data
    res_noise_grad = noise.grad.data

    assert torch.abs(actual_c_grad - res_c_grad).mean() < 1e-3
    assert torch.abs(actual_y_grad - res_y_grad).mean() < 1e-4
    assert torch.abs(actual_noise_grad - res_noise_grad).mean() < 1e-4


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
