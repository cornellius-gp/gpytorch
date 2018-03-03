import math
import torch
import gpytorch
from gpytorch.lazy import ToeplitzLazyVariable
from torch.autograd import Variable


def test_exact_gp_mll():
    labels_var = Variable(torch.randn(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([12.5, 2.5, 5, 0]), requires_grad=True)
    toeplitz_lazy_var = ToeplitzLazyVariable(c1_var) * 2.5
    actual = ToeplitzLazyVariable(c2_var)

    # Test forward
    with gpytorch.settings.num_trace_samples(1000):
        mll_res = toeplitz_lazy_var.exact_gp_marginal_log_likelihood(labels_var)
        mll_actual = actual.exact_gp_marginal_log_likelihood(labels_var)

    # Test backwards
    mll_res.backward()
    mll_actual.backward()

    assert(math.fabs(mll_res.data.squeeze()[0] - mll_actual.data.squeeze()[0]) < 6e-1)
    assert(math.fabs(c1_var.grad.data[0] - c2_var.grad.data[0]) < 1)


def test_getitem():
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([12.5, 2.5, 5, 0]), requires_grad=True)
    toeplitz_lazy_var = ToeplitzLazyVariable(c1_var) * 2.5
    actual = ToeplitzLazyVariable(c2_var)

    assert torch.norm(actual[2:, 2:].evaluate().data - toeplitz_lazy_var[2:, 2:].evaluate().data) < 1e-3
