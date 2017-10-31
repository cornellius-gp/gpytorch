import math
import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable, InterpolatedLazyVariable


def make_sum_lazy_var():
    c1 = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    t1 = ToeplitzLazyVariable(c1)
    c2 = Variable(torch.Tensor([6, 0, 1, -1]), requires_grad=True)
    t2 = ToeplitzLazyVariable(c2)
    return t1 + t2


t1, t2 = make_sum_lazy_var().lazy_vars
t1_eval = t1.evaluate().data
t2_eval = t2.evaluate().data


def test_add_diag():
    diag = Variable(torch.Tensor([4]))
    lazy_var = make_sum_lazy_var().add_diag(diag)
    assert torch.equal(lazy_var.evaluate().data, (t1_eval + t2_eval + torch.eye(4) * 4))


def test_add_jitter():
    lazy_var = make_sum_lazy_var().add_jitter()
    assert torch.max(torch.abs(lazy_var.evaluate().data - (t1_eval + t2_eval))) < 1e-1


def test_inv_matmul():
    mat = torch.randn(4, 4)
    res = make_sum_lazy_var().inv_matmul(Variable(mat))
    assert torch.norm(res.data - (t1_eval + t2_eval).inverse().matmul(mat)) < 1e-3


def test_exact_gp_mll():
    labels_var = Variable(torch.randn(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([6, 0, 1, -1]), requires_grad=True)
    actual = ToeplitzLazyVariable(c1_var + c2_var)

    # Actual case
    sum_lv = make_sum_lazy_var()
    t1, t2 = sum_lv.lazy_vars

    # Test forward
    mll_res = sum_lv.exact_gp_marginal_log_likelihood(labels_var)
    mll_actual = actual.exact_gp_marginal_log_likelihood(labels_var)
    assert(math.fabs(mll_res.data.squeeze()[0] - mll_actual.data.squeeze()[0]) < 5e-1)

    # Test backwards
    mll_res.backward()
    mll_actual.backward()
    assert(math.fabs(c1_var.grad.data[0] - t1.column.grad.data[0]) < 1e-1)
    assert(math.fabs(c2_var.grad.data[0] - t2.column.grad.data[0]) < 1e-1)


def test_trace_log_det_quad_form():
    mu_diffs_var = Variable(torch.randn(4))
    chol_covar_1_var = Variable(torch.eye(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([6, 0, 1, -1]), requires_grad=True)
    actual = ToeplitzLazyVariable(c1_var + c2_var)

    # Actual case
    sum_lv = make_sum_lazy_var()
    t1, t2 = sum_lv.lazy_vars

    # Test forward
    tldqf_res = sum_lv.trace_log_det_quad_form(mu_diffs_var, chol_covar_1_var)
    tldqf_actual = actual.trace_log_det_quad_form(mu_diffs_var, chol_covar_1_var)
    assert(math.fabs(tldqf_res.data.squeeze()[0] - tldqf_actual.data.squeeze()[0]) < 1.5)

    # Test backwards
    tldqf_res.backward()
    tldqf_actual.backward()
    assert(math.fabs(c1_var.grad.data[0] - t1.column.grad.data[0]) < 1e-1)
    assert(math.fabs(c2_var.grad.data[0] - t2.column.grad.data[0]) < 1e-1)


def test_getitem():
    res = make_sum_lazy_var()[1, 1]
    assert torch.norm(res.data - (t1_eval + t2_eval)[1, 1]) < 1e-3


def test_exact_posterior():
    train_mean = Variable(torch.randn(4))
    train_y = Variable(torch.randn(4))
    test_mean = Variable(torch.randn(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([6, 0, 1, -1]), requires_grad=True)
    indices = Variable(torch.arange(0, 4).long().view(4, 1))
    values = Variable(torch.ones(4).view(4, 1))
    toeplitz_1 = InterpolatedLazyVariable(ToeplitzLazyVariable(c1_var), indices, values, indices, values)
    toeplitz_2 = InterpolatedLazyVariable(ToeplitzLazyVariable(c2_var), indices, values, indices, values)
    sum_lv = toeplitz_1 + toeplitz_2

    # Actual case
    actual = sum_lv.evaluate()

    # Test forward
    actual_alpha = gpytorch.posterior_strategy(actual).exact_posterior_alpha(train_mean, train_y)
    actual_mean = gpytorch.posterior_strategy(actual).exact_posterior_mean(test_mean, actual_alpha)
    sum_lv_alpha = sum_lv.posterior_strategy().exact_posterior_alpha(train_mean, train_y)
    sum_lv_mean = sum_lv.posterior_strategy().exact_posterior_mean(test_mean, sum_lv_alpha)
    assert(torch.norm(actual_mean.data - sum_lv_mean.data) < 1e-4)
