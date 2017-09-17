import math
import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable


def make_mul_lazy_var():
    diag = Variable(torch.Tensor([1]), requires_grad=True)
    c1 = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    t1 = ToeplitzLazyVariable(c1)
    c2 = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    t2 = KroneckerProductLazyVariable(c2)
    return (t1 * t2).add_diag(diag), diag


t1, t2 = make_mul_lazy_var()[0].lazy_vars
added_diag = make_mul_lazy_var()[0].added_diag.data
t1_eval = t1.evaluate().data
t2_eval = t2.evaluate().data


def test_add_diag():
    gpytorch.functions.fastest = False
    lazy_var = make_mul_lazy_var()[0]
    assert torch.equal(lazy_var.evaluate().data, (t1_eval * t2_eval + added_diag.diag()))


def test_add_jitter():
    lazy_var = make_mul_lazy_var()[0].add_jitter()
    assert torch.max(torch.abs(lazy_var.evaluate().data - (t1_eval * t2_eval + added_diag.diag()))) < 1e-1


def test_inv_matmul():
    gpytorch.functions.fastest = False
    mat = torch.randn(4, 4)
    res = make_mul_lazy_var()[0].inv_matmul(Variable(mat))
    assert torch.norm(res.data - (t1_eval.mul(t2_eval) + added_diag.diag()).inverse().matmul(mat)) < 1e-3


def test_matmul():
    gpytorch.functions.fastest = False
    mat = torch.randn(4, 4)
    res = make_mul_lazy_var()[0].matmul(Variable(mat))
    assert torch.norm(res.data - (t1_eval.mul(t2_eval) + added_diag.diag()).matmul(mat)) < 1e-3


def test_exact_gp_mll():

    gpytorch.functions.fastest = False
    labels_var = Variable(torch.arange(1, 5, 1))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    diag_var = Variable(torch.Tensor([1]), requires_grad=True)
    diag_var_expand = diag_var.expand(4)
    toeplitz = ToeplitzLazyVariable(c1_var).evaluate()
    kronecker_product = KroneckerProductLazyVariable(c2_var).evaluate()
    actual = toeplitz.mul(kronecker_product) + diag_var_expand.diag()

    # Actual case
    mul_lv, diag = make_mul_lazy_var()
    t1, t2 = mul_lv.lazy_vars

    # Test forward
    mll_res = mul_lv.exact_gp_marginal_log_likelihood(labels_var)
    mll_actual = gpytorch.exact_gp_marginal_log_likelihood(actual, labels_var)
    assert(math.fabs(mll_res.data.squeeze()[0] - mll_actual.data.squeeze()[0]) < 5e-1)
    # Test backwards
    mll_res.backward()
    mll_actual.backward()

    assert((c1_var.grad.data - t1.c.grad.data).abs().norm() / c1_var.grad.data.abs().norm() < 1e-1)
    assert((c2_var.grad.data - t2.columns.grad.data).abs().norm() / c2_var.grad.data.abs().norm() < 1e-1)
    assert((diag_var.grad.data - diag.grad.data).abs().norm() / diag_var.grad.data.abs().norm() < 1e-1)


def test_trace_log_det_quad_form():
    gpytorch.functions.fastest = False
    mu_diffs_var = Variable(torch.arange(1, 5, 1))
    chol_covar_1_var = Variable(torch.eye(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    diag_var = Variable(torch.Tensor([1]), requires_grad=True)
    diag_var_expand = diag_var.expand(4)
    toeplitz = ToeplitzLazyVariable(c1_var).evaluate()
    kronecker_product = KroneckerProductLazyVariable(c2_var).evaluate()
    actual = toeplitz.mul(kronecker_product) + diag_var_expand.diag()

    # Actual case
    mul_lv, diag = make_mul_lazy_var()
    t1, t2 = mul_lv.lazy_vars

    # Test forward
    tldqf_res = mul_lv.trace_log_det_quad_form(mu_diffs_var, chol_covar_1_var)
    tldqf_actual = gpytorch._trace_logdet_quad_form_factory_class()(mu_diffs_var, chol_covar_1_var, actual)
    assert(math.fabs(tldqf_res.data.squeeze()[0] - tldqf_actual.data.squeeze()[0]) < 1)

    # Test backwards
    tldqf_res.backward()
    tldqf_actual.backward()
    assert((c1_var.grad.data - t1.c.grad.data).abs().norm() / c1_var.grad.data.abs().norm() < 1e-1)
    assert((c2_var.grad.data - t2.columns.grad.data).abs().norm() / c2_var.grad.data.abs().norm() < 1e-1)
    assert((diag_var.grad.data - diag.grad.data).abs().norm() / diag_var.grad.data.abs().norm() < 1e-1)


def test_getitem():
    gpytorch.functions.fastest = False
    res = make_mul_lazy_var()[0][1, 1]
    assert torch.norm(res.evaluate().data - (t1_eval.mul(t2_eval) + torch.ones(4))[1, 1]) < 1e-3


def test_exact_posterior():
    gpytorch.functions.fastest = False
    train_mean = Variable(torch.randn(4))
    train_y = Variable(torch.randn(4))
    test_mean = Variable(torch.randn(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    indices_1 = torch.arange(0, 4).long().view(4, 1)
    values_1 = torch.ones(4).view(4, 1)
    indices_2 = torch.arange(0, 2).expand(4, 2).long().view(2, 4, 1)
    values_2 = torch.ones(8).view(2, 4, 1)
    toeplitz = ToeplitzLazyVariable(c1_var, indices_1, values_1, indices_1, values_1)
    kronecker_product = KroneckerProductLazyVariable(c2_var, indices_2, values_2, indices_2, values_2)
    mul_lv = toeplitz * kronecker_product

    # Actual case
    actual = mul_lv.evaluate()
    # Test forward
    actual_alpha = gpytorch.posterior_strategy(actual).exact_posterior_alpha(train_mean, train_y)
    actual_mean = gpytorch.posterior_strategy(actual).exact_posterior_mean(test_mean, actual_alpha)
    mul_lv_alpha = mul_lv.posterior_strategy().exact_posterior_alpha(train_mean, train_y)
    mul_lv_mean = mul_lv.posterior_strategy().exact_posterior_mean(test_mean, mul_lv_alpha)
    assert(torch.norm(actual_mean.data - mul_lv_mean.data) < 1e-4)
