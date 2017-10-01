import math
import torch
import gpytorch
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable, KroneckerProductLazyVariable, MulLazyVariable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable


def make_mul_lazy_var():
    diag = Variable(torch.Tensor([1]), requires_grad=True)
    c1 = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    t1 = ToeplitzLazyVariable(c1)
    c2 = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    t2 = KroneckerProductLazyVariable(c2)
    c3 = Variable(torch.Tensor([7, 2, 1, 0]), requires_grad=True)
    t3 = ToeplitzLazyVariable(c3)
    return (t1 * t2 * t3).add_diag(diag), diag


t1, t2, t3 = make_mul_lazy_var()[0].lazy_vars
added_diag = make_mul_lazy_var()[0].added_diag.data
t1_eval = t1.evaluate().data
t2_eval = t2.evaluate().data
t3_eval = t3.evaluate().data
t1_t2_t3_eval = t1_eval * t2_eval * t3_eval


def test_add_diag():
    lazy_var = make_mul_lazy_var()[0]
    assert torch.equal(lazy_var.evaluate().data, (t1_t2_t3_eval + added_diag.diag()))


def test_add_jitter():
    lazy_var = make_mul_lazy_var()[0].add_jitter()
    assert torch.max(torch.abs(lazy_var.evaluate().data - (t1_t2_t3_eval + added_diag.diag()))) < 1e-1


def test_inv_matmul():
    mat = torch.randn(4, 4)
    res = make_mul_lazy_var()[0].inv_matmul(Variable(mat))
    assert torch.norm(res.data - (t1_t2_t3_eval + added_diag.diag()).inverse().matmul(mat)) < 1e-3


def test_matmul_deterministic():
    mat = torch.randn(4, 4)
    res = make_mul_lazy_var()[0].matmul(Variable(mat))
    assert torch.norm(res.data - (t1_t2_t3_eval + added_diag.diag()).matmul(mat)) < 1e-3


def test_matmul_approx():
    class KissGPModel(gpytorch.GPModel):
        def __init__(self):
            likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
            super(KissGPModel, self).__init__(likelihood)
            self.mean_module = ConstantMean(constant_bounds=(-1, 1))
            covar_module = RBFKernel(log_lengthscale_bounds=(-100, 100))
            covar_module.log_lengthscale.data = torch.FloatTensor([-2])
            self.grid_covar_module = GridInterpolationKernel(covar_module)
            self.initialize_interpolation_grid(300, grid_bounds=[(0, 1)])

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.grid_covar_module(x)
            return GaussianRandomVariable(mean_x, covar_x)

    model = KissGPModel()

    n = 100
    d = 4

    lazy_var_list = []
    lazy_var_eval_list = []

    for i in range(d):
        x = Variable(torch.rand(n))
        y = Variable(torch.rand(n))
        model.condition(x, y)
        toeplitz_var = model.forward(x).covar()
        lazy_var_list.append(toeplitz_var)
        lazy_var_eval_list.append(toeplitz_var.evaluate().data)

    mul_lazy_var = MulLazyVariable(*lazy_var_list, matmul_mode='approximate', max_iter=30)
    mul_lazy_var_eval = torch.ones(n, n)
    for i in range(d):
        mul_lazy_var_eval *= (lazy_var_eval_list[i].matmul(torch.eye(lazy_var_eval_list[i].size()[0])))

    vec = torch.randn(n)

    actual = mul_lazy_var_eval.matmul(vec)
    res = mul_lazy_var.matmul(Variable(vec)).data

    assert torch.norm(actual - res) / torch.norm(actual) < 1e-2


def test_exact_gp_mll():
    labels_var = Variable(torch.arange(1, 5, 1))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    c3_var = Variable(torch.Tensor([7, 2, 1, 0]), requires_grad=True)
    diag_var = Variable(torch.Tensor([1]), requires_grad=True)
    diag_var_expand = diag_var.expand(4)
    toeplitz_1 = ToeplitzLazyVariable(c1_var).evaluate()
    kronecker_product = KroneckerProductLazyVariable(c2_var).evaluate()
    toeplitz_2 = ToeplitzLazyVariable(c3_var).evaluate()
    actual = toeplitz_1 * kronecker_product * toeplitz_2 + diag_var_expand.diag()

    # Actual case
    mul_lv, diag = make_mul_lazy_var()
    t1, t2, t3 = mul_lv.lazy_vars

    # Test forward
    mll_res = mul_lv.exact_gp_marginal_log_likelihood(labels_var)
    mll_actual = gpytorch.exact_gp_marginal_log_likelihood(actual, labels_var)
    assert(math.fabs(mll_res.data.squeeze()[0] - mll_actual.data.squeeze()[0]) < 1)
    # Test backwards
    mll_res.backward()
    mll_actual.backward()

    assert((c1_var.grad.data - t1.c.grad.data).abs().norm() / c1_var.grad.data.abs().norm() < 1e-1)
    assert((c2_var.grad.data - t2.columns.grad.data).abs().norm() / c2_var.grad.data.abs().norm() < 1e-1)
    assert((c3_var.grad.data - t3.c.grad.data).abs().norm() / c3_var.grad.data.abs().norm() < 1e-1)
    assert((diag_var.grad.data - diag.grad.data).abs().norm() / diag_var.grad.data.abs().norm() < 1e-1)


def test_trace_log_det_quad_form():
    mu_diffs_var = Variable(torch.arange(1, 5, 1))
    chol_covar_1_var = Variable(torch.eye(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    c3_var = Variable(torch.Tensor([7, 2, 1, 0]), requires_grad=True)
    diag_var = Variable(torch.Tensor([1]), requires_grad=True)
    diag_var_expand = diag_var.expand(4)
    toeplitz_1 = ToeplitzLazyVariable(c1_var).evaluate()
    kronecker_product = KroneckerProductLazyVariable(c2_var).evaluate()
    toeplitz_2 = ToeplitzLazyVariable(c3_var).evaluate()
    actual = toeplitz_1 * kronecker_product * toeplitz_2 + diag_var_expand.diag()

    # Actual case
    mul_lv, diag = make_mul_lazy_var()
    t1, t2, t3 = mul_lv.lazy_vars

    # Test forward
    tldqf_res = mul_lv.trace_log_det_quad_form(mu_diffs_var, chol_covar_1_var)
    tldqf_actual = gpytorch._trace_logdet_quad_form_factory_class()(mu_diffs_var, chol_covar_1_var, actual)
    assert(math.fabs(tldqf_res.data.squeeze()[0] - tldqf_actual.data.squeeze()[0]) < 1.5)

    # Test backwards
    tldqf_res.backward()
    tldqf_actual.backward()
    assert((c1_var.grad.data - t1.c.grad.data).abs().norm() / c1_var.grad.data.abs().norm() < 1e-1)
    assert((c2_var.grad.data - t2.columns.grad.data).abs().norm() / c2_var.grad.data.abs().norm() < 1e-1)
    assert((c3_var.grad.data - t3.c.grad.data).abs().norm() / c3_var.grad.data.abs().norm() < 1e-1)
    assert((diag_var.grad.data - diag.grad.data).abs().norm() / diag_var.grad.data.abs().norm() < 1e-1)


def test_getitem():
    res = make_mul_lazy_var()[0][1, 1]
    assert torch.norm(res.evaluate().data - (t1_t2_t3_eval + torch.ones(4))[1, 1]) < 1e-3


def test_exact_posterior():
    train_mean = Variable(torch.randn(4))
    train_y = Variable(torch.randn(4))
    test_mean = Variable(torch.randn(4))

    # Test case
    c1_var = Variable(torch.Tensor([5, 1, 2, 0]), requires_grad=True)
    c2_var = Variable(torch.Tensor([[6, 0], [1, -1]]), requires_grad=True)
    c3_var = Variable(torch.Tensor([7, 2, 1, 0]), requires_grad=True)
    indices_1 = torch.arange(0, 4).long().view(4, 1)
    values_1 = torch.ones(4).view(4, 1)
    indices_2 = torch.arange(0, 2).expand(4, 2).long().view(2, 4, 1)
    values_2 = torch.ones(8).view(2, 4, 1)
    indices_3 = torch.arange(0, 4).long().view(4, 1)
    values_3 = torch.ones(4).view(4, 1)
    toeplitz_1 = ToeplitzLazyVariable(c1_var, indices_1, values_1, indices_1, values_1)
    kronecker_product = KroneckerProductLazyVariable(c2_var, indices_2, values_2, indices_2, values_2)
    toeplitz_2 = ToeplitzLazyVariable(c3_var, indices_3, values_3, indices_3, values_3)
    mul_lv = toeplitz_1 * kronecker_product * toeplitz_2

    # Actual case
    actual = mul_lv.evaluate()
    # Test forward
    actual_alpha = gpytorch.posterior_strategy(actual).exact_posterior_alpha(train_mean, train_y)
    actual_mean = gpytorch.posterior_strategy(actual).exact_posterior_mean(test_mean, actual_alpha)
    mul_lv_alpha = mul_lv.posterior_strategy().exact_posterior_alpha(train_mean, train_y)
    mul_lv_mean = mul_lv.posterior_strategy().exact_posterior_mean(test_mean, mul_lv_alpha)
    assert(torch.norm(actual_mean.data - mul_lv_mean.data) < 1e-3)
