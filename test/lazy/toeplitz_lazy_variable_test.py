import torch
import gpytorch
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.lazy import ToeplitzLazyVariable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

x = Variable(torch.linspace(0, 1, 51))


class Model(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
        super(Model, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        covar_module = RBFKernel()
        self.grid_covar_module = GridInterpolationKernel(covar_module)
        self.initialize_interpolation_grid(50, grid_bounds=[(0, 1)])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.grid_covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


prior_observation_model = Model()
pred = prior_observation_model(x)
lazy_toeplitz_var_no_diag = prior_observation_model.grid_covar_module(x)
lazy_toeplitz_var = pred.covar()
T = utils.toeplitz.sym_toeplitz(lazy_toeplitz_var.c.data)
W_left = utils.toeplitz.index_coef_to_sparse(lazy_toeplitz_var.J_left,
                                             lazy_toeplitz_var.C_left,
                                             len(lazy_toeplitz_var.c))
W_right = utils.toeplitz.index_coef_to_sparse(lazy_toeplitz_var.J_right,
                                              lazy_toeplitz_var.C_right,
                                              len(lazy_toeplitz_var.c))
WTW = torch.dsmm(W_right, torch.dsmm(W_left, T).t()) + torch.diag(lazy_toeplitz_var.added_diag.data)


def test_explicit_interpolate_T():
    WT_res = lazy_toeplitz_var.explicit_interpolate_T(lazy_toeplitz_var.J_left, lazy_toeplitz_var.C_left)
    WT_actual = torch.dsmm(W_left, T)
    assert utils.approx_equal(WT_res.data, WT_actual)


def test_evaluate():
    WTW_res = lazy_toeplitz_var.evaluate()
    assert utils.approx_equal(WTW_res, WTW)


def test_diag():
    diag_actual = torch.diag(WTW)
    diag_res = lazy_toeplitz_var.diag()
    assert utils.approx_equal(diag_res.data, diag_actual)


def test_mul_constant():
    product_var = lazy_toeplitz_var * 2.5
    assert utils.approx_equal(product_var.c, lazy_toeplitz_var.c * 2.5)


def test_mul_other_toeplitz_lazy_variable():
    product_var = lazy_toeplitz_var_no_diag * lazy_toeplitz_var_no_diag
    assert utils.approx_equal(product_var.c, lazy_toeplitz_var_no_diag.c * lazy_toeplitz_var_no_diag.c)


def test_get_item_on_interpolated_variable_no_diagonal():
    no_diag_toeplitz = ToeplitzLazyVariable(lazy_toeplitz_var.c, lazy_toeplitz_var.J_left, lazy_toeplitz_var.C_left,
                                            lazy_toeplitz_var.J_right, lazy_toeplitz_var.C_right)
    evaluated = no_diag_toeplitz.evaluate().data

    assert utils.approx_equal(no_diag_toeplitz[4:6].evaluate().data, evaluated[4:6])
    assert utils.approx_equal(no_diag_toeplitz[4:6, 2:6].evaluate().data, evaluated[4:6, 2:6])


def test_get_item_square_on_interpolated_variable():
    assert utils.approx_equal(lazy_toeplitz_var[4:6, 4:6].evaluate().data, WTW[4:6, 4:6])


def test_get_item_square_on_variable():
    toeplitz_var = ToeplitzLazyVariable(Variable(torch.Tensor([1, 2, 3, 4])),
                                        added_diag=Variable(torch.ones(4) * 3))
    evaluated = toeplitz_var.evaluate().data

    assert utils.approx_equal(toeplitz_var[2:4, 2:4].evaluate().data, evaluated[2:4, 2:4])
