import torch
import gpytorch
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
from gpytorch.utils.kronecker_product import kronecker_product, list_of_indices_and_values_to_sparse

x = torch.zeros(2, 11)
x[0] = torch.linspace(0, 1, 11)
x[1] = torch.linspace(0, 0.95, 11)
x = Variable(x.t())

class Model(gpytorch.GPModel):
    def __init__(self):
        likelihood = GaussianLikelihood(log_noise_bounds=(-3, 3))
        super(Model, self).__init__(likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        covar_module = RBFKernel()
        self.grid_covar_module = GridInterpolationKernel(covar_module)
        self.initialize_interpolation_grid(10)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.grid_covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


prior_observation_model = Model()
pred = prior_observation_model(x)
lazy_kronecker_product_var = pred.covar()
Ts = torch.zeros(lazy_kronecker_product_var.columns.size()[0],
                 lazy_kronecker_product_var.columns.size()[1],
                 lazy_kronecker_product_var.columns.size()[1])
for i in range(lazy_kronecker_product_var.columns.size()[0]):
    Ts[i] = utils.toeplitz.sym_toeplitz(lazy_kronecker_product_var.columns[i].data)
K = kronecker_product(Ts)
W_left = list_of_indices_and_values_to_sparse(lazy_kronecker_product_var.J_lefts,
                                              lazy_kronecker_product_var.C_lefts,
                                              lazy_kronecker_product_var.columns)
W_right = list_of_indices_and_values_to_sparse(lazy_kronecker_product_var.J_rights,
                                               lazy_kronecker_product_var.C_rights,
                                               lazy_kronecker_product_var.columns)
WKW = torch.dsmm(W_right, torch.dsmm(W_left, K).t()) + torch.diag(lazy_kronecker_product_var.added_diag.data)


def test_explicit_interpolate_K():
    WK_res = lazy_kronecker_product_var.explicit_interpolate_K(lazy_kronecker_product_var.J_lefts,
                                                               lazy_kronecker_product_var.C_lefts)
    WK_actual = torch.dsmm(W_left, K)
    assert utils.approx_equal(WK_res.data, WK_actual)


def test_evaluate():
    WKW_res = lazy_kronecker_product_var.evaluate()
    assert utils.approx_equal(WKW_res, WKW)


# def test_diag():
#     diag_actual = torch.diag(WKW)
#     diag_res = lazy_toeplitz_var.diag()
#     assert utils.approx_equal(diag_res.data, diag_actual)
