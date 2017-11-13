import torch
import gpytorch
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.lazy import KroneckerProductLazyVariable
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.random_variables import GaussianRandomVariable
from gpytorch.utils.kronecker_product import kronecker_product, list_of_indices_and_values_to_sparse

x = torch.zeros(2, 11)
x[0] = torch.linspace(0, 1, 11)
x[1] = torch.linspace(0, 0.95, 11)
x = Variable(x.t())


class Model(gpytorch.GridInducingPointModule):
    def __init__(self):
        super(Model, self).__init__(grid_size=10, grid_bounds=[(0, 1), (0, 1)])
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.covar_module = RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x).add_diag(Variable(torch.Tensor([1e-2])))
        return GaussianRandomVariable(mean_x, covar_x)


prior_observation_model = Model()
prior_observation_model.eval()
pred = prior_observation_model(x)
lazy_kronecker_product_var = pred.covar().add_diag(Variable(torch.Tensor([1e-2])))
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


def test_evaluate():
    WKW_res = lazy_kronecker_product_var.evaluate()
    assert utils.approx_equal(WKW_res, WKW)


def test_diag():
    diag_actual = torch.diag(WKW)
    diag_res = lazy_kronecker_product_var.diag()
    assert utils.approx_equal(diag_res.data, diag_actual)


def test_get_item_on_interpolated_variable_no_diagonal():
    no_diag_kronecker_product = KroneckerProductLazyVariable(lazy_kronecker_product_var.columns,
                                                             lazy_kronecker_product_var.J_lefts,
                                                             lazy_kronecker_product_var.C_lefts,
                                                             lazy_kronecker_product_var.J_rights,
                                                             lazy_kronecker_product_var.C_rights)
    evaluated = no_diag_kronecker_product.evaluate().data
    assert utils.approx_equal(no_diag_kronecker_product[4:6].evaluate().data, evaluated[4:6])
    assert utils.approx_equal(no_diag_kronecker_product[4:6, 2:6].evaluate().data, evaluated[4:6, 2:6])


def test_get_item_square_on_interpolated_variable():
    assert utils.approx_equal(lazy_kronecker_product_var[4:6, 4:6].evaluate().data, WKW[4:6, 4:6])


def test_get_item_square_on_variable():
    kronecker_product_var = KroneckerProductLazyVariable(Variable(torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])),
                                                         added_diag=Variable(torch.ones(16) * 3))
    evaluated = kronecker_product_var.evaluate().data

    assert utils.approx_equal(kronecker_product_var[2:4, 2:4].evaluate().data, evaluated[2:4, 2:4])
