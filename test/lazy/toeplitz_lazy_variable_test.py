import torch
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import ObservationModel
from gpytorch.random_variables import GaussianRandomVariable
from gpytorch.parameters import MLEParameterGroup, BoundedParameter

x = Variable(torch.linspace(0, 1, 51))


class Model(ObservationModel):
    def __init__(self):
        super(Model, self).__init__(GaussianLikelihood())
        self.mean_module = ConstantMean()
        covar_module = RBFKernel()
        self.grid_covar_module = GridInterpolationKernel(covar_module, 50)
        self.params = MLEParameterGroup(
            log_lengthscale=BoundedParameter(torch.Tensor([0]), -3, 5),
        )

    def forward(self, x):
        mean_x = self.mean_module(x, constant=Variable(torch.Tensor([0])))
        covar_x = self.grid_covar_module(x, log_lengthscale=self.params.log_lengthscale)

        latent_pred = GaussianRandomVariable(mean_x, covar_x)
        return latent_pred, Variable(torch.Tensor([-5]))


prior_observation_model = Model()
pred = prior_observation_model(x)
lazy_toeplitz_var = pred.covar()
T = utils.toeplitz(lazy_toeplitz_var.c.data, lazy_toeplitz_var.r.data)
W_left = utils.index_coef_to_sparse(lazy_toeplitz_var.J_left, lazy_toeplitz_var.C_left, len(lazy_toeplitz_var.c))
W_right = utils.index_coef_to_sparse(lazy_toeplitz_var.J_right, lazy_toeplitz_var.C_right, len(lazy_toeplitz_var.c))
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
