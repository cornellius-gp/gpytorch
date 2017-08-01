import torch
from torch.autograd import Variable
import utils
from lazy import ToeplitzLazyVariable
from .random_variables import GaussianRandomVariable
from .distribution import Distribution
from .observation_model import ObservationModel
from .math.functions import AddDiag, ExactGPMarginalLogLikelihood, Invmm, \
    Invmv, NormalCDF, LogNormalCDF, MVNKLDivergence, ToeplitzMV, ToeplitzMM, \
    InterpolatedToeplitzGPMarginalLogLikelihood
from .utils import LinearCG


__all__ = [
    ToeplitzLazyVariable,
    Distribution,
    ObservationModel,
    AddDiag,
    ExactGPMarginalLogLikelihood,
    Invmm,
    Invmv,
    NormalCDF,
    LogNormalCDF,
    MVNKLDivergence,
    ToeplitzMV,
]


def add_diag(input, diag):
    # LazyVariables don't typecheck, functions do.
    if not isinstance(diag, Variable):
        raise RuntimeError('Expected a variable for the diagonal component.')

    if isinstance(input, ToeplitzLazyVariable):
        if input.J_left is not None:
            toeplitz_diag = diag.expand(len(input.J_left))
        else:
            toeplitz_diag = diag.expand_as(input.c)

        return ToeplitzLazyVariable(input.c, input.r, input.J_left, input.C_left,
                                    input.J_right, input.C_right, toeplitz_diag)
    else:
        return AddDiag()(input, diag)


def exact_gp_marginal_log_likelihood(covar, target):
    if isinstance(covar, ToeplitzLazyVariable):
        W_left = Variable(utils.index_coef_to_sparse(covar.J_left, covar.C_left, len(covar.c)))
        W_right = Variable(utils.index_coef_to_sparse(covar.J_right, covar.C_right, len(covar.c)))
        noise_diag = covar.added_diag
        return InterpolatedToeplitzGPMarginalLogLikelihood(W_left, W_right)(covar.c, target, noise_diag)
    else:
        return ExactGPMarginalLogLikelihood()(covar, target)


def invmm(mat1, mat2):
    return Invmm()(mat1, mat2)


def invmv(mat, vec):
    return Invmv()(mat, vec)


def normal_cdf(x):
    return NormalCDF()(x)


def log_normal_cdf(x):
    return LogNormalCDF()(x)


def mvn_kl_divergence(mean_1, chol_covar_1, mean_2, covar_2):
    return MVNKLDivergence()(mean_1, chol_covar_1, mean_2, covar_2)


def interpolated_toeplitz_mul(c, W_left, W_right, v, noise_diag=None):
    noise_term = None
    if v.ndimension() == 1:
        if noise_diag is not None:
            noise_term = noise_diag.expand_as(v) * v
        v = v.unsqueeze(1)
        mul_func = ToeplitzMV()
    else:
        if noise_diag is not None:
            noise_term = noise_diag.unsqueeze(1).expand_as(v) * v
        mul_func = ToeplitzMM()

    # Get W_{r}^{T}v
    Wt_times_v = torch.dsmm(W_right.t(), v)
    # Get (TW_{r}^{T})v
    TWt_v = mul_func.forward(c, c, Wt_times_v.squeeze())

    if TWt_v.ndimension() == 1:
        TWt_v.unsqueeze_(1)

    # Get (W_{l}TW_{r}^{T})v
    WTWt_v = torch.dsmm(W_left, TWt_v).squeeze()

    if noise_term is not None:
        # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)v
        WTWt_v = WTWt_v + noise_term

    return WTWt_v


def _exact_predict(test_mean, test_test_covar, train_y, train_mean,
                   train_train_covar, train_test_covar, test_train_covar, alpha=None):
    if isinstance(train_train_covar, ToeplitzLazyVariable):
        W_test_left = utils.index_coef_to_sparse(test_train_covar.J_left,
                                                 test_train_covar.C_left,
                                                 len(test_train_covar.c))
        W_test_right = utils.index_coef_to_sparse(test_train_covar.J_right,
                                                  test_train_covar.C_right,
                                                  len(test_train_covar.c))
        W_train_left = utils.index_coef_to_sparse(train_train_covar.J_left,
                                                  train_train_covar.C_left,
                                                  len(train_train_covar.c))
        W_train_right = utils.index_coef_to_sparse(train_train_covar.J_right,
                                                   train_train_covar.C_right,
                                                   len(train_train_covar.c))
        noise_diag = train_train_covar.added_diag

        def train_mul_closure(v):
            return interpolated_toeplitz_mul(train_train_covar.c.data, W_train_left, W_train_right, v, noise_diag.data)

        def test_train_mul_closure(v):
            return interpolated_toeplitz_mul(test_train_covar.c.data, W_test_left, W_test_right, v)

        # Update test mean
        if alpha is None:
            alpha = LinearCG().solve(train_mul_closure, train_y - train_mean.data).unsqueeze(1)
            alpha = torch.dsmm(W_train_right.t(), alpha).squeeze()
            alpha = ToeplitzMV().forward(train_train_covar.c.data, train_train_covar.c.data, alpha).unsqueeze(1)

        test_mean = Variable(test_mean.data.add(torch.dsmm(W_test_left, alpha).squeeze()))

        # TODO: Add a diagonal only mode / use implicit math
        train_test_covar = train_test_covar.evaluate()
        test_test_covar = test_test_covar.evaluate()

        test_test_covar_correction = test_train_mul_closure(LinearCG().solve(train_mul_closure, train_test_covar))
        test_test_covar = Variable(test_test_covar.sub(test_test_covar_correction))
    else:
        train_y_var = Variable(train_y)
        if alpha is None:
            alpha = invmv(train_train_covar, train_y_var - train_mean)

        test_mean = test_mean.add(torch.mv(test_train_covar, alpha))

        # Update test-test covar
        test_test_covar_correction = torch.mm(test_train_covar, invmm(train_train_covar, train_test_covar))
        test_test_covar = test_test_covar.sub(test_test_covar_correction)

    return GaussianRandomVariable(test_mean, test_test_covar), alpha
