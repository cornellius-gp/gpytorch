from distribution import Distribution
from observation_model import ObservationModel
from .math.functions import AddDiag, ExactGPMarginalLogLikelihood, Invmm, \
    Invmv, NormalCDF, LogNormalCDF, MVNKLDivergence

__all__ = [
    Distribution,
    ObservationModel
]


def add_diag(input, diag):
    return AddDiag()(input, diag)


def exact_gp_marginal_log_likelihood(covar, target):
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
