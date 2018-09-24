from __future__ import absolute_import, division, print_function, unicode_literals

from . import (
    beta_features,
    kernels,
    lazy,
    likelihoods,
    means,
    mlls,
    models,
    priors,
    random_variables,
    settings,
    utils,
    variational,
)
from .beta_features import fast_pred_var
from .functions import (
    add_diag,
    add_jitter,
    dsmm,
    exact_predictive_covar,
    exact_predictive_mean,
    inv_matmul,
    inv_quad,
    inv_quad_log_det,
    log_det,
    log_normal_cdf,
    matmul,
    normal_cdf,
    root_decomposition,
    root_inv_decomposition,
)
from .mlls import ExactMarginalLogLikelihood, VariationalMarginalLogLikelihood
from .module import Module


__all__ = [
    # Submodules
    "distributions",
    "kernels",
    "lazy",
    "likelihoods",
    "means",
    "mlls",
    "models",
    "priors",
    "random_variables",
    "utils",
    "variational",
    # Classes
    "Module",
    "ExactMarginalLogLikelihood",
    "VariationalMarginalLogLikelihood",
    # Functions
    "add_diag",
    "add_jitter",
    "dsmm",
    "exact_predictive_mean",
    "exact_predictive_covar",
    "inv_matmul",
    "inv_quad",
    "inv_quad_log_det",
    "log_det",
    "log_normal_cdf",
    "matmul",
    "normal_cdf",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "beta_features",
    "fast_pred_var",
    "settings",
]
