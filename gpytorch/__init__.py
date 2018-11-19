#!/usr/bin/env python3

from .module import Module
from . import (
    beta_features,
    distributions,
    kernels,
    lazy,
    likelihoods,
    means,
    mlls,
    models,
    priors,
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
