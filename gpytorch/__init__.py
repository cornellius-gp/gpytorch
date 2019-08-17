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
from .functions import (
    add_diag,
    add_jitter,
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_logdet,
    logdet,
    log_normal_cdf,
    matmul,
    root_decomposition,
    root_inv_decomposition,
    # Deprecated
    inv_quad_log_det,
    log_det,
)
from .mlls import ExactMarginalLogLikelihood, VariationalMarginalLogLikelihood
from .lazy import lazify, delazify, cat


__version__ = "0.3.5"

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
    "cat",
    "delazify",
    "dsmm",
    "inv_matmul",
    "inv_quad",
    "inv_quad_logdet",
    "lazify",
    "logdet",
    "log_normal_cdf",
    "matmul",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "beta_features",
    "settings",
    # Other
    "__version__",
    # Deprecated
    "inv_quad_log_det",
    "log_det",
]