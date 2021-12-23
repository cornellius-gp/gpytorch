#!/usr/bin/env python3
from . import (
    beta_features,
    distributions,
    kernels,
    lazy,
    likelihoods,
    means,
    metrics,
    mlls,
    models,
    optim,
    priors,
    settings,
    utils,
    variational,
)
from .functions import (  # Deprecated
    add_diag,
    add_jitter,
    dsmm,
    inv_matmul,
    inv_quad,
    inv_quad_logdet,
    log_normal_cdf,
    logdet,
    matmul,
    pivoted_cholesky,
    root_decomposition,
    root_inv_decomposition,
)
from .lazy import cat, delazify, lazify
from .mlls import ExactMarginalLogLikelihood
from .module import Module

__version__ = "1.6.0"

__all__ = [
    # Submodules
    "distributions",
    "kernels",
    "lazy",
    "likelihoods",
    "means",
    "metrics",
    "mlls",
    "models",
    "optim",
    "priors",
    "utils",
    "variational",
    # Classes
    "Module",
    "ExactMarginalLogLikelihood",
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
    "pivoted_cholesky",
    "root_decomposition",
    "root_inv_decomposition",
    # Context managers
    "beta_features",
    "settings",
    # Other
    "__version__",
]
