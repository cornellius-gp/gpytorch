from . import models
from . import likelihoods
from . import means
from . import mlls
from . import kernels
from . import priors
from . import random_variables
from . import lazy
from . import variational
from . import utils
from .module import Module
from .mlls import ExactMarginalLogLikelihood, VariationalMarginalLogLikelihood
from .functions import add_diag, add_jitter, dsmm, log_normal_cdf, normal_cdf
from .functions import inv_matmul, inv_quad, inv_quad_log_det, log_det, matmul
from .functions import root_decomposition, root_inv_decomposition
from .functions import exact_predictive_mean, exact_predictive_covar
from . import beta_features
from . import settings
from .beta_features import fast_pred_var


__all__ = [
    # Submodules
    'models',
    'likelihoods',
    'mlls',
    'means',
    'kernels',
    'priors',
    'random_variables',
    'lazy',
    'variational',
    'utils',
    # Classes
    'Module',
    'ExactMarginalLogLikelihood',
    'VariationalMarginalLogLikelihood',
    # Functions
    'add_diag',
    'add_jitter',
    'dsmm',
    'exact_predictive_mean',
    'exact_predictive_covar',
    'inv_matmul',
    'inv_quad',
    'inv_quad_log_det',
    'matmul',
    'log_det',
    'log_normal_cdf',
    'normal_cdf',
    'root_decomposition',
    'root_inv_decomposition',
    # Context managers
    'beta_features',
    'fast_pred_var',
    'settings',
]
