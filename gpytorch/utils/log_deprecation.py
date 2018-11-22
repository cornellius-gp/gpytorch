#!/usr/bin/env python3

from ..kernels import (
    CosineKernel,
    IndexKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    ScaleKernel,
    SpectralMixtureKernel,
)
from ..likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood


MODULES_WITH_LOG_PARAMS = [
    CosineKernel,
    IndexKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    ScaleKernel,
    SpectralMixtureKernel,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
]

LOG_DEPRECATION_MSG = (
    "The '{log_name}' parameter is deprecated in favor of '{name}'  because we no longer ensure "
    "positiveness with torch.exp for improved stability reasons and will be removed in a future "
    "release."
)
