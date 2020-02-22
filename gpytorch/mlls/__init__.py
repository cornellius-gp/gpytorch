#!/usr/bin/env python3

import warnings

from .added_loss_term import AddedLossTerm
from .crps_elbo import CRPSELBO
from .deep_approximate_mll import DeepApproximateMLL
from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from .gamma_robust_variational_elbo import GammaRobustVariationalELBO
from .inducing_point_kernel_added_loss_term import InducingPointKernelAddedLossTerm
from .marginal_log_likelihood import MarginalLogLikelihood
from .noise_model_added_loss_term import NoiseModelAddedLossTerm
from .predictive_log_likelihood import PredictiveLogLikelihood
from .sum_marginal_log_likelihood import SumMarginalLogLikelihood
from .variational_elbo import VariationalELBO
from .fml_variational_elbo import FMLVariationalELBO
from .variational_fitc_elbo import VariationalFITCELBO


# Deprecated for 0.4 release
class VariationalMarginalLogLikelihood(VariationalELBO):
    def __init__(self, *args, **kwargs):
        # Remove after 1.0
        warnings.warn(
            "VariationalMarginalLogLikelihood is deprecated. Please use VariationalELBO instead.", DeprecationWarning
        )
        super().__init__(*args, **kwargs)


class VariationalELBOEmpirical(VariationalELBO):
    def __init__(self, *args, **kwargs):
        # Remove after 1.0
        warnings.warn("VariationalELBOEmpirical is deprecated. Please use VariationalELBO instead.", DeprecationWarning)
        super().__init__(*args, **kwargs)


__all__ = [
    "AddedLossTerm",
    "CRPSELBO",
    "DeepApproximateMLL",
    "ExactMarginalLogLikelihood",
    "InducingPointKernelAddedLossTerm",
    "MarginalLogLikelihood",
    "NoiseModelAddedLossTerm",
    "PredictiveLogLikelihood",
    "GammaRobustVariationalELBO",
    "SumMarginalLogLikelihood",
    "VariationalELBO",
    "FMLVariationalELBO",
    "VariationalFITCELBO",
]
