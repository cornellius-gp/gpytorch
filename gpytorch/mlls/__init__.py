#!/usr/bin/env python3

import warnings
from .added_loss_term import AddedLossTerm
from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from .inducing_point_kernel_added_loss_term import InducingPointKernelAddedLossTerm
from .marginal_log_likelihood import MarginalLogLikelihood
from .predictive_cross_entropy import PredictiveCrossEntropy
from .sum_marginal_log_likelihood import SumMarginalLogLikelihood
from .variational_elbo import VariationalELBO, VariationalELBOEmpirical
from .noise_model_added_loss_term import NoiseModelAddedLossTerm


# Deprecated for 0.4 release
class VariationalMarginalLogLikelihood(VariationalELBO):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "VariationalMarginalLogLikelihood is deprecated. Please use VariationalELBO instead.",
            DeprecationWarning
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "AddedLossTerm",
    "ExactMarginalLogLikelihood",
    "InducingPointKernelAddedLossTerm",
    "MarginalLogLikelihood",
    "NoiseModelAddedLossTerm",
    "PredictiveCrossEntropy",
    "SumMarginalLogLikelihood",
    "VariationalELBO",
    "VariationalELBOEmpirical",
]
