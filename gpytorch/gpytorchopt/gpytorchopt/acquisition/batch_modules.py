#!/usr/bin/env python3
from gpytorch import Module
from .modules import AcquisitionFunction
from .functional import batch_expected_improvement
from .functional import batch_probability_of_improvement
from .functional import batch_upper_confidence_bound
import torch


"""
Implements batch acquisition functions using the reparameterization trick
as outlined by:
    Wilson, J. T., Moriconi, R., Hutter, F., & Deisenroth, M. P. (2017).
    The reparameterization trick for acquisition functions.
    arXiv preprint arXiv:1712.00424.
"""


class qExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp_model: Module, best_y) -> None:
        super(qExpectedImprovement, self).__init__(gp_model)
        self.best_y = best_y

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_expected_improvement(
            self.gp_model,
            candidate_set,
            self.best_y,
        )


class qProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, gp_model: Module, best_y) -> None:
        super(qProbabilityOfImprovement, self).__init__(gp_model)

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_probability_of_improvement(
            self.gp_model,
            candidate_set,
            self.best_y,
        )


class qUpperConfidenceBound(AcquisitionFunction):
    def __init__(self, gp_model: Module, beta: float) -> None:
        super(qUpperConfidenceBound, self).__init__(gp_model)
        self.beta = beta

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_upper_confidence_bound(
            self.gp_model,
            candidate_set,
            self.beta,
        )
