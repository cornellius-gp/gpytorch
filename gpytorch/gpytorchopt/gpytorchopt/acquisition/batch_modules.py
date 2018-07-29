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
    def __init__(self, gp_model: Module, best_y, mc_samples: int = 1000) -> None:
        super(qExpectedImprovement, self).__init__(gp_model)
        self.best_y = best_y
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_expected_improvement(
            model=self.gp_model,
            x=candidate_set,
            alpha=self.best_y,
            mc_samples=self.mc_samples,
        )


class qProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, gp_model: Module, best_y, mc_samples: int = 1000) -> None:
        super(qProbabilityOfImprovement, self).__init__(gp_model)
        self.best_y = best_y
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_probability_of_improvement(
            model=self.gp_model,
            x=candidate_set,
            alpha=self.best_y,
            mc_samples=self.mc_samples,
        )


class qUpperConfidenceBound(AcquisitionFunction):
    def __init__(self, gp_model: Module, beta: float, mc_samples: int = 1000) -> None:
        super(qUpperConfidenceBound, self).__init__(gp_model)
        self.beta = beta
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_upper_confidence_bound(
            model=self.gp_model,
            x=candidate_set,
            beta=self.beta,
            mc_samples=self.mc_samples,
        )


class qSimpleRegret(AcquisitionFunction):
    def __init__(self, gp_model: Module, mc_samples: int = 1000) -> None:
        super(qSimpleRegret, self).__init__(gp_model)
        self.mc_samples = mc_samples

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return batch_simple_regret(
            x=candidate_set,
            model=self.gp_model,
            mc_samples=self.mc_samples,
        )
