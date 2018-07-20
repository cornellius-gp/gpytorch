#!/usr/bin/env python3

from gpytorch import Module
import torch
from .functional import expected_improvement
from .functional import max_value_entropy_search
from .functional import probability_of_improvement
from .functional import upper_confidence_bound


class AcquisitionFunction(Module):
    def __init__(self, gp_model: Module) -> None:
        super(AcquisitionFunction, self).__init__()
        self.gp_model = gp_model

    def forward(self, candidate_set: torch.Tensor) -> torch.tensor:
        # takes in an n*d candidate_set tensor and return an n*1 tensor
        raise NotImplementedError("AcquisitionFunction cannot be used directly")


class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp_model: Module, best_y) -> None:
        super(ExpectedImprovement, self).__init__(gp_model)
        self.best_y = best_y

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return expected_improvement(self.gp_model, candidate_set, self.best_y)


class MaxValueEntropySearch(AcquisitionFunction):
    def __init__(self, gp_model, num_samples):
        # K: # of sampled function maxima
        super(MaxValueEntropySearch, self).__init__(gp_model)
        self.num_samples = num_samples

    def forward(self, candidate_set):
        return max_value_entropy_search(self.gp_model, candidate_set, self.num_samples)


class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, gp_model: Module, best_y) -> None:
        super(ProbabilityOfImprovement, self).__init__(gp_model)

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return probability_of_improvement(self.gp_model, candidate_set, self.best_y)


class UpperConfidenceBound(AcquisitionFunction):
    def __init__(self, gp_model: Module, beta: float) -> None:
        super(UpperConfidenceBound, self).__init__(gp_model)
        self.beta = beta

    def forward(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return upper_confidence_bound(self.gp_model, candidate_set, self.beta)
