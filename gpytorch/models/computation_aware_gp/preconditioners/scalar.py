"""Scalar preconditioner."""

import torch
from linear_operator.operators import DiagLinearOperator

from .preconditioner import Preconditioner


class Scalar(Preconditioner):
    def __init__(self, kernel, noise, X, **kwargs) -> None:
        super().__init__(kernel=kernel, noise=noise, X=X, **kwargs)

    def inv_matmul(self, input):
        return DiagLinearOperator(self._rescale(torch.as_tensor(1.0)) * torch.ones(self.X.shape[0])) @ input
