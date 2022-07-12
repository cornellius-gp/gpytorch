#!/usr/bin/env python3

import torch
from torch import Tensor

from .qr import stable_qr


def stable_pinverse(A: Tensor) -> Tensor:
    """Compute a pseudoinverse of a matrix. Employs a stabilized QR decomposition."""
    if A.shape[-2] >= A.shape[-1]:
        # skinny (or square) matrix
        Q, R = stable_qr(A)
        if hasattr(torch.linalg, "solve_triangular"):
            # PyTorch 1.11+
            return torch.linalg.solve_triangular(R, Q.transpose(-1, -2))
        else:
            # PyTorch 1.10
            return torch.triangular_solve(Q.transpose(-1, -2), R).solution
    else:
        # fat matrix
        Q, R = stable_qr(A.transpose(-1, -2))
        if hasattr(torch.linalg, "solve_triangular"):
            # PyTorch 1.11+
            return torch.linalg.solve_triangular(R, Q.transpose(-1, -2)).transpose(-1, -2)
        else:
            # PyTorch 1.10
            return torch.triangular_solve(Q.transpose(-1, -2), R).solution.transpose(-1, -2)
