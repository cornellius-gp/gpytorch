from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch


def batch_symeig(mat):
    """
    """
    mat_orig = mat
    if mat.size(-1) <= 32:
        mat = mat.cpu()

    if mat.dim() == 3:
        mat = mat.unsqueeze(0)
    batch_dim1 = mat.size(0)
    batch_dim2 = mat.size(1)
    n = mat.size(2)

    eigenvectors = torch.empty_like(mat)
    eigenvalues = torch.empty(batch_dim1, batch_dim2, n, dtype=mat.dtype, device=mat.device)

    for i in range(batch_dim1):
        for j in range(batch_dim2):
            evals, evecs = mat[i, j].symeig(eigenvectors=True)
            mask = evals.ge(0)
            eigenvectors[i, j] = evecs * mask.type_as(evecs).unsqueeze(0)
            eigenvalues[i, j] = evals.masked_fill_(1 - mask, 1)

    return eigenvalues.type_as(mat_orig), eigenvectors.type_as(mat_orig)
