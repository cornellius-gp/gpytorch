#!/usr/bin/env python3

import torch


def batch_symeig(mat):
    """
    """
    mat_orig = mat
    batch_shape = torch.Size(mat_orig.shape[:-2])
    matrix_shape = torch.Size(mat_orig.shape[-2:])

    # Smaller matrices are faster on the CPU than the GPU
    if mat.size(-1) <= 32:
        mat = mat.cpu()

    mat = mat.view(-1, *matrix_shape)
    eigenvectors = torch.empty(batch_shape.numel(), *matrix_shape, dtype=mat.dtype, device=mat.device)
    eigenvalues = torch.empty(batch_shape.numel(), matrix_shape[-1], dtype=mat.dtype, device=mat.device)

    for i in range(batch_shape.numel()):
        evals, evecs = mat[i].symeig(eigenvectors=True)
        mask = evals.ge(0)
        eigenvectors[i] = evecs * mask.type_as(evecs).unsqueeze(0)
        eigenvalues[i] = evals.masked_fill_(1 - mask, 1)

    return eigenvalues.type_as(mat_orig).view(*batch_shape, -1), eigenvectors.type_as(mat_orig).view_as(mat_orig)
