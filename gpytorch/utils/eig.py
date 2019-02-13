#!/usr/bin/env python3

import torch
import _gpytorch_solver

if torch.cuda.is_available():
    import _gpytorch_solver_cuda


def batch_symeig(mat):
    """
    """
    with torch.no_grad():
        batch_shape = torch.Size(mat.shape[:-2])
        matrix_shape = torch.Size(mat.shape[-2:])

        # Smaller matrices are faster on the CPU than the GPU
        if torch.cuda.is_available and mat.is_cuda:
            # TODO: make these numbers not hard-coded
            if (batch_shape.numel() <= 64) and (matrix_shape.numel() <= 32 * 32):
                eigenvalues, eigenvectors = _gpytorch_solver.batch_symeig(mat.cpu())
                eigenvectors = eigenvectors.cuda()
                eigenvalues = eigenvalues.cuda()
            else:
                eigenvalues, eigenvectors = _gpytorch_solver_cuda.batch_symeig_cuda(mat)

        else:
            eigenvalues, eigenvectors = _gpytorch_solver.batch_symeig(mat.cpu())

        # Transpose results to match what PyTorch does (columns are eigenvectors)
        eigenvectors = eigenvectors.transpose(-2, -1)

        # We're going to mask out any eigenvalues (and associated eigenvectors) less than zero
        zero_mask = eigenvalues.lt(0)
        eigenvalues.masked_fill_(zero_mask, 1)
        eigenvectors.masked_fill_(zero_mask.unsqueeze_(-2), 0)

        # Result
        return eigenvalues, eigenvectors
