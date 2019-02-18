#!/usr/bin/env python3

import torch
import _gpytorch_solver


def batch_svd(mat):
    """
    TODO: Replace with torch.svd once PyTorch supports batch SVD
    """
    batch_shape = torch.Size(mat.shape[:-2])
    matrix_shape = torch.Size(mat.shape[-2:])
    min_size = min(*matrix_shape)

    # Smaller matrices are faster on the CPU than the GPU
    if min_size <= 32:
        mat = mat.cpu()

    umats, svecs, vmats = _gpytorch_solver.batch_svd(mat)
    umats = umats.view(batch_shape + umats.shape[-2:])
    svecs = svecs.view(*batch_shape, svecs.size(-1))
    vmats = vmats.view(batch_shape + vmats.shape[-2:])
    return umats, svecs, vmats
