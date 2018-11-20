#!/usr/bin/env python3

import torch


def batch_qr(mat):
    """
    TODO: Replace with torch.qr once PyTorch implements a batch QR decomposition
    """
    mat_orig = mat
    batch_shape = torch.Size(mat_orig.shape[:-2])
    matrix_shape = torch.Size(mat_orig.shape[-2:])

    # Smaller matrices are faster on the CPU than the GPU
    if mat.size(-2) <= 32:
        mat = mat.cpu()

    mat = mat.view(-1, *matrix_shape)
    q_mats = torch.empty(batch_shape.numel(), *matrix_shape)

    for i in range(batch_shape.numel()):
        q_mat, _ = torch.qr(mat[i])
        q_mats[i] = q_mat

    return q_mats.type_as(mat_orig).view_as(mat_orig)
