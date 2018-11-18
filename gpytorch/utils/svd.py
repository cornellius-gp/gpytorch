#!/usr/bin/env python3

import torch


def batch_svd(mat):
    """
    TODO: Replace with torch.svd once PyTorch supports batch SVD
    """
    mat_orig = mat
    batch_shape = torch.Size(mat_orig.shape[:-2])
    matrix_shape = torch.Size(mat_orig.shape[-2:])

    # Smaller matrices are faster on the CPU than the GPU
    if mat.size(-1) <= 32:
        mat = mat.cpu()

    mat = mat.view(-1, *matrix_shape)
    left_vecs = torch.empty(
        batch_shape.numel(), *(mat_orig.size(-2), mat_orig.size(-2)), dtype=mat.dtype, device=mat.device
    )

    singular_values = torch.empty(batch_shape.numel(), mat_orig.size(-2), dtype=mat.dtype, device=mat.device)

    right_vecs = torch.empty(
        batch_shape.numel(), *(mat_orig.size(-1), mat_orig.size(-2)), dtype=mat.dtype, device=mat.device
    )

    for i in range(batch_shape.numel()):
        left, sigma, right = mat[i].svd()
        left_vecs[i] = left
        singular_values[i] = sigma
        right_vecs[i] = right

    return (
        left_vecs.type_as(mat_orig).view(*batch_shape, *(mat_orig.size(-2), mat_orig.size(-2))),
        singular_values.type_as(mat_orig).view(*batch_shape, -1),
        right_vecs.type_as(mat_orig).view(*batch_shape, *(mat_orig.size(-1), mat_orig.size(-2))),
    )
