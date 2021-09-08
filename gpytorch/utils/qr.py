#!/usr/bin/env python3

import torch


def stable_qr(mat):
    """
    performs a QR decomposition on the batched matrix mat.
    We need to use these functions because of

    1. slow batched QR in pytorch (pytorch/pytorch#22573)
    2. possible singularity in R
    """
    if mat.shape[-1] <= 2048:
        # Dispatch to CPU so long as pytorch/pytorch#22573 is not fixed
        device = mat.device
        Q, R = torch.linalg.qr(mat.cpu())
        Q = Q.to(device)
        R = R.to(device)
    else:
        Q, R = torch.linalg.qr(mat)

    Rdiag = torch.diagonal(R, dim1=-2, dim2=-1)
    # if R is almost singular, add jitter
    zeroish = Rdiag.abs() < 1e-6
    if torch.any(zeroish):
        # can't use in-place operation here b/c it would mess up backward pass
        # haven't found a more elegant way to add a jitter diagonal yet...
        Rdiag_sign = torch.sign(Rdiag)
        # force zero diagonals to have jitter added to them.
        Rdiag_sign[Rdiag_sign == 0] = 1.0
        jitter_diag = 1e-6 * Rdiag_sign * zeroish.to(Rdiag)
        R = R + torch.diag_embed(jitter_diag)
    return Q, R
