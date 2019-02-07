#!/usr/bin/env python3


import torch


def tridiag_batch_potrf(trid, upper=False):
    """
    """
    if not torch.is_tensor(trid):
        raise RuntimeError("tridiag_batch_potrf is only defined for tensors")

    batch_size, diag_size, _ = trid.size()
    batch_index = torch.arange(0, batch_size, dtype=torch.long, device=trid.device)
    off_batch_index = batch_index.unsqueeze(1).repeat(diag_size - 1, 1).view(-1)
    batch_index = batch_index.unsqueeze(1).repeat(diag_size, 1).view(-1)
    diag_index = torch.arange(0, diag_size, dtype=torch.long, device=trid.device)
    diag_index = diag_index.unsqueeze(1).repeat(1, batch_size).view(-1)
    off_diag_index = torch.arange(0, diag_size - 1, dtype=torch.long, device=trid.device)
    off_diag_index = off_diag_index.unsqueeze(1).repeat(1, batch_size).view(-1)

    t_main_diag = trid[batch_index, diag_index, diag_index].view(diag_size, batch_size)
    t_off_diag = trid[off_batch_index, off_diag_index + 1, off_diag_index].view(diag_size - 1, batch_size)

    chol_main_diag = torch.empty_like(t_main_diag)
    chol_off_diag = torch.empty_like(t_off_diag)

    chol_main_diag[0].copy_(t_main_diag[0].sqrt())
    for i in range(1, diag_size):
        chol_off_diag[i - 1].copy_(t_off_diag[i - 1] / chol_main_diag[i - 1])
        sq_value = t_main_diag[i] - chol_off_diag[i - 1] ** 2
        chol_main_diag[i].copy_(torch.sqrt(sq_value))

    res = torch.zeros_like(trid)
    main_flattened_indices = batch_index * (batch_size * diag_size) + diag_index * (diag_size + 1)
    off_flattened_indices = sum(
        [off_batch_index * (batch_size * (diag_size - 1)), (off_diag_index + 1) * diag_size, off_diag_index]
    )
    res.view(-1).index_copy_(0, main_flattened_indices, chol_main_diag.view(-1))
    res.view(-1).index_copy_(0, off_flattened_indices, chol_off_diag.view(-1))

    if upper:
        res = res.transpose(-1, -2)
    return res


def tridiag_batch_potrs(tensor, chol_trid, upper=True):
    """
    """
    if not torch.is_tensor(chol_trid):
        raise RuntimeError("tridiag_batch_potrf is only defined for tensors")

    if not tensor.ndimension() == 3:
        raise RuntimeError("Tensor should be 3 dimensional")

    batch_size, diag_size, _ = chol_trid.size()
    batch_index = torch.arange(0, batch_size, dtype=torch.long, device=tensor.device)
    off_batch_index = batch_index.unsqueeze(1).repeat(1, diag_size - 1).view(-1)
    batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
    diag_index = torch.arange(0, diag_size, dtype=torch.long, device=tensor.device)
    diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
    off_diag_index = torch.arange(0, diag_size - 1, dtype=torch.long, device=tensor.device)
    off_diag_index = off_diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)

    if upper:
        chol_trid = chol_trid.transpose(-1, -2)

    chol_main_diag = chol_trid[batch_index, diag_index, diag_index].view(batch_size, diag_size)
    chol_off_diag = chol_trid[off_batch_index, off_diag_index + 1, off_diag_index].view(batch_size, diag_size - 1)

    chol_solution = torch.empty_like(tensor)
    chol_solution[:, 0, :].copy_(tensor[:, 0, :] / chol_main_diag[:, 0].unsqueeze(-1))
    for i in range(1, diag_size):
        inner_part = tensor[:, i, :]
        inner_part = inner_part - chol_off_diag[:, i - 1].unsqueeze(-1) * chol_solution[:, i - 1, :]
        chol_solution[:, i, :].copy_(inner_part / chol_main_diag[:, i].unsqueeze(-1))

    solution = torch.empty_like(chol_solution)
    solution[:, -1, :].copy_(chol_solution[:, -1, :] / chol_main_diag[:, -1].unsqueeze(-1))
    for i in range(diag_size - 2, -1, -1):
        inner_part = chol_solution[:, i, :] - chol_off_diag[:, i].unsqueeze(-1) * solution[:, i + 1, :]
        solution[:, i, :].copy_(inner_part / chol_main_diag[:, i].unsqueeze(-1))

    return solution
