#!/usr/bin/env python3

import torch

from .. import settings

def pivoting_heuristics(pd_mat, k):
    n = pd_mat.size(-1)

    num_blocks = n // k

    identity = torch.arange(n, dtype=torch.long, device=pd_mat.device)
    pi = identity.clone()

    for i in range(num_blocks):
        """
        i * k: (i + 1) * k
        """
        # sorted=True is not required, but it is easier to debug
        idx_topk = torch.topk(pd_mat[pi[i * k], pi[i * k + 1:]].abs(), k=k - 1, largest=True, sorted=True).indices

        # swap pi[i * k + 1:i * k + k] with pi[i * k + 1:][idx]
        # notice that those two indices might overlap
        first_indices = pi[i * k + 1:i * k + k]
        second_indices = pi[i * k + 1:][idx_topk]

        # before swapping, use mask to select non-overlapping indices
        first_mask = (first_indices.unsqueeze(-1) == second_indices.unsqueeze(-2)).any(dim=-1).logical_not()
        second_mask = (second_indices.unsqueeze(-1) == first_indices.unsqueeze(-2)).any(dim=-1).logical_not()

        # finally, do the swap
        tmp = pi[i * k + 1:i * k + k][first_mask]
        pi[i * k + 1:i * k + k][first_mask] = pi[i * k + 1:][idx_topk[second_mask]]
        pi[i * k + 1:][idx_topk[second_mask]] = tmp

        # tmp = pi[i * k + 1:i * k + k].clone()
        # pi[i * k + 1:i * k + k] = pi[i * k + 1:][idx_topk].clone()
        # pi[i * k + 1:][idx_topk] = tmp

    # make sure pi is a valid permutation
    assert (pi.sort().values == identity).all()

    return pi


def block_jacobi(pd_mat, k):
    # import ipdb
    # ipdb.set_trace()

    from ..lazy import BlockTriangularLazyTensor, PermutationLazyTensor

    n = pd_mat.size(-1)

    if settings.use_pivoting_heuristic.on():
        pi = pivoting_heuristics(pd_mat, k)
    else:
        pi = torch.arange(n, dtype=torch.long, device=pd_mat.device)

    pd_mat = pd_mat.clone()[pi, :][:, pi]

    num_blocks = n // k
    # last_block_size = n - k * num_blocks

    batched_blocks = torch.zeros(
        num_blocks, k, k,
        device=pd_mat.device,
        dtype=pd_mat.dtype,
    )

    # TODO: Vectorize the for loop
    # for i in range(num_blocks):
    #     batched_blocks[i] = pd_mat[i * k: (i + 1) * k, i * k: (i + 1) * k]
    #     error += 0.
    tmp = pd_mat[:num_blocks * k, :num_blocks * k].view(num_blocks, k, num_blocks * k)
    tmp = tmp.transpose(-2, -1)
    tmp = tmp.view(num_blocks, num_blocks, k, k)
    idx = torch.arange(num_blocks, dtype=torch.long, device=pd_mat.device)
    batched_blocks = tmp[idx, idx, :, :]

    last_block = pd_mat[num_blocks * k:, num_blocks * k:]

    batched_chol = torch.linalg.cholesky(batched_blocks)
    last_chol = torch.linalg.cholesky(last_block)

    # tmp = tmp.clone()
    # tmp[idx, idx, :, :] = 0.
    # error = tmp.square().sum() + pd_mat[num_blocks * k:, 0:num_blocks * k].square().sum() * 2
    # print("fro norm diff {:f}".format(error.sqrt().item()))
    # return batched_chol, last_chol, pi

    return PermutationLazyTensor(pi).transpose(-2, -1) @ BlockTriangularLazyTensor(batched_chol, last_chol)
