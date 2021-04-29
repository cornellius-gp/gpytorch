#!/usr/bin/env python3

from typing import Optional, Union

import torch


def apply_permutation(
    matrix: Union["LazyTensor", torch.Tensor],  # noqa: F821
    left_permutation: Optional[torch.Tensor] = None,
    right_permutation: Optional[torch.Tensor] = None,
):
    r"""
    Applies a left and/or right (partial) permutation to a given matrix :math:`\mathbf K`:

    .. math::

        \begin{equation}
            \boldsymbol{\Pi}_\text{left} \mathbf K \boldsymbol{\Pi}_\text{right}^\top
        \end{equation}

    where the permutation matrices :math:`\boldsymbol{\Pi}_\text{left}` and :math:`\boldsymbol{\Pi}_\text{right}^\top`
    are represented by vectors :attr:`left_permutation` and :attr:`right_permutation`.

    The permutation matrices may be partial permutations (only selecting a subset of rows/columns)
    or full permutations (permuting all rows/columns).

    Importantly, if :math:`\mathbf K` is a batch of matrices, :attr:`left_permutation` and :attr:`right_permutation`
    can be a batch of permutation vectors, and this function will apply the appropriate permutation to each batch entry.
    Broadcasting rules apply.

    :param matrix: :math:`\mathbf K`
    :type matrix: ~gpytorch.lazy.LazyTensor or ~torch.Tensor (... x n x n)
    :param left_permutation: vector representing :math:`\boldsymbol{\Pi}_\text{left}`
    :type left_permutation: ~torch.Tensor, optional (... x <= n)
    :param right_permutation: vector representing :math:`\boldsymbol{\Pi}_\text{right}`
    :type right_permutation: ~torch.Tensor, optional (... x <= n)
    :return: :math:`\boldsymbol{\Pi}_\text{left} \mathbf K \boldsymbol{\Pi}_\text{right}^\top`
    :rtype: ~torch.Tensor

    Example:
        >>> _factor = torch.randn(2, 3, 5, 5)
        >>> matrix = factor @ factor.transpose(-1, -2)  # 2 x 3 x 5 x 5
        >>> left_permutation = torch.tensor([
        >>>     [ 1, 3, 2, 4, 0 ],
        >>>     [ 2, 1, 0, 3, 4 ],
        >>>     [ 0, 1, 2, 4, 3 ],
        >>> ])  # Full permutation: 2 x 3 x 5
        >>> right_permutation = torch.tensor([
        >>>     [ 1, 3, 2 ],
        >>>     [ 2, 1, 0 ],
        >>>     [ 0, 1, 2 ],
        >>> ])  # Partial permutation: 2 x 3 x 3
        >>> apply_permutation(matrix, left_permutation, right_permutation)  # 2 x 3 x 5 x 3
    """
    from ..lazy import delazify

    if left_permutation is None and right_permutation is None:
        return delazify(matrix)

    # Create a set of index vectors for each batch dimension
    # This will ensure that the indexing operations with left_permutation and right_permutation
    # only select the elements from the appropriate batch
    batch_shape = matrix.shape[:-2]
    batch_idx = []
    for i, batch_size in enumerate(batch_shape):
        expanded_shape = [1 for _ in batch_shape] + [1, 1]
        expanded_shape[i] = batch_size
        sub_batch_idx = torch.arange(batch_size, device=matrix.device).view(*expanded_shape)
        batch_idx.append(sub_batch_idx)

    # If we don't have a left_permutation vector, we'll just use a slice
    if left_permutation is None:
        left_permutation = torch.arange(matrix.size(-2), device=matrix.device)
    if right_permutation is None:
        right_permutation = torch.arange(matrix.size(-1), device=matrix.device)

    # Apply permutations
    return delazify(matrix.__getitem__((*batch_idx, left_permutation.unsqueeze(-1), right_permutation.unsqueeze(-2))))


def inverse_permutation(permutation):
    r"""
    Given a (batch of) permutation vector(s),
    return a permutation vector that inverts the original permutation.

    Example:
        >>> permutation = torch.tensor([ 1, 3, 2, 4, 0 ])
        >>> inverse_permutation(permutation)  # torch.tensor([ 4, 0, 2, 1, 3 ])
    """
    arange = torch.arange(permutation.size(-1), device=permutation.device)
    res = torch.zeros_like(permutation).scatter_(-1, permutation, arange.expand_as(permutation))
    return res
