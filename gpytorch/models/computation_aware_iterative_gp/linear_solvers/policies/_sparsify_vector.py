#!/usr/bin/env python3

import torch
from linear_operator.operators import BlockSparseLinearOperator


def sparsify_vector(vector: torch.Tensor, num_non_zero: int, strategy="topk") -> BlockSparseLinearOperator:
    """Sparsifies a vector by retaining a given number of entries."""
    if strategy == "topk":
        _, topk_idcs = torch.topk(torch.abs(vector), k=num_non_zero, largest=True)
        return BlockSparseLinearOperator(non_zero_idcs=topk_idcs, blocks=vector[topk_idcs], size_sparse_dim=len(vector))

    else:
        raise NotImplementedError(f"Sparsification strategy {strategy} not available.")
