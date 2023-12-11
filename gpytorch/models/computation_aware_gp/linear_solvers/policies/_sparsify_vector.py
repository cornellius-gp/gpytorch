#!/usr/bin/env python3

import torch


def sparsify_vector(vector: torch.Tensor, num_non_zero: int, strategy="topk") -> torch.Tensor:
    """Sparsifies a vector by retaining a given number of entries."""
    if strategy == "topk":
        _, topk_idcs = torch.topk(torch.abs(vector), k=num_non_zero, largest=True)
        sparse_vector = torch.zeros(
            vector.shape[0],
            dtype=vector.dtype,
            device=vector.device,
        )
        sparse_vector[topk_idcs] = vector[topk_idcs]
        return sparse_vector

    else:
        raise NotImplementedError(f"Sparsification strategy {strategy} not available.")
