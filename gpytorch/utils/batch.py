#!/usr/bin/env python3

import torch


def _create_batch_indices(batch_shape, repeat_size, device):
    """
    A helper method which creates a list of batch indices for use with the LazyTensor _get_indices method
    """
    batch_indices = []
    for i, batch_size in enumerate(batch_shape):
        batch_index = torch.arange(0, batch_size, dtype=torch.long, device=device).unsqueeze(-1)
        batch_index = batch_index.repeat(
            torch.Size(batch_shape[:i]).numel(),
            torch.Size(batch_shape[i + 1:]).numel() * repeat_size
        ).view(-1)
        batch_indices.append(batch_index)
    return batch_indices
