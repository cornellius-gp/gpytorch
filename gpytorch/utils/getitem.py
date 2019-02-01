#!/usr/bin/env python3

import torch
from .. import settings

# A slice that does nothing to a dimension
_noop_index = slice(None, None, None)


def _compute_getitem_size(obj, indices):
    if obj.dim() != len(indices):
        raise RuntimeError(
            "_compute_getitem_size assumes that obj (size: {}) and indices (len: {}) have the "
            "same dimensionality.".format(obj.shape, len(indices))
        )

    final_shape = []
    first_tensor_idx = None
    tensor_idx_shape = None
    continuous_tensor_index = True
    slice_after_tensor_idx = False

    for i, (size, idx) in enumerate(zip(obj.shape, indices)):
        # Handle slice: that dimension gets downsized
        if isinstance(idx, slice):
            if idx == _noop_index:
                final_shape.append(size)
            else:
                final_shape.append(len(range(*idx.indices(size))))

            # If we don't have a continuous set of tensor indices, then the tensor indexed part
            # goes to the front
            if first_tensor_idx is not None:
                slice_after_tensor_idx = True

        # Handle int: we "lose" that dimension
        elif isinstance(idx, int):
            if settings.debug.on():
                try:
                    range(size)[idx]
                except IndexError:
                    raise IndexError(
                        "index element {} ({}) is invalid: out of range for obj of size "
                        "{}.".format(i, idx, obj.shape)
                    )

        # Handle tensor index - this one is complicated
        elif torch.is_tensor(idx):
            if tensor_idx_shape is None:
                tensor_idx_shape = idx.numel()
                first_tensor_idx = len(final_shape)
                final_shape.append(tensor_idx_shape)

            # If we don't have a continuous set of tensor indices, then the tensor indexed part
            # goes to the front
            elif slice_after_tensor_idx:
                continuous_tensor_index = False

            else:
                if settings.debug.on():
                    if idx.numel() != tensor_idx_shape:
                        raise IndexError(
                            "index element {} is an invalid size: expected tensor indices of size {}, got "
                            "{}.".format(i, tensor_idx_shape, idx.numel())
                        )

    # If we don't have a continuous set of tensor indices, then the tensor indexed part
    # goes to the front
    if not continuous_tensor_index:
        del final_shape[first_tensor_idx]
        final_shape.insert(0, tensor_idx_shape)

    return torch.Size(final_shape)


def _tensor_index_to_start(indices):
    """
    Given an index, determine if the indexed part of the getitem is moved to the zero'th dimension
    """
    has_tensor_index = False
    continuous_tensor_index = True

    if torch.is_tensor(indices[0]):
        return True

    for index in indices[1:]:
        if torch.is_tensor(index):
            if not has_tensor_index:
                has_tensor_index = True
            elif not continuous_tensor_index:
                return True

        elif isinstance(index, slice):
            if has_tensor_index:
                continuous_tensor_index = False

    return False


def _equal_indices(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    elif not torch.is_tensor(a) and not torch.is_tensor(b):
        return a == b
    else:
        return False
