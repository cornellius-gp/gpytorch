#!/usr/bin/env python3

import torch

from .. import settings
from .broadcasting import _mul_broadcast_shape, _pad_with_singletons

# A slice that does nothing to a dimension
_noop_index = slice(None, None, None)


def _compute_getitem_size(obj, indices):
    """
    Given an object and a tuple of indices, computes the final size of the
    Indices is a tuple containing ints, slices, and tensors

    .. note::
        The length of indices must match the dimensionality of obj

    Args:
        obj - tensor or LazyTensor
        indices - tuple of ints, slices, tensors

    Returns:
        :class:`torch.Size`
    """
    if obj.dim() != len(indices):
        raise RuntimeError(
            "_compute_getitem_size assumes that obj (size: {}) and indices (len: {}) have the "
            "same dimensionality.".format(obj.shape, len(indices))
        )

    final_shape = []
    tensor_idx = None
    tensor_idx_shape = None
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
            if tensor_idx is not None:
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
                tensor_idx_shape = idx.shape
                tensor_idx = len(final_shape)

            # If we don't have a continuous set of tensor indices, then the tensor indexed part
            # goes to the front
            else:
                try:
                    tensor_idx_shape = _mul_broadcast_shape(tensor_idx_shape, idx.shape)
                except RuntimeError:
                    raise IndexError(
                        "Incompatible tensor indices in index - got shapes of {} .".format(
                            [idx.shape for idx in indices if torch.is_tensor(idx)]
                        )
                    )

                if slice_after_tensor_idx:
                    tensor_idx = 0

    # If we don't have a continuous set of tensor indices, then the tensor indexed part
    # goes to the front
    if tensor_idx is not None:
        final_shape = final_shape[:tensor_idx] + list(tensor_idx_shape) + final_shape[tensor_idx:]

    return torch.Size(final_shape)


def _convert_indices_to_tensors(obj, indices):
    """
    Given an index made up of tensors/slices/ints, returns a tensor-only index that has the
    same outcome as the original index (when applied to the obj)

    .. note::
        The length of indices must match the dimensionality of obj

    Args:
        obj - tensor or LazyTensor
        indices - tuple of slices, tensors, ints

    Returns:
        tuple of tensor indices (shapes of tensors will involve broadcasting)

    Example:
        >>> x = torch.randn(3, 6, 4)
        >>> _convert_indices_to_tensors(x, (torch.tensor([0, 1]), 2, slice(None, None, None)))
        >>> # (torch.tensor([[[0]], [[1]]]), torch.tensor([[[2]]]), torch.tensor([[[0, 1, 2, 3]]]))
    """
    slice_indices = tuple(index for index in indices if isinstance(index, slice))
    tensor_indices = tuple(index for index in indices if torch.is_tensor(index))
    tensor_index_shape = _mul_broadcast_shape(*[tensor_index.shape for tensor_index in tensor_indices])

    # How many dimensions will the new tensor index have?
    num_final_dims = len(slice_indices) + len(tensor_index_shape)
    # Determine if the tensor index is being moved to the front
    tensor_index_moved_to_start = _is_tensor_index_moved_to_start(indices)

    # These are counters of the number of singleton dimensions that we need to append to
    # the left and right of the indices that we're converting to tensor indices
    num_singletons_before = len(tensor_index_shape) if tensor_index_moved_to_start else 0
    num_singletons_after = (num_final_dims - len(tensor_index_shape)) if tensor_index_moved_to_start else num_final_dims
    # These are counters of the number of singleton dimensions that we need to append to
    # the left and right of the indices that are currently tensor indices
    num_singletons_before_tensor = 0 if tensor_index_moved_to_start else None
    num_singletons_after_tensor = (num_final_dims - len(tensor_index_shape)) if tensor_index_moved_to_start else None

    # Compute the size suggested by the tensor indices
    new_indices = []
    for dim, index in enumerate(indices):
        # slice - the tensor index will represent the slice
        if isinstance(index, slice):
            num_singletons_after -= 1
            new_index = torch.arange(0, obj.size(dim), device=obj.device)[index]
            new_index = _pad_with_singletons(new_index, num_singletons_before, num_singletons_after)
            num_singletons_before += 1

        # int - the tensor index will have only one element
        elif isinstance(index, int):
            new_index = torch.tensor(index, dtype=torch.long, device=obj.device)
            new_index = _pad_with_singletons(new_index, num_singletons_before, num_singletons_after)

        elif torch.is_tensor(index):
            # If this is the first tensor index we've seen, and we aren't moving all tensor indices to the start
            # Then let's mark how much padding we need for subsequent tensor indices
            if num_singletons_before_tensor is None:
                num_singletons_after -= len(tensor_index_shape)
                num_singletons_before_tensor = num_singletons_before
                num_singletons_after_tensor = num_singletons_after
                num_singletons_before += len(tensor_index_shape)
            new_index = _pad_with_singletons(index, num_singletons_before_tensor, num_singletons_after_tensor)

        new_indices.append(new_index)

    return tuple(new_indices)


def _equal_indices(a, b):
    """
    Helper which checks whether two index components (int, slice, tensor) are equal
    """
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    elif not torch.is_tensor(a) and not torch.is_tensor(b):
        return a == b
    else:
        return False


def _is_noop_index(index):
    """
    Determine if a given index is a noop (e.g. ":")
    """
    return isinstance(index, slice) and index == _noop_index


def _is_tensor_index_moved_to_start(indices):
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
