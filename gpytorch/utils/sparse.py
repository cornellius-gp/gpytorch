#!/usr/bin/env python3

import torch

from .broadcasting import _matmul_broadcast_shape


def make_sparse_from_indices_and_values(interp_indices, interp_values, num_rows):
    """
    This produces a sparse tensor with a fixed number of non-zero entries in each column.

    Args:
        - interp_indices - Tensor (batch_size) x num_cols x n_nonzero_entries
            A matrix which has the indices of the nonzero_entries for each column
        - interp_values - Tensor (batch_size) x num_cols x n_nonzero_entries
            The corresponding values
        - num_rows - the number of rows in the result matrix

    Returns:
        - SparseTensor - (batch_size) x num_cols x num_rows
    """

    if not torch.is_tensor(interp_indices):
        raise RuntimeError("interp_indices and interp_values should be tensors")

    # Is it batch mode?
    batch_shape = interp_values.shape[:-2]
    n_target_points, n_coefficients = interp_values.shape[-2:]

    # Index tensor
    batch_tensors = []
    for i, batch_size in enumerate(batch_shape):
        batch_tensor = torch.arange(0, batch_size, dtype=torch.long, device=interp_values.device)
        batch_tensor = (
            batch_tensor.unsqueeze_(1)
            .repeat(batch_shape[:i].numel(), batch_shape[i + 1 :].numel() * n_target_points * n_coefficients)
            .view(-1)
        )
        batch_tensors.append(batch_tensor)

    row_tensor = torch.arange(0, n_target_points, dtype=torch.long, device=interp_values.device)
    row_tensor = row_tensor.unsqueeze_(1).repeat(batch_shape.numel(), n_coefficients).view(-1)
    index_tensor = torch.stack([*batch_tensors, interp_indices.reshape(-1), row_tensor], 0)

    # Value tensor
    value_tensor = interp_values.reshape(-1)
    nonzero_indices = value_tensor.nonzero(as_tuple=False)
    if nonzero_indices.storage():
        nonzero_indices.squeeze_()
        index_tensor = index_tensor.index_select(1, nonzero_indices)
        value_tensor = value_tensor.index_select(0, nonzero_indices)
    else:
        index_tensor = index_tensor.resize_(interp_indices.dim(), 1).zero_()
        value_tensor = value_tensor.resize_(1).zero_()

    # Make the sparse tensor
    type_name = value_tensor.type().split(".")[-1]  # e.g. FloatTensor
    interp_size = torch.Size((*batch_shape, num_rows, n_target_points))
    if index_tensor.is_cuda:
        cls = getattr(torch.cuda.sparse, type_name)
    else:
        cls = getattr(torch.sparse, type_name)
    res = cls(index_tensor, value_tensor, interp_size)

    # Wrap things as a variable, if necessary
    return res


def bdsmm(sparse, dense):
    """
    Batch dense-sparse matrix multiply
    """
    # Make the batch sparse matrix into a block-diagonal matrix
    if sparse.ndimension() > 2:
        # Expand the tensors to account for broadcasting
        output_shape = _matmul_broadcast_shape(sparse.shape, dense.shape)
        expanded_sparse_shape = output_shape[:-2] + sparse.shape[-2:]
        unsqueezed_sparse_shape = [1 for _ in range(len(output_shape) - sparse.dim())] + list(sparse.shape)
        repeat_sizes = tuple(
            output_size // sparse_size
            for output_size, sparse_size in zip(expanded_sparse_shape, unsqueezed_sparse_shape)
        )
        sparse = sparse_repeat(sparse, *repeat_sizes)
        dense = dense.expand(*output_shape[:-2], dense.size(-2), dense.size(-1))

        # Figure out how much need to be added to the row/column indices to create
        # a block-diagonal matrix
        *batch_shape, num_rows, num_cols = sparse.shape
        batch_size = torch.Size(batch_shape).numel()
        batch_multiplication_factor = torch.tensor(
            [torch.Size(batch_shape[i + 1 :]).numel() for i in range(len(batch_shape))],
            dtype=torch.long,
            device=sparse.device,
        )
        if batch_multiplication_factor.is_cuda:
            batch_assignment = (sparse._indices()[:-2].float().t() @ batch_multiplication_factor.float()).long()
        else:
            batch_assignment = sparse._indices()[:-2].t() @ batch_multiplication_factor

        # Create block-diagonal sparse tensor
        indices = sparse._indices()[-2:].clone()
        indices[0].add_(batch_assignment, alpha=num_rows)
        indices[1].add_(batch_assignment, alpha=num_cols)
        sparse_2d = torch.sparse_coo_tensor(
            indices,
            sparse._values(),
            torch.Size((batch_size * num_rows, batch_size * num_cols)),
            dtype=sparse._values().dtype,
            device=sparse._values().device,
        )

        dense_2d = dense.reshape(batch_size * num_cols, -1)
        res = torch.dsmm(sparse_2d, dense_2d)
        res = res.view(*batch_shape, num_rows, -1)
        return res

    elif dense.dim() > 2:
        *batch_shape, num_rows, num_cols = dense.size()
        batch_size = torch.Size(batch_shape).numel()
        dense = dense.view(batch_size, num_rows, num_cols)
        res = torch.dsmm(sparse, dense.transpose(0, 1).reshape(-1, batch_size * num_cols))
        res = res.view(-1, batch_size, num_cols)
        res = res.transpose(0, 1).reshape(*batch_shape, -1, num_cols)
        return res

    else:
        return torch.dsmm(sparse, dense)


def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


def sparse_getitem(sparse, idxs):
    """ """
    if not isinstance(idxs, tuple):
        idxs = (idxs,)

    if not sparse.ndimension() <= 2:
        raise RuntimeError("Must be a 1d or 2d sparse tensor")

    if len(idxs) > sparse.ndimension():
        raise RuntimeError("Invalid index for %d-order tensor" % sparse.ndimension())

    indices = sparse._indices()
    values = sparse._values()
    size = list(sparse.size())

    for i, idx in list(enumerate(idxs))[::-1]:
        if isinstance(idx, int):
            del size[i]
            mask = indices[i].eq(idx)
            if torch.any(mask):
                new_indices = torch.zeros(
                    indices.size(0) - 1, torch.sum(mask), dtype=indices.dtype, device=indices.device
                )
                for j in range(indices.size(0)):
                    if i > j:
                        new_indices[j].copy_(indices[j][mask])
                    elif i < j:
                        new_indices[j - 1].copy_(indices[j][mask])
                indices = new_indices
                values = values[mask]
            else:
                indices.resize_(indices.size(0) - 1, 1).zero_()
                values.resize_(1).zero_()

            if not len(size):
                return sum(values)

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(size[i])
            size = list(size[:i]) + [stop - start] + list(size[i + 1 :])
            if step != 1:
                raise RuntimeError("Slicing with step is not supported")
            mask = indices[i].lt(stop) & indices[i].ge(start)
            if torch.any(mask):
                new_indices = torch.zeros(indices.size(0), torch.sum(mask), dtype=indices.dtype, device=indices.device)
                for j in range(indices.size(0)):
                    new_indices[j].copy_(indices[j][mask])
                new_indices[i].sub_(start)
                indices = new_indices
                values = values[mask]
            else:
                indices.resize_(indices.size(0), 1).zero_()
                values.resize_(1).zero_()

        else:
            raise RuntimeError("Unknown index type")

    return torch.sparse_coo_tensor(indices, values, torch.Size(size), dtype=values.dtype, device=values.device)


def sparse_repeat(sparse, *repeat_sizes):
    """ """
    if len(repeat_sizes) == 1 and isinstance(repeat_sizes, tuple):
        repeat_sizes = repeat_sizes[0]

    if len(repeat_sizes) > len(sparse.shape):
        num_new_dims = len(repeat_sizes) - len(sparse.shape)
        new_indices = sparse._indices()
        new_indices = torch.cat(
            [
                torch.zeros(num_new_dims, new_indices.size(1), dtype=new_indices.dtype, device=new_indices.device),
                new_indices,
            ],
            0,
        )
        sparse = torch.sparse_coo_tensor(
            new_indices,
            sparse._values(),
            torch.Size((*[1 for _ in range(num_new_dims)], *sparse.shape)),
            dtype=sparse.dtype,
            device=sparse.device,
        )

    for i, repeat_size in enumerate(repeat_sizes):
        if repeat_size > 1:
            new_indices = sparse._indices().repeat(1, repeat_size)
            adding_factor = torch.arange(0, repeat_size, dtype=new_indices.dtype, device=new_indices.device).unsqueeze_(
                1
            )
            new_indices[i].view(repeat_size, -1).add_(adding_factor)
            sparse = torch.sparse_coo_tensor(
                new_indices,
                sparse._values().repeat(repeat_size),
                torch.Size((*sparse.shape[:i], repeat_size * sparse.size(i), *sparse.shape[i + 1 :])),
                dtype=sparse.dtype,
                device=sparse.device,
            )

    return sparse


def to_sparse(dense):
    """ """
    mask = dense.ne(0)
    indices = mask.nonzero(as_tuple=False)
    if indices.storage():
        values = dense[mask]
    else:
        indices = indices.resize_(1, dense.ndimension()).zero_()
        values = torch.tensor(0, dtype=dense.dtype, device=dense.device)

    # Construct sparse tensor
    klass = getattr(torch.sparse, dense.type().split(".")[-1])
    res = klass(indices.t(), values, dense.size())
    if dense.is_cuda:
        res = res.cuda()
    return res
