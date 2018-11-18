#!/usr/bin/env python3

from operator import mul

import torch


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
    is_batch = interp_indices.ndimension() > 2
    if is_batch:
        batch_size, n_target_points, n_coefficients = interp_values.size()
    else:
        n_target_points, n_coefficients = interp_values.size()

    # Index tensor
    row_tensor = torch.arange(0, n_target_points, dtype=torch.long, device=interp_values.device)
    row_tensor.unsqueeze_(1)
    if is_batch:
        batch_tensor = torch.arange(0, batch_size, dtype=torch.long, device=interp_values.device)
        batch_tensor.unsqueeze_(1).unsqueeze_(2)

        row_tensor = row_tensor.repeat(batch_size, 1, n_coefficients)
        batch_tensor = batch_tensor.repeat(1, n_target_points, n_coefficients)
        index_tensor = torch.stack(
            [
                batch_tensor.contiguous().view(-1),
                interp_indices.contiguous().view(-1),
                row_tensor.contiguous().view(-1),
            ],
            0,
        )
    else:
        row_tensor = row_tensor.repeat(1, n_coefficients)
        index_tensor = torch.cat([interp_indices.contiguous().view(1, -1), row_tensor.contiguous().view(1, -1)], 0)

    # Value tensor
    value_tensor = interp_values.contiguous().view(-1)
    nonzero_indices = value_tensor.nonzero()
    if nonzero_indices.storage():
        nonzero_indices.squeeze_()
        index_tensor = index_tensor.index_select(1, nonzero_indices)
        value_tensor = value_tensor.index_select(0, nonzero_indices)
    else:
        index_tensor = index_tensor.resize_(3 if is_batch else 2, 1).zero_()
        value_tensor = value_tensor.resize_(1).zero_()

    # Size
    if is_batch:
        interp_size = torch.Size([batch_size, num_rows, n_target_points])
    else:
        interp_size = torch.Size([num_rows, n_target_points])

    # Make the sparse tensor
    type_name = value_tensor.type().split(".")[-1]  # e.g. FloatTensor
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
    if sparse.ndimension() > 2:
        batch_size, num_rows, num_cols = sparse.size()
        batch_assignment = sparse._indices()[0]
        indices = sparse._indices()[1:].clone()
        indices[0].add_(num_rows, batch_assignment)
        indices[1].add_(num_cols, batch_assignment)
        sparse_2d = torch.sparse_coo_tensor(
            indices,
            sparse._values(),
            torch.Size((batch_size * num_rows, batch_size * num_cols)),
            dtype=sparse._values().dtype,
            device=sparse._values().device,
        )

        if dense.size(0) == 1:
            dense = dense.repeat(batch_size, 1, 1)
        dense_2d = dense.contiguous().view(batch_size * num_cols, -1)
        res = torch.dsmm(sparse_2d, dense_2d)
        res = res.view(batch_size, num_rows, -1)
        return res
    elif dense.ndimension() == 3:
        batch_size, _, num_cols = dense.size()
        res = torch.dsmm(sparse, dense.transpose(0, 1).contiguous().view(-1, batch_size * num_cols))
        res = res.view(-1, batch_size, num_cols)
        res = res.transpose(0, 1).contiguous()
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
    """
    """
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
            if sum(mask):
                new_indices = torch.zeros(indices.size(0) - 1, sum(mask), dtype=indices.dtype, device=indices.device)
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
            mask = indices[i].lt(stop) * indices[i].ge(start)
            if sum(mask):
                new_indices = torch.zeros(indices.size(0), sum(mask), dtype=indices.dtype, device=indices.device)
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
    """
    """
    orig_ndim = sparse.ndimension()
    new_ndim = len(repeat_sizes)
    orig_nvalues = sparse._indices().size(1)

    # Expand the number of dimensions to match repeat_sizes
    indices = torch.cat(
        [
            torch.zeros(new_ndim - orig_ndim, orig_nvalues, dtype=torch.long, device=sparse._indices().device),
            sparse._indices(),
        ]
    )
    values = sparse._values()
    size = [1] * (new_ndim - orig_ndim) + list(sparse.size())

    # Expand each dimension
    new_indices = torch.zeros(
        indices.size(0), indices.size(1) * mul(*repeat_sizes), dtype=indices.dtype, device=indices.device
    )
    new_values = values.repeat(mul(*repeat_sizes))
    new_size = [dim_size * repeat_size for dim_size, repeat_size in zip(size, repeat_sizes)]

    # Fill in new indices
    new_indices[:, :orig_nvalues].copy_(indices)
    unit_size = orig_nvalues
    for i in range(new_ndim)[::-1]:
        repeat_size = repeat_sizes[i]
        for j in range(1, repeat_size):
            new_indices[:, unit_size * j : unit_size * (j + 1)].copy_(new_indices[:, :unit_size])
            new_indices[i, unit_size * j : unit_size * (j + 1)] += j * size[i]
        unit_size *= repeat_size

    return torch.sparse_coo_tensor(
        new_indices, new_values, torch.Size(new_size), dtype=new_values.dtype, device=new_values.device
    )


def to_sparse(dense):
    """
    """
    mask = dense.ne(0)
    indices = mask.nonzero()
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
