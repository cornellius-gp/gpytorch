import torch
from torch.autograd import Variable


def make_sparse_from_indices_and_values(interp_indices, interp_values, n_rows):
    """
    This produces a sparse tensor with a fixed number of non-zero entries in each column.

    Args:
        interp_indices - Tensor (batch_size) x n_cols x n_nonzero_entries
            A matrix which has the indices of the nonzero_entries for each column
        interp_values - Tensor (batch_size) x n_cols x n_nonzero_entries
            The corresponding values
        n_rows - the number of rows in the result matrix

    Returns:
        SparseTensor - (batch_size) x n_cols x n_rows
    """

    if isinstance(interp_indices, Variable):
        raise RuntimeError('interp_indices and interp_values should be tensors')

    # Is it batch mode?
    is_batch = interp_indices.ndimension() > 2
    if is_batch:
        batch_size, n_target_points, n_coefficients = interp_values.size()
    else:
        n_target_points, n_coefficients = interp_values.size()

    # Index tensor
    row_tensor = interp_indices.new(n_target_points)
    torch.arange(0, n_target_points, out=row_tensor)
    row_tensor.unsqueeze_(1)
    if is_batch:
        batch_tensor = interp_indices.new(batch_size)
        torch.arange(0, batch_size, out=batch_tensor)
        batch_tensor.unsqueeze_(1).unsqueeze_(2)

        row_tensor = row_tensor.repeat(batch_size, 1, n_coefficients)
        batch_tensor = batch_tensor.repeat(1, n_target_points, n_coefficients)
        index_tensor = torch.stack([batch_tensor.contiguous().view(-1),
                                    interp_indices.contiguous().view(-1),
                                    row_tensor.contiguous().view(-1)], 0)
    else:
        row_tensor = row_tensor.repeat(1, n_coefficients)
        index_tensor = torch.cat([interp_indices.contiguous().view(1, -1),
                                  row_tensor.contiguous().view(1, -1)], 0)

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
        interp_size = torch.Size([batch_size, n_rows, n_target_points])
    else:
        interp_size = torch.Size([n_rows, n_target_points])

    # Make the sparse tensor
    if index_tensor.is_cuda:
        res = torch.cuda.sparse.FloatTensor(index_tensor, value_tensor, interp_size)
    else:
        res = torch.sparse.FloatTensor(index_tensor, value_tensor, interp_size)

    # Wrap things as a variable, if necessary
    return res
