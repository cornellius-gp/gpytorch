import torch
from operator import mul
from torch.autograd import Variable
from .interpolation import Interpolation
from .lincg import LinearCG
from .lanczos_quadrature import StochasticLQ


def reverse(input, dim=0):
    """
    Reverses a tensor
    Args:
        - input: tensor to reverse
        - dim: dimension to reverse on
    Returns:
        - reversed input
    """
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return input.index_select(dim, reverse_index)


def rcumsum(input, dim=0):
    """
    Computes a reverse cumulative sum
    Args:
        - input: tensor
        - dim: dimension to reverse on
    Returns:
        - rcumsum on input
    """
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return torch.index_select(input, dim, reverse_index).cumsum(dim).index_select(dim, reverse_index)


def approx_equal(self, other, epsilon=1e-4):
    """
    Determines if two tensors are approximately equal
    Args:
        - self: tensor
        - other: tensor
    Returns:
        - bool
    """
    if isinstance(self, Variable):
        self = self.data
    if isinstance(other, Variable):
        other = other.data
    return torch.max((self - other).abs()) <= epsilon


def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.Tensor([1]).expand(size)
    return torch.sparse.FloatTensor(indices, values, torch.Size([size, size]))


def sparse_getitem(sparse, idxs):
    if not isinstance(idxs, tuple):
        idxs = idxs,

    if not sparse.ndimension() <= 2:
        raise RuntimeError('Must be a 1d or 2d sparse tensor')

    if len(idxs) > sparse.ndimension():
        raise RuntimeError('Invalid index for %d-order tensor' % sparse.ndimension())

    indices = sparse._indices()
    values = sparse._values()
    size = list(sparse.size())

    for i, idx in list(enumerate(idxs))[::-1]:
        if isinstance(idx, int):
            del size[i]
            mask = indices[i].eq(idx)
            if sum(mask):
                new_indices = indices.new().resize_(indices.size(0) - 1, sum(mask)).zero_()
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
            size = list(size[:i]) + [stop - start] + list(size[i + 1:])
            if step != 1:
                raise RuntimeError('Slicing with step is not supported')
            mask = indices[i].lt(stop) * indices[i].ge(start)
            if sum(mask):
                new_indices = indices.new().resize_(indices.size(0), sum(mask)).zero_()
                for j in range(indices.size(0)):
                    new_indices[j].copy_(indices[j][mask])
                new_indices[i].sub_(start)
                indices = new_indices
                values = values[mask]
            else:
                indices.resize_(indices.size(0), 1).zero_()
                values.resize_(1).zero_()

        else:
            raise RuntimeError('Unknown index type')

    return sparse.__class__(indices, values, torch.Size(size))


def sparse_repeat(sparse, *repeat_sizes):
    orig_ndim = sparse.ndimension()
    new_ndim = len(repeat_sizes)
    orig_nvalues = sparse._indices().size(1)

    # Expand the number of dimensions to match repeat_sizes
    indices = torch.cat([sparse._indices().new().resize_(new_ndim - orig_ndim, orig_nvalues).zero_(),
                         sparse._indices()])
    values = sparse._values()
    size = [1] * (new_ndim - orig_ndim) + list(sparse.size())

    # Expand each dimension
    new_indices = indices.new().resize_(indices.size(0), indices.size(1) * mul(*repeat_sizes)).zero_()
    new_values = values.repeat(mul(*repeat_sizes))
    new_size = [dim_size * repeat_size for dim_size, repeat_size in zip(size, repeat_sizes)]

    # Fill in new indices
    new_indices[:, :orig_nvalues].copy_(indices)
    unit_size = orig_nvalues
    for i in range(new_ndim)[::-1]:
        repeat_size = repeat_sizes[i]
        for j in range(1, repeat_size):
            new_indices[:, unit_size * j:unit_size * (j + 1)].copy_(new_indices[:, :unit_size])
            new_indices[i, unit_size * j:unit_size * (j + 1)] += j * size[i]
        unit_size *= repeat_size

    return sparse.__class__(new_indices, new_values, torch.Size(new_size))


__all__ = [
    Interpolation,
    LinearCG,
    StochasticLQ,
    reverse,
    rcumsum,
    approx_equal,
    sparse_eye,
    sparse_getitem,
    sparse_repeat,
]
