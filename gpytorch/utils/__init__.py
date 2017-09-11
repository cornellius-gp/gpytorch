import torch
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

    print(indices, values, size)
    return sparse.__class__(indices, values, torch.Size(size))


__all__ = [
    Interpolation,
    LinearCG,
    StochasticLQ,
    reverse,
    rcumsum,
    approx_equal,
    sparse_eye,
    sparse_getitem
]
