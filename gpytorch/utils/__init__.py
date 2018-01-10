import torch
from operator import mul
from torch.autograd import Variable
from .interpolation import Interpolation
from .lincg import LinearCG
from .lanczos_quadrature import StochasticLQ
from .trace import trace_components


def reverse(input, dim=0):
    """
    Reverses a tensor
    Args:
        - input: tensor to reverse
        - dim: dimension to reverse on
    Returns:
        - reversed input
    """
    reverse_index = input.new(input.size(dim)).long()
    torch.arange(1 - input.size(dim), 1, out=reverse_index)
    reverse_index.mul_(-1)
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


def bdsmm(sparse, dense):
    """
    Batch dense-sparse matrix multiply
    """
    if sparse.ndimension() > 2:
        batch_size, n_rows, n_cols = sparse.size()
        batch_assignment = sparse._indices()[0]
        indices = sparse._indices()[1:].clone()
        indices[0].add_(n_rows, batch_assignment)
        indices[1].add_(n_cols, batch_assignment)
        sparse_2d = sparse.__class__(indices, sparse._values(),
                                     torch.Size((batch_size * n_rows, batch_size * n_cols)))

        if dense.size(0) == 1:
            dense = dense.repeat(batch_size, 1, 1)
        dense_2d = dense.contiguous().view(batch_size * n_cols, -1)
        res = torch.dsmm(sparse_2d, dense_2d)
        res = res.view(batch_size, n_rows, -1)
        return res
    elif dense.ndimension() == 3:
        batch_size, _, n_cols = dense.size()
        res = torch.dsmm(sparse, dense.transpose(0, 1).contiguous().view(-1, batch_size * n_cols))
        res = res.view(-1, batch_size, n_cols)
        res = res.transpose(0, 1).contiguous()
        return res
    else:
        return torch.dsmm(sparse, dense)


def left_interp(interp_indices, interp_values, rhs):
    is_vector = rhs.ndimension() == 1

    if is_vector:
        res = rhs.index_select(0, interp_indices.view(-1)).view(*interp_values.size())
        res = res.mul(interp_values)
        res = res.sum(-1)
        return res

    else:
        if interp_indices.ndimension() == 3:
            n_batch, n_data, n_interp = interp_indices.size()
            interp_indices = interp_indices.view(-1)
            if isinstance(interp_indices, Variable):
                interp_indices = interp_indices.data
            interp_values = interp_values.view(-1, 1)

            if rhs.ndimension() == 3:
                if rhs.size(0) == 1 and interp_indices.size(0) > 1:
                    rhs = rhs.expand(interp_indices.size(0), rhs.size(1), rhs.size(2))
                batch_indices = interp_indices.new(n_batch, 1)
                torch.arange(0, n_batch, out=batch_indices[:, 0])
                batch_indices = batch_indices.repeat(1, n_data * n_interp).view(-1)
                res = rhs[batch_indices, interp_indices, :] * interp_values
            else:
                res = rhs[interp_indices, :].unsqueeze(0) * interp_values
            res = res.view(n_batch, n_data, n_interp, -1)
            res = res.sum(-2)
            return res
        else:
            n_data, n_interp = interp_indices.size()
            interp_indices = interp_indices.view(-1)
            interp_values = interp_values.view(-1, 1)
            if rhs.ndimension() == 3:
                n_batch, _, n_cols = rhs.size()
                rhs = rhs.transpose(0, 1).contiguous().view(-1, n_batch * n_cols)
                res = rhs[interp_indices, :] * interp_values
                res = res.view(n_data, n_interp, n_batch, n_cols)
                res = res.sum(-2).transpose(0, 1).contiguous()
            else:
                res = rhs[interp_indices, :] * interp_values
                res = res.view(n_data, n_interp, -1)
                res = res.sum(-2)
            return res


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


def scale_to_bounds(x, lower_bound, upper_bound):
    # Scale features so they fit inside grid bounds
    min_val = x.data.min()
    max_val = x.data.max()
    diff = max_val - min_val
    x = (x - min_val) * (0.95 * (upper_bound - lower_bound) / diff) + 0.95 * lower_bound
    return x


def to_sparse(dense):
    mask = dense.ne(0)
    indices = mask.nonzero()
    if indices.storage():
        values = dense[mask]
    else:
        indices = indices.resize_(1, dense.ndimension()).zero_()
        values = dense.new().resize_(1).zero_()

    # Construct sparse tensor
    klass = getattr(torch.sparse, dense.__class__.__name__)
    res = klass(indices.t(), values, dense.size())
    if dense.is_cuda:
        res = res.cuda()
    return res


__all__ = [
    Interpolation,
    LinearCG,
    StochasticLQ,
    left_interp,
    reverse,
    rcumsum,
    approx_equal,
    bdsmm,
    sparse_eye,
    sparse_getitem,
    sparse_repeat,
    trace_components,
]
