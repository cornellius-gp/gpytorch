import torch
from copy import deepcopy
from operator import mul
from torch.autograd import Variable
from .interpolation import Interpolation
from .lincg import LinearCG
from . import lanczos
from . import sparse
from .stochastic_lq import StochasticLQ
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
        # Special cuda version -- this is faster on the GPU for some reason
        if interp_indices.is_cuda:
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

        # Special non-cuda version -- this is faster on the CPU
        else:
            interp_size = list(interp_indices.size()) + [rhs.size(-1)]
            rhs_size = deepcopy(interp_size)
            rhs_size[-3] = rhs.size()[-2]
            interp_indices_expanded = interp_indices.unsqueeze(-1).expand(*interp_size)
            res = rhs.unsqueeze(-2).expand(*rhs_size).gather(-3, interp_indices_expanded)
            res = res.mul(interp_values.unsqueeze(-1).expand(interp_size))
            return res.sum(-2)


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


def tridiag_batch_potrf(trid, upper=False):
    if not torch.is_tensor(trid):
        raise RuntimeError('tridiag_batch_potrf is only defined for tensors')

    batch_size, diag_size, _ = trid.size()
    batch_index = trid.new(batch_size).long()
    torch.arange(0, batch_size, out=batch_index)
    off_batch_index = batch_index.unsqueeze(1).repeat(diag_size - 1, 1).view(-1)
    batch_index = batch_index.unsqueeze(1).repeat(diag_size, 1).view(-1)
    diag_index = trid.new(diag_size).long()
    torch.arange(0, diag_size, out=diag_index)
    diag_index = diag_index.unsqueeze(1).repeat(1, batch_size).view(-1)
    off_diag_index = trid.new(diag_size - 1).long()
    torch.arange(0, diag_size - 1, out=off_diag_index)
    off_diag_index = off_diag_index.unsqueeze(1).repeat(1, batch_size).view(-1)

    t_main_diag = trid[batch_index, diag_index, diag_index].view(diag_size, batch_size)
    t_off_diag = trid[off_batch_index, off_diag_index + 1, off_diag_index].view(diag_size - 1, batch_size)

    chol_main_diag = t_main_diag.new(*t_main_diag.size())
    chol_off_diag = t_off_diag.new(*t_off_diag.size())

    chol_main_diag[0].copy_(t_main_diag[0].sqrt())
    for i in range(1, diag_size):
        chol_off_diag[i - 1].copy_(t_off_diag[i - 1] / chol_main_diag[i - 1])
        sq_value = t_main_diag[i] - chol_off_diag[i - 1] ** 2
        chol_main_diag[i].copy_(torch.sqrt(sq_value))

    res = trid.new(*trid.size()).zero_()
    main_flattened_indices = batch_index * (batch_size * diag_size) + diag_index * (diag_size + 1)
    off_flattened_indices = sum([
        off_batch_index * (batch_size * (diag_size - 1)),
        (off_diag_index + 1) * diag_size,
        off_diag_index
    ])
    res.view(-1).index_copy_(0, main_flattened_indices, chol_main_diag.view(-1))
    res.view(-1).index_copy_(0, off_flattened_indices, chol_off_diag.view(-1))

    if upper:
        res = res.transpose(-1, -2)
    return res


def tridiag_batch_potrs(tensor, chol_trid, upper=True):
    if not torch.is_tensor(chol_trid):
        raise RuntimeError('tridiag_batch_potrf is only defined for tensors')

    if not tensor.ndimension() == 3:
        raise RuntimeError('Tensor should be 3 dimensional')

    batch_size, diag_size, _ = chol_trid.size()
    batch_index = chol_trid.new(batch_size).long()
    torch.arange(0, batch_size, out=batch_index)
    off_batch_index = batch_index.unsqueeze(1).repeat(1, diag_size - 1).view(-1)
    batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
    diag_index = chol_trid.new(diag_size).long()
    torch.arange(0, diag_size, out=diag_index)
    diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
    off_diag_index = chol_trid.new(diag_size - 1).long()
    torch.arange(0, diag_size - 1, out=off_diag_index)
    off_diag_index = off_diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)

    if upper:
        chol_trid = chol_trid.transpose(-1, -2)

    chol_main_diag = chol_trid[batch_index, diag_index, diag_index].view(batch_size, diag_size)
    chol_off_diag = chol_trid[off_batch_index, off_diag_index + 1, off_diag_index].view(batch_size, diag_size - 1)

    chol_solution = tensor.new(*tensor.size())
    chol_solution[:, 0, :].copy_(tensor[:, 0, :] / chol_main_diag[:, 0].unsqueeze(-1))
    for i in range(1, diag_size):
        inner_part = tensor[:, i, :]
        inner_part = inner_part - chol_off_diag[:, i - 1].unsqueeze(-1) * chol_solution[:, i - 1, :]
        chol_solution[:, i, :].copy_(inner_part / chol_main_diag[:, i].unsqueeze(-1))

    solution = chol_solution.new(*chol_solution.size())
    solution[:, -1, :].copy_(chol_solution[:, -1, :] / chol_main_diag[:, -1].unsqueeze(-1))
    for i in range(diag_size - 2, -1, -1):
        inner_part = chol_solution[:, i, :] - chol_off_diag[:, i].unsqueeze(-1) * solution[:, i + 1, :]
        solution[:, i, :].copy_(inner_part / chol_main_diag[:, i].unsqueeze(-1))

    return solution


__all__ = [
    Interpolation,
    LinearCG,
    StochasticLQ,
    left_interp,
    reverse,
    rcumsum,
    approx_equal,
    bdsmm,
    lanczos,
    sparse,
    sparse_eye,
    sparse_getitem,
    sparse_repeat,
    trace_components,
    to_sparse,
    tridiag_batch_potrf,
    tridiag_batch_potrs,
]
