import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from copy import deepcopy
from ..utils import bdsmm


def _make_sparse_from_indices_and_values(right_interp_indices, right_interp_values, n_inducing):
    # Is it batch mode?
    is_batch = right_interp_indices.ndimension() > 2
    if is_batch:
        batch_size, n_target_points, n_coefficients = right_interp_values.size()
    else:
        n_target_points, n_coefficients = right_interp_values.size()

    # Index tensor
    row_tensor = right_interp_indices.new(n_target_points)
    torch.arange(0, n_target_points, out=row_tensor)
    row_tensor.unsqueeze_(1)
    if is_batch:
        batch_tensor = right_interp_indices.new(batch_size)
        torch.arange(0, batch_size, out=batch_tensor)
        batch_tensor.unsqueeze_(1).unsqueeze_(2)

        row_tensor = row_tensor.repeat(batch_size, 1, n_coefficients)
        batch_tensor = batch_tensor.repeat(1, n_target_points, n_coefficients)
        index_tensor = torch.stack([batch_tensor.contiguous().view(-1),
                                    right_interp_indices.contiguous().view(-1),
                                    row_tensor.contiguous().view(-1)], 0)
    else:
        row_tensor = row_tensor.repeat(1, n_coefficients)
        index_tensor = torch.cat([right_interp_indices.contiguous().view(1, -1),
                                  row_tensor.contiguous().view(1, -1)], 0)

    # Value tensor
    value_tensor = right_interp_values.contiguous().view(-1)
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
        right_interp_size = torch.Size([batch_size, n_inducing, n_target_points])
    else:
        right_interp_size = torch.Size([n_inducing, n_target_points])

    # Make the sparse tensor
    if index_tensor.is_cuda:
        res = torch.cuda.sparse.FloatTensor(index_tensor, value_tensor, right_interp_size)
    else:
        res = torch.sparse.FloatTensor(index_tensor, value_tensor, right_interp_size)
    return res


class InterpolatedLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable, left_interp_indices=None, left_interp_values=None,
                 right_interp_indices=None, right_interp_values=None):
        super(InterpolatedLazyVariable, self).__init__(base_lazy_variable, left_interp_indices, left_interp_values,
                                                       right_interp_indices, right_interp_values)
        self.base_lazy_variable = base_lazy_variable
        self.tensor_cls = type(self.base_lazy_variable.representation()[0].data)

        if left_interp_indices is None:
            n_rows = self.base_lazy_variable.size(-2)
            self.left_interp_indices = Variable(self.tensor_cls(n_rows).long())
            torch.arange(0, n_rows, out=self.left_interp_indices.data)
        else:
            self.left_interp_indices = left_interp_indices

        if left_interp_values is None:
            self.left_interp_values = Variable(self.tensor_cls(left_interp_values.size()).fill_(1))
        else:
            self.left_interp_values = left_interp_values

        if right_interp_indices is None:
            n_rows = self.base_lazy_variable.size(-2)
            self.right_interp_indices = Variable(self.tensor_cls(n_rows).long())
            torch.arange(0, n_rows, out=self.right_interp_indices.data)
        else:
            self.right_interp_indices = right_interp_indices

        if right_interp_values is None:
            self.right_interp_values = Variable(self.tensor_cls(right_interp_values.size()).fill_(1))
        else:
            self.right_interp_values = right_interp_values

    def _matmul_closure_factory(self, *args):
        base_lazy_variable_representation = args[:-4]
        base_lazy_variable_matmul = self.base_lazy_variable._matmul_closure_factory(*base_lazy_variable_representation)
        left_interp_indices, left_interp_values, right_interp_indices, right_interp_values = args[-4:]

        def closure(tensor):
            if tensor.ndimension() == 1:
                is_vector = True
                tensor = tensor.unsqueeze(-1)
            else:
                is_vector = False

            # right_interp^T * tensor
            right_interp = _make_sparse_from_indices_and_values(right_interp_indices,
                                                                right_interp_values,
                                                                self.base_lazy_variable.size()[-1])
            right_interp_res = bdsmm(right_interp, tensor)

            # base_lazy_var * right_interp^T * tensor
            base_res = base_lazy_variable_matmul(right_interp_res)

            # left_interp * base_lazy_var * right_interp^T * tensor
            left_interp_size = list(left_interp_indices.size()) + [tensor.size(-1)]
            base_res_size = deepcopy(left_interp_size)
            base_res_size[-3] = self.base_lazy_variable.size()[-2]
            left_interp_indices_expanded = left_interp_indices.unsqueeze(-1).expand(*left_interp_size)
            left_interp_res = base_res.unsqueeze(-2).expand(*base_res_size).gather(-3, left_interp_indices_expanded)
            left_interp_res.mul_(left_interp_values.unsqueeze(-1).expand(left_interp_size))

            # Squeeze if necessary
            res = left_interp_res.sum(-2)
            if is_vector:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        base_lazy_var_repr = args[:-4]
        base_lazy_var_deriv = self.base_lazy_variable._derivative_quadratic_form_factory(*base_lazy_var_repr)
        left_interp_indices, left_interp_values, right_interp_indices, right_interp_values = args[-4:]

        def closure(left_factor, right_factor):
            if left_factor.ndimension() == 1:
                left_factor = left_factor.unsqueeze(0)
                right_factor = right_factor.unsqueeze(0)

            # Left factor
            left_interp = _make_sparse_from_indices_and_values(left_interp_indices,
                                                               left_interp_values,
                                                               self.base_lazy_variable.size()[-2])
            left_factor = left_factor.transpose(-1, -2)
            left_res = bdsmm(left_interp, left_factor).transpose(-1, -2).contiguous()

            # Right factor
            right_interp = _make_sparse_from_indices_and_values(right_interp_indices,
                                                                right_interp_values,
                                                                self.base_lazy_variable.size()[-1])
            right_factor = right_factor.transpose(-1, -2)
            right_res = bdsmm(right_interp, right_factor).transpose(-1, -2).contiguous()

            res = tuple(list(base_lazy_var_deriv(left_res, right_res)) + [None] * 4)
            return res

        return closure

    def _transpose_nonbatch(self):
        res = self.__class__(self.base_lazy_variable.transpose(-1, -2), self.right_interp_indices,
                             self.right_interp_values,
                             self.left_interp_indices, self.left_interp_values)
        return res

    def diag(self):
        n_data, n_interp = self.left_interp_indices.size()

        # Batch compute the non-zero values of the outer products w_left^k w_right^k^T
        left_interp_values = self.left_interp_values.unsqueeze(2)
        right_interp_values = self.right_interp_values.unsqueeze(1)
        interp_values = torch.matmul(left_interp_values, right_interp_values)

        # Batch compute Toeplitz values that will be non-zero for row k
        left_interp_indices = self.left_interp_indices.unsqueeze(2).expand(n_data, n_interp, n_interp).contiguous()
        right_interp_indices = self.right_interp_indices.unsqueeze(1).expand(n_data, n_interp, n_interp).contiguous()
        base_var_vals = self.base_lazy_variable._get_indices(left_interp_indices.view(-1),
                                                             right_interp_indices.view(-1))
        base_var_vals = base_var_vals.view(left_interp_indices.size())

        diag = (interp_values * base_var_vals).sum(2).sum(1)
        return diag

    def repeat(self, *sizes):
        """
        Repeat elements of the Variable.
        Right now it only works to create a batched version of a InterpolatedLazyVariable.

        e.g. `var.repeat(3, 1, 1)` creates a batched version of length 3
        """
        if not len(sizes) == 3 and sizes[1] == 1 and sizes[2] == 1:
            raise RuntimeError('Repeat only works to create a batched version at the moment.')

        return self.__class__(self.base_lazy_variable, self.left_interp_indices.repeat(*sizes),
                              self.left_interp_values.repeat(*sizes),
                              self.right_interp_indices.repeat(*sizes),
                              self.right_interp_values.repeat(*sizes))

    def representation(self):
        return tuple(list(self.base_lazy_variable.representation()) +
                     [self.left_interp_indices, self.left_interp_values, self.right_interp_indices,
                      self.right_interp_values])

    def size(self):
        if self.left_interp_indices.ndimension() == 3:
            return torch.Size((self.left_interp_indices.size(0), self.left_interp_indices.size(1),
                               self.right_interp_indices.size(1)))
        else:
            return torch.Size((self.left_interp_indices.size(0), self.right_interp_indices.size(0)))

    def __getitem__(self, index):
        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))

        # Check that left interp index and right interp indices are not scalar values
        squeeze_left = False
        squeeze_right = False
        if isinstance(index[-2], int):
            index[-2] = slice(index[-2], index[-2] + 1, None)
            squeeze_left = True
        if isinstance(index[-1], int):
            index[-1] = slice(index[-1], index[-1] + 1, None)
            squeeze_right = True

        base_lazy_variable = self.base_lazy_variable
        left_interp_indices = self.left_interp_indices
        left_interp_values = self.left_interp_values
        right_interp_indices = self.right_interp_indices
        right_interp_values = self.right_interp_values

        # Handle batch dimensions
        isbatch = ndimension >= 3
        if isbatch:
            batch_index = tuple(index[:-2])
            base_lazy_variable = self.base_lazy_variable[batch_index]
            left_interp_indices = self.left_interp_indices[batch_index]
            left_interp_values = self.left_interp_values[batch_index]
            right_interp_indices = self.right_interp_indices[batch_index]
            right_interp_values = self.right_interp_values[batch_index]

        ndimension = base_lazy_variable.ndimension()

        # Handle left interp
        left_index = tuple([slice(None, None, None)] * (ndimension - 2) + [index[-2]])
        left_interp_indices = left_interp_indices[left_index]
        left_interp_values = left_interp_values[left_index]

        # Handle right interp
        right_index = tuple([slice(None, None, None)] * (ndimension - 2) + [index[-1]])
        right_interp_indices = right_interp_indices[right_index]
        right_interp_values = right_interp_values[right_index]

        res = self.__class__(base_lazy_variable, left_interp_indices, left_interp_values,
                             right_interp_indices, right_interp_values)

        if squeeze_left or squeeze_right:
            res = res.evaluate()
            if squeeze_left:
                res = res.squeeze(-2)
            if squeeze_right:
                res = res.squeeze(-1)

        return res
