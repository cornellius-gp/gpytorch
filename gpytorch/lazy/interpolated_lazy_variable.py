import gpytorch
import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable
from .root_lazy_variable import RootLazyVariable
from ..utils import bdsmm, left_interp, sparse


class InterpolatedLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable, left_interp_indices=None, left_interp_values=None,
                 right_interp_indices=None, right_interp_values=None):
        tensor_cls = base_lazy_variable.tensor_cls

        if left_interp_indices is None:
            n_rows = base_lazy_variable.size()[-2]
            left_interp_indices = Variable(tensor_cls(n_rows).long())
            torch.arange(0, n_rows, out=left_interp_indices.data)
            left_interp_indices = left_interp_indices.unsqueeze(-1)
            if base_lazy_variable.ndimension() == 3:
                left_interp_indices = left_interp_indices.unsqueeze(0).expand(base_lazy_variable.size(0),
                                                                              n_rows, 1)
            elif right_interp_indices is not None and right_interp_indices.ndimension() == 3:
                left_interp_indices = left_interp_indices.unsqueeze(0).expand(right_interp_indices.size(0),
                                                                              n_rows, 1)

        if left_interp_values is None:
            left_interp_values = Variable(tensor_cls(left_interp_indices.size()).fill_(1))

        if right_interp_indices is None:
            n_rows = base_lazy_variable.size()[-2]
            right_interp_indices = Variable(tensor_cls(n_rows).long())
            torch.arange(0, n_rows, out=right_interp_indices.data)
            right_interp_indices = right_interp_indices.unsqueeze(-1)
            if base_lazy_variable.ndimension() == 3:
                right_interp_indices = right_interp_indices.unsqueeze(0).expand(base_lazy_variable.size(0),
                                                                                n_rows, 1)
            elif left_interp_indices.ndimension() == 3:
                right_interp_indices = right_interp_indices.unsqueeze(0).expand(left_interp_indices.size(0),
                                                                                n_rows, 1)

        if right_interp_values is None:
            right_interp_values = Variable(tensor_cls(right_interp_indices.size()).fill_(1))

        super(InterpolatedLazyVariable, self).__init__(base_lazy_variable, left_interp_indices, left_interp_values,
                                                       right_interp_indices, right_interp_values)
        self.base_lazy_variable = base_lazy_variable
        self.left_interp_indices = left_interp_indices
        self.left_interp_values = left_interp_values
        self.right_interp_indices = right_interp_indices
        self.right_interp_values = right_interp_values

    def _matmul_closure_factory(self, *args):
        base_lazy_variable_representation = args[:-4]
        base_lazy_variable_matmul = self.base_lazy_variable._matmul_closure_factory(*base_lazy_variable_representation)
        left_interp_indices, left_interp_values = args[-4:-2]
        right_interp_indices, right_interp_values = args[-2:]

        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(left_interp_indices, left_interp_values)
        right_interp_t = self._sparse_right_interp_t(right_interp_indices, right_interp_values)

        def closure(tensor):
            if tensor.ndimension() == 1:
                is_vector = True
                tensor = tensor.unsqueeze(-1)
            else:
                is_vector = False

            # right_interp^T * tensor
            right_interp_res = bdsmm(right_interp_t, tensor)

            # base_lazy_var * right_interp^T * tensor
            base_res = base_lazy_variable_matmul(right_interp_res)

            # left_interp * base_lazy_var * right_interp^T * tensor
            if len(left_interp_t.size()) == 3:
                left_interp_mat = left_interp_t.transpose(1, 2)
            else:
                left_interp_mat = left_interp_t.t()
            res = bdsmm(left_interp_mat, base_res)

            # Squeeze if necessary
            if is_vector:
                res = res.squeeze(-1)
            return res

        return closure

    def _t_matmul_closure_factory(self, *args):
        base_t_matmul_closure_factory = self.base_lazy_variable._t_matmul_closure_factory
        base_lazy_variable_representation = args[:-4]
        base_lazy_variable_t_matmul = base_t_matmul_closure_factory(*base_lazy_variable_representation)
        left_interp_indices, left_interp_values = args[-4:-2]
        right_interp_indices, right_interp_values = args[-2:]

        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(left_interp_indices, left_interp_values)
        right_interp_t = self._sparse_right_interp_t(right_interp_indices, right_interp_values)

        def closure(tensor):
            if tensor.ndimension() == 1:
                is_vector = True
                tensor = tensor.unsqueeze(-1)
            else:
                is_vector = False

            # left_interp^T * tensor
            left_interp_res = bdsmm(left_interp_t, tensor)

            # base_lazy_var * left_interp^T * tensor
            base_res = base_lazy_variable_t_matmul(left_interp_res)

            # right_interp * base_lazy_var * right_interp^T * tensor
            if len(right_interp_t.size()) == 3:
                right_interp_mat = right_interp_t.transpose(1, 2)
            else:
                right_interp_mat = right_interp_t.t()
            res = bdsmm(right_interp_mat, base_res)

            # Squeeze if necessary
            if is_vector:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        base_lazy_var_repr = args[:-4]
        base_lazy_var_matmul = self.base_lazy_variable._matmul_closure_factory(*base_lazy_var_repr)
        base_lazy_var_t_matmul = self.base_lazy_variable._t_matmul_closure_factory(*base_lazy_var_repr)
        base_lazy_var_deriv = self.base_lazy_variable._derivative_quadratic_form_factory(*base_lazy_var_repr)
        left_interp_indices, left_interp_values = args[-4:-2]
        right_interp_indices, right_interp_values = args[-2:]

        # Get sparse tensor representations of left/right interp matrices
        left_interp_t = self._sparse_left_interp_t(left_interp_indices, left_interp_values)
        right_interp_t = self._sparse_right_interp_t(right_interp_indices, right_interp_values)

        def closure(left_factor, right_factor):
            if left_factor.ndimension() == 1:
                left_factor = left_factor.unsqueeze(0)
                right_factor = right_factor.unsqueeze(0)

            left_factor_t = left_factor.transpose(-1, -2)
            right_factor_t = right_factor.transpose(-1, -2)

            # base_lazy_variable grad
            left_res = bdsmm(left_interp_t, left_factor_t)
            right_res = bdsmm(right_interp_t, right_factor_t)
            base_lv_grad = list(base_lazy_var_deriv(left_res.transpose(-1, -2).contiguous(),
                                                    right_res.transpose(-1, -2).contiguous()))

            # left_interp_values grad
            n_factors = right_res.size(-1)
            n_left_rows = left_interp_indices.size(-2)
            n_right_rows = right_interp_indices.size(-2)
            n_left_interp = left_interp_indices.size(-1)
            n_right_interp = right_interp_indices.size(-1)
            n_inducing = right_res.size(-2)
            if left_interp_indices.ndimension() == 3:
                batch_size = left_interp_indices.size(0)

            # left_interp_values grad
            right_interp_right_res = base_lazy_var_matmul(right_res).contiguous()
            if left_interp_indices.ndimension() == 3:
                batch_offset = left_interp_indices.new(batch_size, 1, 1)
                torch.arange(0, batch_size, out=batch_offset[:, 0, 0])
                batch_offset.mul_(n_inducing)

                batched_left_interp_indices = (left_interp_indices + batch_offset).view(-1)
                flattened_right_interp_right_res = right_interp_right_res.view(batch_size * n_inducing, n_factors)

                selected_right_vals = flattened_right_interp_right_res.index_select(0, batched_left_interp_indices)
                selected_right_vals = selected_right_vals.view(batch_size, n_left_rows, n_left_interp, n_factors)
            else:
                selected_right_vals = right_interp_right_res.index_select(0, left_interp_indices.view(-1))
                selected_right_vals = selected_right_vals.view(n_left_rows, n_left_interp, n_factors)
            left_values_grad = (selected_right_vals * left_factor_t.unsqueeze(-2)).sum(-1)

            # right_interp_values_grad
            left_interp_left_res = base_lazy_var_t_matmul(left_res).contiguous()
            if right_interp_indices.ndimension() == 3:
                batch_offset = right_interp_indices.new(batch_size, 1, 1)
                torch.arange(0, batch_size, out=batch_offset[:, 0, 0])
                batch_offset.mul_(n_inducing)

                batched_right_interp_indices = (right_interp_indices + batch_offset).view(-1)
                flattened_left_interp_left_res = left_interp_left_res.view(batch_size * n_inducing, n_factors)

                selected_left_vals = flattened_left_interp_left_res.index_select(0, batched_right_interp_indices)
                selected_left_vals = selected_left_vals.view(batch_size, n_right_rows, n_right_interp, n_factors)
            else:
                selected_left_vals = left_interp_left_res.index_select(0, right_interp_indices.view(-1))
                selected_left_vals = selected_left_vals.view(n_right_rows, n_right_interp, n_factors)
            right_values_grad = (selected_left_vals * right_factor_t.unsqueeze(-2)).sum(-1)

            res = tuple(base_lv_grad + [None, left_values_grad, None, right_values_grad])
            return res

        return closure

    def _size(self):
        if self.left_interp_indices.ndimension() == 3:
            return torch.Size((self.left_interp_indices.size(0), self.left_interp_indices.size(1),
                               self.right_interp_indices.size(1)))
        else:
            return torch.Size((self.left_interp_indices.size(0), self.right_interp_indices.size(0)))

    def _transpose_nonbatch(self):
        res = self.__class__(self.base_lazy_variable.transpose(-1, -2), self.right_interp_indices,
                             self.right_interp_values,
                             self.left_interp_indices, self.left_interp_values, **self._kwargs)
        return res

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        left_interp_indices = self.left_interp_indices[batch_indices.data, left_indices.data, :]
        left_interp_values = self.left_interp_values[batch_indices.data, left_indices.data, :]
        right_interp_indices = self.right_interp_indices[batch_indices.data, right_indices.data, :]
        right_interp_values = self.right_interp_values[batch_indices.data, right_indices.data, :]

        n_data, n_interp = left_interp_indices.size()

        # Batch compute the non-zero values of the outer products w_left^k w_right^k^T
        left_interp_values = left_interp_values.unsqueeze(-1)
        right_interp_values = right_interp_values.unsqueeze(-2)
        interp_values = torch.matmul(left_interp_values, right_interp_values)

        # Batch compute values that will be non-zero for row k
        left_interp_indices = left_interp_indices.unsqueeze(-1).expand(n_data, n_interp, n_interp)
        right_interp_indices = right_interp_indices.unsqueeze(-2).expand(n_data, n_interp, n_interp)
        left_interp_indices = left_interp_indices.contiguous()
        right_interp_indices = right_interp_indices.contiguous()
        batch_indices = batch_indices.unsqueeze(1).repeat(1, n_interp ** 2).view(-1)
        base_var_vals = self.base_lazy_variable._batch_get_indices(batch_indices, left_interp_indices.view(-1),
                                                                   right_interp_indices.view(-1))
        base_var_vals = base_var_vals.view(left_interp_indices.size())
        res = (interp_values * base_var_vals).sum(-1).sum(-1)
        return res

    def _get_indices(self, left_indices, right_indices):
        left_interp_indices = self.left_interp_indices[left_indices.data, :]
        left_interp_values = self.left_interp_values[left_indices.data, :]
        right_interp_indices = self.right_interp_indices[right_indices.data, :]
        right_interp_values = self.right_interp_values[right_indices.data, :]

        n_data, n_interp = left_interp_indices.size()

        # Batch compute the non-zero values of the outer products w_left^k w_right^k^T
        left_interp_values = left_interp_values.unsqueeze(-1)
        right_interp_values = right_interp_values.unsqueeze(-2)
        interp_values = torch.matmul(left_interp_values, right_interp_values)

        # Batch compute values that will be non-zero for row k
        if left_interp_indices.ndimension() == 3:
            left_interp_indices = left_interp_indices.unsqueeze(-1).expand(n_data, n_interp, n_interp).contiguous()
            right_interp_indices = right_interp_indices.unsqueeze(-2).expand(n_data, n_interp, n_interp).contiguous()
        else:
            left_interp_indices = left_interp_indices.unsqueeze(-1).expand(n_data, n_interp, n_interp).contiguous()
            right_interp_indices = right_interp_indices.unsqueeze(-2).expand(n_data, n_interp, n_interp).contiguous()
        base_var_vals = self.base_lazy_variable._get_indices(left_interp_indices.view(-1),
                                                             right_interp_indices.view(-1))
        base_var_vals = base_var_vals.view(left_interp_indices.size())
        res = (interp_values * base_var_vals).sum(-1).sum(-1)
        return res

    def _sparse_left_interp_t(self, left_interp_indices_tensor, left_interp_values_tensor):
        if hasattr(self, '_sparse_left_interp_t_memo'):
            if torch.equal(self._left_interp_indices_memo, left_interp_indices_tensor) and \
                    torch.equal(self._left_interp_values_memo, left_interp_values_tensor):
                return self._sparse_left_interp_t_memo

        left_interp_t = sparse.make_sparse_from_indices_and_values(left_interp_indices_tensor,
                                                                   left_interp_values_tensor,
                                                                   self.base_lazy_variable.size()[-1])
        self._left_interp_indices_memo = left_interp_indices_tensor
        self._left_interp_values_memo = left_interp_values_tensor
        self._sparse_left_interp_t_memo = left_interp_t
        return self._sparse_left_interp_t_memo

    def _sparse_right_interp_t(self, right_interp_indices_tensor, right_interp_values_tensor):
        if hasattr(self, '_sparse_right_interp_t_memo'):
            if torch.equal(self._right_interp_indices_memo, right_interp_indices_tensor) and \
                    torch.equal(self._right_interp_values_memo, right_interp_values_tensor):
                return self._sparse_right_interp_t_memo

        right_interp_t = sparse.make_sparse_from_indices_and_values(right_interp_indices_tensor,
                                                                    right_interp_values_tensor,
                                                                    self.base_lazy_variable.size()[-1])
        self._right_interp_indices_memo = right_interp_indices_tensor
        self._right_interp_values_memo = right_interp_values_tensor
        self._sparse_right_interp_t_memo = right_interp_t
        return self._sparse_right_interp_t_memo

    def matmul(self, tensor):
        # We're using a custom matmul here, because it is significantly faster than
        # what we get from the function factory.
        # The _matmul_closure is optimized for repeated calls, such as for inv_matmul

        if tensor.ndimension() == 1:
            is_vector = True
            tensor = tensor.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * tensor
        right_interp_t = Variable(sparse.make_sparse_from_indices_and_values(self.right_interp_indices.data,
                                                                             self.right_interp_values.data,
                                                                             self.base_lazy_variable.size()[-1]))
        right_interp_res = gpytorch.dsmm(right_interp_t, tensor)

        # base_lazy_var * right_interp^T * tensor
        base_res = self.base_lazy_variable.matmul(right_interp_res)

        # left_interp * base_lazy_var * right_interp^T * tensor
        res = left_interp(self.left_interp_indices, self.left_interp_values, base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def mul(self, other):
        # We're using a custom method here - the constant mul is applied to the base lazy variable
        # This preserves the interpolated structure
        if not (isinstance(other, Variable) or isinstance(other, LazyVariable)) or \
               (isinstance(other, Variable) and other.numel() == 1):
            from .constant_mul_lazy_variable import ConstantMulLazyVariable
            return self.__class__(ConstantMulLazyVariable(self.base_lazy_variable, other),
                                  self.left_interp_indices, self.left_interp_values,
                                  self.right_interp_indices, self.right_interp_values)
        else:
            return super(InterpolatedLazyVariable, self).mul(other)

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
                              self.right_interp_values.repeat(*sizes), **self._kwargs)

    def root_decomposition(self):
        if isinstance(self.base_lazy_variable, RootLazyVariable):
            interp_root = InterpolatedLazyVariable(self.base_lazy_variable.root, self.left_interp_indices,
                                                   self.left_interp_values)
            return RootLazyVariable(interp_root)
        else:
            super(InterpolatedLazyVariable, self).root_decomposition()

    def root_decomposition_size(self):
        if isinstance(self.base_lazy_variable, RootLazyVariable):
            return self.base_lazy_variable.root_decomposition_size()
        else:
            super(InterpolatedLazyVariable, self).root_decomposition_size()

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
                             right_interp_indices, right_interp_values, **self._kwargs)

        if squeeze_left or squeeze_right:
            res = res.evaluate()
            if squeeze_left:
                res = res.squeeze(-2)
            if squeeze_right:
                res = res.squeeze(-1)

        return res
