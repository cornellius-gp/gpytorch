import torch
from torch.autograd import Variable
from .interpolated_lazy_variable import InterpolatedLazyVariable


class SumInterpolatedLazyVariable(InterpolatedLazyVariable):
    def __init__(self, base_lazy_variable, left_interp_indices, left_interp_values,
                 right_interp_indices, right_interp_values):
        if not left_interp_indices.ndimension() == 3 or not right_interp_indices.ndimension() == 3:
            raise RuntimeError

        if not left_interp_indices.size() == left_interp_values.size() or not \
                left_interp_indices.size() == left_interp_values.size():
            raise RuntimeError

        if not left_interp_indices.size(0) == right_interp_indices.size(0):
            raise RuntimeError

        if base_lazy_variable.ndimension() == 2:
            base_lazy_variable = base_lazy_variable.repeat(left_interp_indices.size(0), 1, 1)
        elif base_lazy_variable.ndimension() != 3:
            raise RuntimeError

        super(SumInterpolatedLazyVariable, self).__init__(base_lazy_variable, left_interp_indices, left_interp_values,
                                                          right_interp_indices, right_interp_values)
        self.base_lazy_variable = base_lazy_variable
        self.left_interp_indices = left_interp_indices
        self.left_interp_values = left_interp_values
        self.right_interp_indices = right_interp_indices
        self.right_interp_values = right_interp_values
        self.tensor_cls = type(self.base_lazy_variable.representation()[0].data)

    def _matmul_closure_factory(self, *args):
        super_closure = super(SumInterpolatedLazyVariable, self)._matmul_closure_factory(*args)
        batch_size = self.left_interp_indices.size(0)

        def closure(tensor):
            isvector = tensor.ndimension() == 1
            if isvector:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.unsqueeze(0)
            tensor_size = list(tensor.size())
            tensor_size[0] = batch_size
            tensor = tensor.expand(*tensor_size)

            res = super_closure(tensor).sum(0)
            if isvector:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        super_closure = super(SumInterpolatedLazyVariable, self)._derivative_quadratic_form_factory(*args)
        batch_size = self.left_interp_indices.size(0)

        def closure(left_factor, right_factor):
            left_factor = left_factor.unsqueeze(0)
            left_factor_size = list(left_factor.size())
            left_factor_size[0] = batch_size
            left_factor = left_factor.expand(*left_factor_size)

            right_factor = right_factor.unsqueeze(0)
            right_factor_size = list(right_factor.size())
            right_factor_size[0] = batch_size
            right_factor = right_factor.expand(*right_factor_size)

            res = super_closure(left_factor, right_factor)
            return res

        return closure

    def diag(self):
        batch_size, n_data, n_interp = self.left_interp_indices.size()

        # Batch compute the non-zero values of the outer products w_left^k w_right^k^T
        left_interp_values = self.left_interp_values.unsqueeze(3)
        right_interp_values = self.right_interp_values.unsqueeze(2)
        interp_values = torch.matmul(left_interp_values, right_interp_values)

        # Batch compute Toeplitz values that will be non-zero for row k
        left_interp_indices = self.left_interp_indices.unsqueeze(3).expand(batch_size, n_data, n_interp, n_interp)
        left_interp_indices = left_interp_indices.contiguous()
        right_interp_indices = self.right_interp_indices.unsqueeze(2).expand(batch_size, n_data, n_interp, n_interp)
        right_interp_indices = right_interp_indices.contiguous()
        batch_interp_indices = Variable(left_interp_indices.data.new(batch_size))
        torch.arange(0, batch_size, out=batch_interp_indices.data)
        batch_interp_indices = batch_interp_indices.view(batch_size, 1, 1, 1)
        batch_interp_indices = batch_interp_indices.expand(batch_size, n_data, n_interp, n_interp).contiguous()
        base_var_vals = self.base_lazy_variable._batch_get_indices(batch_interp_indices.view(-1),
                                                                   left_interp_indices.view(-1),
                                                                   right_interp_indices.view(-1))
        base_var_vals = base_var_vals.view(left_interp_indices.size())

        diag = (interp_values * base_var_vals).sum(3).sum(2).sum(0)
        return diag

    def size(self):
        return torch.Size((self.left_interp_indices.size(1), self.right_interp_indices.size(1)))
