import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable


class SumBatchLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable, sum_batch_size=None):
        """
        Representat a lazy variable that is the sum of many lazy variables
        Unlike with a SumLazyVariable, this variable is stored as a batch
        (i.e. a tensor of batch_size x _ x _)

        By specifying sum_batch_size, you can have two batch variables: one
        that is summed over, and one that is not
        (i.e. the input is the representation of a tensor that is
         (sum_batch_size * inner_batch_size x _ x _),
         and will return (inner_batch_size x _ x _))
        """
        if base_lazy_variable.ndimension() != 3:
            raise RuntimeError('Base lazy variable must be a batch matrix (i.e. 3 dimensions)')
        super(SumBatchLazyVariable, self).__init__(base_lazy_variable)
        self.base_lazy_variable = base_lazy_variable
        self.sum_batch_size = sum_batch_size

    def _matmul_closure_factory(self, *args):
        super_closure = self.base_lazy_variable._matmul_closure_factory(*args)

        def closure(tensor):
            isvector = tensor.ndimension() == 1
            if isvector:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.unsqueeze(0)
            tensor_size = list(tensor.size())
            tensor_size[0] = self.batch_size()
            tensor = tensor.expand(*tensor_size)

            res = super_closure(tensor)
            if self.sum_batch_size is not None:
                res = res.view(self.sum_batch_size, -1, res.size(1), res.size(2))
            res = res.sum(0)

            if isvector:
                res = res.squeeze(-1)
            return res

        return closure

    def _t_matmul_closure_factory(self, *args):
        super_closure = self.base_lazy_variable._t_matmul_closure_factory(*args)

        def closure(tensor):
            isvector = tensor.ndimension() == 1
            if isvector:
                tensor = tensor.unsqueeze(1)

            tensor = tensor.unsqueeze(0)
            tensor_size = list(tensor.size())
            tensor_size[0] = self.batch_size()
            tensor = tensor.expand(*tensor_size)

            res = super_closure(tensor)
            if self.sum_batch_size is not None:
                res = res.view(self.sum_batch_size, -1, res.size(1), res.size(2))
            res = res.sum(0)

            if isvector:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        super_closure = self.base_lazy_variable._derivative_quadratic_form_factory(*args)

        def closure(left_factor, right_factor):
            left_factor = left_factor.unsqueeze(0)
            left_factor_size = list(left_factor.size())
            left_factor_size[0] = self.batch_size()
            left_factor = left_factor.expand(*left_factor_size)

            right_factor = right_factor.unsqueeze(0)
            right_factor_size = list(right_factor.size())
            right_factor_size[0] = self.batch_size()
            right_factor = right_factor.expand(*right_factor_size)

            res = super_closure(left_factor, right_factor)
            return res

        return closure

    def _size(self):
        base_size = self.base_lazy_variable.size()
        if self.sum_batch_size is None:
            return torch.Size(list(base_size)[1:])
        else:
            inner_batch_size = self.batch_size() - self.sum_batch_size
            return torch.Size(list([inner_batch_size]) + list(base_size)[1:])

    def _transpose_nonbatch(self):
        return SumBatchLazyVariable(self.base_lazy_variable._transpose_nonbatch())

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        raise RuntimeError('Batch get indices is not implmeneted yet')

    def _get_indices(self, left_indices, right_indices):
        batch_indices = Variable(self.tensor_cls(self.batch_size()).long())
        torch.arange(0, self.batch_size(), out=batch_indices.data)
        batch_indices = batch_indices.unsqueeze(1).repeat(1, len(left_indices)).view(-1)
        left_indices = left_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        right_indices = right_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        res = self.base_lazy_variable._batch_get_indices(batch_indices, left_indices, right_indices)
        return res.view(self.batch_size(), -1).sum(0)

    def batch_size(self):
        return self.base_lazy_variable.size(0)

    def zero_mean_mvn_samples(self, n_samples):
        n_dim = self.size(-2)
        res = self.base_lazy_variable.zero_mean_mvn_samples(n_samples)
        if self.sum_batch_size is None:
            res = res.view(-1, n_dim, n_samples).sum(0)
        else:
            res = res.view(self.sum_batch_size, -1, n_dim, n_samples).sum(0)
        return res

    def __getitem__(self, index):
        if self.sum_batch_size is None:
            return super(SumBatchLazyVariable, self).__getitem__(index)

        # Cases for when there's an inner batch
        else:
            batch_index = index if isinstance(index, int) else index[0]

            # Keeping all batch dimensions - recursion base case
            if batch_index == slice(None, None, None) or batch_index is None:
                return super.__getitem__(self, index)

            # Construct a new lazy variable
            # Get rid of sum_batch_index if we're choosing one batch variable
            if isinstance(batch_index, int):
                batch_index = slice(batch_index * self.sum_batch_size, (batch_index + 1) * self.sum_batch_size, None)
                sum_batch_size = None

            # Keep sum_batch_index, because we still have an inner batch
            elif isinstance(batch_index, slice):
                start = batch_index.start
                stop = batch_index.stop
                batch_index = slice(start * self.sum_batch_size, stop * self.sum_batch_size, None)
                sum_batch_size = self.sum_batch_size

            else:
                raise RuntimeError('Unknown batch index type')

            # Now construct a new sum batch lazy variable, and recurse
            components = tuple(component[batch_index] for component in self._args)
            new_var = self.__class__(*components, sum_batch_size=sum_batch_size)

            # If the index was only on the batch index, we're done
            if isinstance(index, int):
                return new_var

            # Else - recurse
            else:
                index = list(index)
                index[0] = None
                return new_var.__getitem__(tuple(index))
