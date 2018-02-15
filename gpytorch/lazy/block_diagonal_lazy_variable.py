import torch
from torch.autograd import Variable
from .lazy_variable import LazyVariable


class BlockDiagonalLazyVariable(LazyVariable):
    def __init__(self, base_lazy_variable, n_blocks=None):
        """
        Represents a lazy variable that is the block diagonal of square matrices
        This variable is stored as a batch
        (i.e. a tensor of batch_size x _ x _)
        Therefore, all the block diagonal components must be the same lazy variable type
        and size

        By specifying n_blocks, you can have two batch variables: one
        that is summed over, and one that is not
        (i.e. the input is the representation of a tensor that is
         (true_batch_size * n_blocks x _ x _),
         and will return (true_batch_size x _ x _))
        """
        if base_lazy_variable.ndimension() != 3:
            raise RuntimeError('Base lazy variable must be a batch matrix (i.e. 3 dimensions)')
        super(BlockDiagonalLazyVariable, self).__init__(base_lazy_variable, n_blocks=n_blocks)
        self.base_lazy_variable = base_lazy_variable
        self.n_blocks = n_blocks

    def _matmul_closure_factory(self, *args):
        super_closure = self.base_lazy_variable._matmul_closure_factory(*args)
        block_size = self.base_lazy_variable.size(-1)

        def closure(tensor):
            isvector = tensor.ndimension() == 1
            if isvector:
                tensor = tensor.unsqueeze(1)

            n_cols = tensor.size(-1)
            tensor = tensor.contiguous().view(-1, block_size, n_cols)

            res = super_closure(tensor)
            if self.n_blocks is not None:
                res = res.contiguous().view(-1, self.n_blocks * res.size(1), res.size(2))
            else:
                res = res.contiguous().view(res.size(0) * res.size(1), res.size(2))

            if isvector:
                res = res.squeeze(-1)
            return res

        return closure

    def _derivative_quadratic_form_factory(self, *args):
        super_closure = self.base_lazy_variable._derivative_quadratic_form_factory(*args)
        block_size = self.base_lazy_variable.size(-1)

        def closure(left_factor, right_factor):
            if left_factor.ndimension() == 1:
                left_factor = left_factor.unsqueeze(0)
                right_factor = right_factor.unsqueeze(0)
            left_factor = left_factor.transpose(-1, -2).contiguous()
            left_factor = left_factor.view(-1, block_size, left_factor.size(-1))
            left_factor = left_factor.transpose(-1, -2).contiguous()
            right_factor = right_factor.transpose(-1, -2).contiguous()
            right_factor = right_factor.view(-1, block_size, right_factor.size(-1))
            right_factor = right_factor.transpose(-1, -2).contiguous()
            res = super_closure(left_factor, right_factor)
            return res

        return closure

    def _size(self):
        base_size = self.base_lazy_variable.size()
        if self.n_blocks is None:
            return torch.Size((base_size[0] * base_size[1], base_size[0] * base_size[2]))
        else:
            true_batch_size = self.base_lazy_variable.size(0) // self.n_blocks
            return torch.Size((true_batch_size, self.n_blocks * base_size[1], self.n_blocks * base_size[2]))

    def _transpose_nonbatch(self):
        return BlockDiagonalLazyVariable(self.base_lazy_variable._transpose_nonbatch())

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        block_size = self.base_lazy_variable.size(-1)
        left_batch_indices = left_indices.div(block_size).long()
        right_batch_indices = left_indices.div(block_size).long()
        batch_indices = batch_indices * block_size + left_batch_indices
        left_indices = left_indices.fmod(block_size)
        right_indices = left_indices.fmod(block_size)

        res = self.base_lazy_variable._batch_get_indices(batch_indices, left_indices, right_indices)
        res = res * torch.eq(left_batch_indices, right_batch_indices).type_as(res)
        return res

    def _get_indices(self, left_indices, right_indices):
        block_size = self.base_lazy_variable.size(-1)
        left_batch_indices = left_indices.div(block_size).long()
        right_batch_indices = left_indices.div(block_size).long()
        left_indices = left_indices.fmod(block_size)
        right_indices = left_indices.fmod(block_size)

        res = self.base_lazy_variable._batch_get_indices(left_batch_indices, left_indices, right_indices)
        res = res * torch.eq(left_batch_indices, right_batch_indices).type_as(res)
        return res

    def mul(self, other):
        # We're using a custom method here - the constant mul is applied to the base lazy variable
        # This preserves the sum batch structure
        if not (isinstance(other, Variable) or isinstance(other, LazyVariable)) or \
               (isinstance(other, Variable) and other.numel() == 1):
            from .constant_mul_lazy_variable import ConstantMulLazyVariable
            return self.__class__(ConstantMulLazyVariable(self.base_lazy_variable, other),
                                  n_blocks=self.n_blocks)
        else:
            return super(BlockDiagonalLazyVariable, self).mul(other)

    def zero_mean_mvn_samples(self, n_samples):
        res = self.base_lazy_variable.zero_mean_mvn_samples(n_samples)
        if self.n_blocks is None:
            res = res.view(-1, n_samples)
        else:
            res = res.view(self.size(0) / self.n_blocks, -1, n_samples)
        return res

    def __getitem__(self, index):
        if self.n_blocks is None:
            return super(BlockDiagonalLazyVariable, self).__getitem__(index)

        # Cases for when there's an inner batch
        else:
            batch_index = index if isinstance(index, int) else index[0]

            # Keeping all batch dimensions - recursion base case
            if batch_index == slice(None, None, None):
                res = super(BlockDiagonalLazyVariable, self).__getitem__(index)
                return res

            # Construct a new lazy variable
            # Get rid of sum_batch_index if we're choosing one batch variable
            if isinstance(batch_index, int):
                batch_index = slice(batch_index * self.n_blocks, (batch_index + 1) * self.n_blocks, None)
                n_blocks = None

            # Keep sum_batch_index, because we still have an inner batch
            elif isinstance(batch_index, slice):
                start, stop, step = batch_index.indices(self.size(0))
                batch_index = slice(start * self.n_blocks, stop * self.n_blocks, step)
                n_blocks = self.n_blocks

            else:
                raise RuntimeError('Unknown batch index type')

            # Now construct a new sum batch lazy variable, and recurse
            components = tuple(component[batch_index] for component in self._args)
            new_var = self.__class__(*components, n_blocks=n_blocks)

            # If the index was only on the batch index, we're done
            if isinstance(index, int) or len(index) == 1:
                return new_var

            # Else - recurse
            else:
                if new_var.n_blocks is None:
                    index = index[1:]
                else:
                    index = list(index)
                    index[0] = slice(None, None, None)
                return new_var.__getitem__(tuple(index))
