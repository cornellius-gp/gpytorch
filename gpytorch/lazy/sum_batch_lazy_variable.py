from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
         (true_batch_size * sum_batch_size x _ x _),
         and will return (true_batch_size x _ x _))
        """
        if base_lazy_variable.ndimension() != 3:
            raise RuntimeError(
                "Base lazy variable must be a batch matrix (i.e. 3 dimensions)"
            )
        super(SumBatchLazyVariable, self).__init__(
            base_lazy_variable, sum_batch_size=sum_batch_size
        )
        self.base_lazy_variable = base_lazy_variable
        self.sum_batch_size = sum_batch_size

    def _matmul(self, rhs):
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        rhs = rhs.unsqueeze(0)
        rhs_size = list(rhs.size())
        rhs_size[0] = self.batch_size()
        rhs = rhs.expand(*rhs_size)

        res = self.base_lazy_variable._matmul(rhs)
        if self.sum_batch_size is not None:
            res = res.view(-1, self.sum_batch_size, res.size(1), res.size(2))
            res = res.sum(1)
        else:
            res = res.sum(0)

        if isvector:
            res = res.squeeze(-1)
        return res

    def _t_matmul(self, rhs):
        isvector = rhs.ndimension() == 1
        if isvector:
            rhs = rhs.unsqueeze(1)

        rhs = rhs.unsqueeze(0)
        rhs_size = list(rhs.size())
        rhs_size[0] = self.batch_size()
        rhs = rhs.expand(*rhs_size)

        res = self.base_lazy_variable._t_matmul(rhs)
        if self.sum_batch_size is not None:
            res = res.view(-1, self.sum_batch_size, res.size(1), res.size(2))
            res = res.sum(1)
        else:
            res = res.sum(0)

        if isvector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        left_vecs = left_vecs.unsqueeze(0)
        left_vecs_size = list(left_vecs.size())
        left_vecs_size[0] = self.batch_size()
        left_vecs = left_vecs.expand(*left_vecs_size)

        right_vecs = right_vecs.unsqueeze(0)
        right_vecs_size = list(right_vecs.size())
        right_vecs_size[0] = self.batch_size()
        right_vecs = right_vecs.expand(*right_vecs_size)

        res = self.base_lazy_variable._quad_form_derivative(left_vecs, right_vecs)
        return res

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
        raise RuntimeError("Batch get indices is not implmeneted yet")

    def _get_indices(self, left_indices, right_indices):
        batch_indices = Variable(self.tensor_cls(self.batch_size()).long())
        torch.arange(0, self.batch_size(), out=batch_indices.data)
        batch_indices = batch_indices.unsqueeze(1).repeat(1, len(left_indices)).view(-1)
        left_indices = left_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        right_indices = right_indices.unsqueeze(1).repeat(self.batch_size(), 1).view(-1)
        res = self.base_lazy_variable._batch_get_indices(
            batch_indices, left_indices, right_indices
        )
        return res.view(self.batch_size(), -1).sum(0)

    def batch_size(self):
        return self.base_lazy_variable.size(0)

    def _exact_predictive_covar_inv_quad_form_cache(
        self, train_train_covar_inv_root, test_train_covar
    ):
        if self.sum_batch_size is None:
            train_train_covar_inv_root = train_train_covar_inv_root.unsqueeze(0)
            train_train_covar_inv_root = train_train_covar_inv_root.expand(
                self.base_lazy_variable.size(0),
                train_train_covar_inv_root.size(-2),
                train_train_covar_inv_root.size(-1),
            )
        else:
            train_train_covar_inv_root = train_train_covar_inv_root.repeat(
                self.sum_batch_size, 1, 1
            )
        return self.base_lazy_variable._exact_predictive_covar_inv_quad_form_cache(
            train_train_covar_inv_root, test_train_covar.base_lazy_variable
        )

    def _exact_predictive_covar_inv_quad_form_root(
        self, precomputed_cache, test_train_covar
    ):
        # Here the precomputed cache is a list
        # where each component in the list is the precomputed cache for each component lazy variable
        res = self.base_lazy_variable._exact_predictive_covar_inv_quad_form_root(
            precomputed_cache, test_train_covar.base_lazy_variable
        )
        if self.sum_batch_size is not None:
            res = res.view(-1, self.sum_batch_size, res.size(1), res.size(2))
            res = res.sum(1)
        else:
            res = res.sum(0)
        return res

    def mul(self, other):
        # We're using a custom method here - the constant mul is applied to the base lazy variable
        # This preserves the sum batch structure
        if (
            not (isinstance(other, Variable) or isinstance(other, LazyVariable))
            or (isinstance(other, Variable) and other.numel() == 1)
        ):
            from .constant_mul_lazy_variable import ConstantMulLazyVariable

            return self.__class__(
                ConstantMulLazyVariable(self.base_lazy_variable, other),
                sum_batch_size=self.sum_batch_size,
            )
        else:
            return super(SumBatchLazyVariable, self).mul(other)

    def zero_mean_mvn_samples(self, n_samples):
        n_dim = self.size(-2)
        res = self.base_lazy_variable.zero_mean_mvn_samples(n_samples)
        if self.sum_batch_size is None:
            res = res.view(-1, n_dim, n_samples).sum(0)
        else:
            res = res.view(-1, self.sum_batch_size, n_dim, n_samples).sum(1)
        return res

    def __getitem__(self, index):
        if self.sum_batch_size is None:
            # Add an all-inclusive batch dimension
            if isinstance(index, int):
                index = tuple(slice(None, None, None), int)
            else:
                index = tuple([slice(None, None, None)] + list(index))

            # Do a __getitem__ on the base lazy variable
            res = self.base_lazy_variable.__getitem__(index)
            if isinstance(res, LazyVariable):
                return SumBatchLazyVariable(res, sum_batch_size=None)
            else:
                return res.sum(0)

        # Cases for when there's an inner batch
        else:
            batch_index = index if isinstance(index, int) else index[0]

            # Keeping all batch dimensions - recursion base case
            if batch_index == slice(None, None, None) or batch_index is None:
                # Do a __getitem__ on the base lazy variable
                res = self.base_lazy_variable.__getitem__(index)
                if isinstance(res, LazyVariable):
                    return SumBatchLazyVariable(res, sum_batch_size=self.sum_batch_size)
                else:
                    res = res.view(-1, self.sum_batch_size, res.size(1), res.size(2))
                    return res.sum(1)

            # Construct a new lazy variable
            # Get rid of sum_batch_index if we're choosing one batch variable
            if isinstance(batch_index, int):
                batch_index = slice(
                    batch_index * self.sum_batch_size,
                    (batch_index + 1) * self.sum_batch_size,
                    None,
                )
                sum_batch_size = None

            # Keep sum_batch_index, because we still have an inner batch
            elif isinstance(batch_index, slice):
                start = batch_index.start
                stop = batch_index.stop
                batch_index = slice(
                    start * self.sum_batch_size, stop * self.sum_batch_size, None
                )
                sum_batch_size = self.sum_batch_size

            else:
                raise RuntimeError("Unknown batch index type")

            # Now construct a new sum batch lazy variable, and recurse
            components = tuple(component[batch_index] for component in self._args)
            new_var = self.__class__(*components, sum_batch_size=sum_batch_size)

            # If the index was only on the batch index, we're done
            if isinstance(index, int) or len(index) == 1:
                return new_var

            # Else - recurse
            else:
                index = list(index)
                index[0] = None
                return new_var.__getitem__(tuple(index))
