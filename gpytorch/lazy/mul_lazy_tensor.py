from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor
from .root_lazy_tensor import RootLazyTensor
from ..utils import prod


class MulLazyTensor(LazyTensor):
    def __init__(self, *lazy_vars):
        """
        Args:
            - lazy_vars (A list of LazyTensor) - A list of LazyTensor to multiplicate with.
        """
        lazy_vars = list(lazy_vars)
        if len(lazy_vars) == 1:
            if isinstance(lazy_vars[0], NonLazyTensor):
                self._non_lazy_self = lazy_vars
            else:
                raise RuntimeError("MulLazyTensor can only have one LazyTensor if it is a NonLazyTensor")
        else:
            for i, lazy_var in enumerate(lazy_vars):
                if not isinstance(lazy_var, LazyTensor):
                    if torch.is_tensor(lazy_var):
                        lazy_vars[i] = NonLazyTensor(lazy_var)
                    else:
                        raise RuntimeError("All arguments of a MulLazyTensor should be LazyTensors or Tensors")

        super(MulLazyTensor, self).__init__(*lazy_vars)
        self.lazy_vars = lazy_vars

    @property
    def non_lazy_self(self):
        if hasattr(self, "_non_lazy_self"):
            return self._non_lazy_self[0]
        elif len(self._args) == 1:
            return self._args[0]
        else:
            return None

    @property
    def left_lazy_var(self):
        return self._args[0]

    @property
    def right_lazy_var(self):
        return self._args[1]

    @property
    def _args(self):
        if not hasattr(self, "_mul_args_memo") and not hasattr(self, "_non_lazy_self"):
            lazy_vars = sorted(
                (lv.evaluate_kernel() for lv in self.lazy_vars), key=lambda lv: lv.root_decomposition_size()
            )

            if any(isinstance(lv, NonLazyTensor) for lv in lazy_vars):
                self._non_lazy_self = [NonLazyTensor(prod([lv.evaluate() for lv in lazy_vars]))]
            else:
                # Sort lazy tensors by root decomposition size (rank)

                # Recursively construct lazy tensors
                # Make sure the recursive components get a mix of low_rank and high_rank variables
                if len(lazy_vars) > 2:
                    interleaved_lazy_vars = lazy_vars[0::2] + lazy_vars[1::2]
                    if len(interleaved_lazy_vars) > 3:
                        left_lazy_var = MulLazyTensor(*interleaved_lazy_vars[: len(interleaved_lazy_vars) // 2])
                        if left_lazy_var.root_decomposition_size() < left_lazy_var.size(-1):
                            left_lazy_var = RootLazyTensor(left_lazy_var.root_decomposition())
                        else:
                            left_lazy_var = NonLazyTensor(left_lazy_var.evaluate())
                    else:
                        # Make sure we're not constructing a MulLazyTensor of length 1
                        left_lazy_var = interleaved_lazy_vars[0]

                    right_lazy_var = MulLazyTensor(*interleaved_lazy_vars[len(interleaved_lazy_vars) // 2 :])
                    if right_lazy_var.root_decomposition_size() < right_lazy_var.size(-1):
                        right_lazy_var = RootLazyTensor(right_lazy_var.root_decomposition())
                    else:
                        right_lazy_var = NonLazyTensor(right_lazy_var.evaluate())
                else:
                    left_lazy_var = lazy_vars[0]
                    right_lazy_var = lazy_vars[1]

                # Choose which we're doing: root decomposition or exact
                if left_lazy_var.root_decomposition_size() < left_lazy_var.size(-1):
                    left_lazy_var = RootLazyTensor(left_lazy_var.root_decomposition())
                    right_lazy_var = RootLazyTensor(right_lazy_var.root_decomposition())

                if isinstance(left_lazy_var, NonLazyTensor) and isinstance(right_lazy_var, NonLazyTensor):
                    self._non_lazy_self = [NonLazyTensor(left_lazy_var.evaluate() * right_lazy_var.evaluate())]
                else:
                    self._mul_args_memo = [left_lazy_var, right_lazy_var]

        if hasattr(self, "_mul_args_memo"):
            return self._mul_args_memo
        else:
            return self._non_lazy_self

    @_args.setter
    def _args(self, args):
        # This is a no-op. We do something different here
        pass

    def _matmul(self, rhs):
        if self.non_lazy_self is not None:
            return self.non_lazy_self._matmul(rhs)

        is_vector = False
        if rhs.ndimension() == 1:
            rhs = rhs.unsqueeze(1)
            is_vector = True
        batch_size = max(rhs.size(0), self.size(0)) if rhs.ndimension() == 3 else None

        # Here we have a root decomposition
        if isinstance(self.left_lazy_var, RootLazyTensor):
            left_root = self.left_lazy_var.root.evaluate()
            rank = left_root.size(-1)
            n = self.size(-1)
            m = rhs.size(-1)
            # Now implement the formula (A . B) v = diag(A D_v B)
            left_res = rhs.unsqueeze(-2) * left_root.unsqueeze(-1)
            left_res = left_res.view(n, rank * m) if batch_size is None else left_res.view(batch_size, n, rank * m)
            left_res = self.right_lazy_var._matmul(left_res)
            left_res = left_res.view(n, rank, m) if batch_size is None else left_res.view(batch_size, n, rank, m)
            res = left_res.mul_(left_root.unsqueeze(-1)).sum(-2)
        # This is the case where we're not doing a root decomposition, because the matrix is too small
        else:
            res = (self.left_lazy_var.evaluate() * self.right_lazy_var.evaluate()).matmul(rhs)
        res = res.squeeze(-1) if is_vector else res
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):
        if self.non_lazy_self is not None:
            return self.non_lazy_self._quad_form_derivative(left_vecs, right_vecs)

        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)
        batch_size = self.size(0) if self.ndimension() == 3 else None

        n = None
        num_vecs = None
        if self.ndimension() == 3:
            _, n, num_vecs = left_vecs.size()
        else:
            n, num_vecs = left_vecs.size()

        if isinstance(self.right_lazy_var, RootLazyTensor):
            right_root = self.right_lazy_var.root.evaluate()
            right_rank = right_root.size(-1)
            left_factor = left_vecs.unsqueeze(-2) * right_root.unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * right_root.unsqueeze(-1)
        else:
            right_rank = n
            eye = self.right_lazy_var.tensor_cls(n).fill_(1).diag()
            left_factor = left_vecs.unsqueeze(-2) * self.right_lazy_var.evaluate().unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * eye.unsqueeze(-1)

        if batch_size is None:
            left_factor = left_factor.view(n, num_vecs * right_rank)
            right_factor = right_factor.view(n, num_vecs * right_rank)
        else:
            left_factor = left_factor.view(batch_size, n, num_vecs * right_rank)
            right_factor = right_factor.view(batch_size, n, num_vecs * right_rank)
        left_deriv_args = self.left_lazy_var._quad_form_derivative(left_factor, right_factor)

        if isinstance(self.left_lazy_var, RootLazyTensor):
            left_root = self.left_lazy_var.root.evaluate()
            left_rank = left_root.size(-1)
            left_factor = left_vecs.unsqueeze(-2) * left_root.unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * left_root.unsqueeze(-1)
        else:
            left_rank = n
            eye = self.left_lazy_var.tensor_cls(n).fill_(1).diag()
            left_factor = left_vecs.unsqueeze(-2) * self.left_lazy_var.evaluate().unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * eye.unsqueeze(-1)

        if batch_size is None:
            left_factor = left_factor.view(n, num_vecs * left_rank)
            right_factor = right_factor.view(n, num_vecs * left_rank)
        else:
            left_factor = left_factor.view(batch_size, n, num_vecs * left_rank)
            right_factor = right_factor.view(batch_size, n, num_vecs * left_rank)
        right_deriv_args = self.right_lazy_var._quad_form_derivative(left_factor, right_factor)

        return tuple(list(left_deriv_args) + list(right_deriv_args))

    def diag(self):
        res = prod([lazy_var.diag() for lazy_var in self.lazy_vars])
        return res

    def evaluate(self):
        res = prod([lazy_var.evaluate() for lazy_var in self.lazy_vars])
        return res

    def mul(self, other):
        if isinstance(other, int) or isinstance(other, float) or (torch.is_tensor(other) and other.numel() == 1):
            lazy_vars = list(self.lazy_vars[:-1])
            lazy_vars.append(self.lazy_vars[-1] * other)
            return MulLazyTensor(*lazy_vars)
        elif isinstance(other, MulLazyTensor):
            res = list(self.lazy_vars) + list(other.lazy_vars)
            return MulLazyTensor(*res)
        elif isinstance(other, NonLazyTensor):
            return NonLazyTensor(self.evaluate() * other.evaluate())
        elif self.non_lazy_self is not None:
            return NonLazyTensor(self.non_lazy_self.evaluate() * other.evaluate())
        elif isinstance(other, LazyTensor):
            return MulLazyTensor(*(list(self.lazy_vars) + [other]))
        else:
            raise RuntimeError("other must be a LazyTensor, int or float.")

    def _size(self):
        return self.lazy_vars[0].size()

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        res = prod(
            [lazy_var._batch_get_indices(batch_indices, left_indices, right_indices) for lazy_var in self.lazy_vars]
        )
        return res

    def _get_indices(self, left_indices, right_indices):
        res = prod([lazy_var._get_indices(left_indices, right_indices) for lazy_var in self.lazy_vars])
        return res

    def _transpose_nonbatch(self):
        # mul.lazy_tensor only works with symmetric matrices
        return self

    def representation(self):
        """
        Returns the Tensors that are used to define the LazyTensor
        """
        if self.non_lazy_self is not None:
            return self.non_lazy_self.representation()
        else:
            return super(MulLazyTensor, self).representation()

    def representation_tree(self):
        if self.non_lazy_self is not None:
            return self.non_lazy_self.representation_tree()
        else:
            return super(MulLazyTensor, self).representation_tree()
