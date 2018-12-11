#!/usr/bin/env python3

import torch

from ..utils import prod
from ..utils.memoize import cached
from .lazy_tensor import LazyTensor
from .non_lazy_tensor import NonLazyTensor
from .root_lazy_tensor import RootLazyTensor


class MulLazyTensor(LazyTensor):
    def __init__(self, *lazy_tensors):
        """
        Args:
            - lazy_tensors (A list of LazyTensor) - A list of LazyTensor to multiplicate with.
        """
        lazy_tensors = list(lazy_tensors)
        if len(lazy_tensors) == 1:
            if isinstance(lazy_tensors[0], NonLazyTensor):
                self._non_lazy_self = lazy_tensors
            else:
                raise RuntimeError("MulLazyTensor can only have one LazyTensor if it is a NonLazyTensor")
        else:
            for i, lazy_tensor in enumerate(lazy_tensors):
                if not isinstance(lazy_tensor, LazyTensor):
                    if torch.is_tensor(lazy_tensor):
                        lazy_tensors[i] = NonLazyTensor(lazy_tensor)
                    else:
                        raise RuntimeError("All arguments of a MulLazyTensor should be LazyTensors or Tensors")

        super(MulLazyTensor, self).__init__(*lazy_tensors)
        self.lazy_tensors = lazy_tensors

    @property
    def non_lazy_self(self):
        if hasattr(self, "_non_lazy_self"):
            return self._non_lazy_self[0]
        elif len(self._args) == 1:
            return self._args[0]
        else:
            return None

    @property
    def left_lazy_tensor(self):
        return self._args[0]

    @property
    def right_lazy_tensor(self):
        return self._args[1]

    @property
    def _args(self):
        if not hasattr(self, "_mul_args_memo") and not hasattr(self, "_non_lazy_self"):
            lazy_tensors = sorted(
                (lv.evaluate_kernel() for lv in self.lazy_tensors), key=lambda lv: lv.root_decomposition_size()
            )

            if any(isinstance(lv, NonLazyTensor) for lv in lazy_tensors):
                self._non_lazy_self = [NonLazyTensor(prod([lv.evaluate() for lv in lazy_tensors]))]
            else:
                # Sort lazy tensors by root decomposition size (rank)

                # Recursively construct lazy tensors
                # Make sure the recursive components get a mix of low_rank and high_rank variables
                if len(lazy_tensors) > 2:
                    interleaved_lazy_tensors = lazy_tensors[0::2] + lazy_tensors[1::2]
                    if len(interleaved_lazy_tensors) > 3:
                        left_lazy_tensor = MulLazyTensor(
                            *interleaved_lazy_tensors[: len(interleaved_lazy_tensors) // 2]
                        )
                        if left_lazy_tensor.root_decomposition_size() < left_lazy_tensor.size(-1):
                            left_lazy_tensor = left_lazy_tensor.root_decomposition()
                        else:
                            left_lazy_tensor = NonLazyTensor(left_lazy_tensor.evaluate())
                    else:
                        # Make sure we're not constructing a MulLazyTensor of length 1
                        left_lazy_tensor = interleaved_lazy_tensors[0]

                    right_lazy_tensor = MulLazyTensor(*interleaved_lazy_tensors[len(interleaved_lazy_tensors) // 2 :])
                    if right_lazy_tensor.root_decomposition_size() < right_lazy_tensor.size(-1):
                        right_lazy_tensor = right_lazy_tensor.root_decomposition()
                    else:
                        right_lazy_tensor = NonLazyTensor(right_lazy_tensor.evaluate())
                else:
                    left_lazy_tensor = lazy_tensors[0]
                    right_lazy_tensor = lazy_tensors[1]

                # Choose which we're doing: root decomposition or exact
                if left_lazy_tensor.root_decomposition_size() < left_lazy_tensor.size(-1):
                    left_lazy_tensor = left_lazy_tensor.root_decomposition()
                    right_lazy_tensor = right_lazy_tensor.root_decomposition()

                if isinstance(left_lazy_tensor, NonLazyTensor) and isinstance(right_lazy_tensor, NonLazyTensor):
                    self._non_lazy_self = [NonLazyTensor(left_lazy_tensor.evaluate() * right_lazy_tensor.evaluate())]
                else:
                    self._mul_args_memo = [left_lazy_tensor, right_lazy_tensor]

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
        if isinstance(self.left_lazy_tensor, RootLazyTensor):
            left_root = self.left_lazy_tensor.root.evaluate()
            left_res = rhs.unsqueeze(-2) * left_root.unsqueeze(-1)

            rank = left_root.size(-1)
            n = self.size(-1)
            m = rhs.size(-1)
            # Now implement the formula (A . B) v = diag(A D_v B)
            left_res = left_res.view(n, rank * m) if batch_size is None else left_res.view(batch_size, n, rank * m)
            left_res = self.right_lazy_tensor._matmul(left_res)
            left_res = left_res.view(n, rank, m) if batch_size is None else left_res.view(batch_size, n, rank, m)
            res = left_res.mul_(left_root.unsqueeze(-1)).sum(-2)
        # This is the case where we're not doing a root decomposition, because the matrix is too small
        else:
            res = (self.left_lazy_tensor.evaluate() * self.right_lazy_tensor.evaluate()).matmul(rhs)
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

        if isinstance(self.right_lazy_tensor, RootLazyTensor):
            right_root = self.right_lazy_tensor.root.evaluate()
            left_factor = left_vecs.unsqueeze(-2) * right_root.unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * right_root.unsqueeze(-1)

            right_rank = right_root.size(-1)
        else:
            right_rank = n
            eye = torch.eye(n, dtype=self.right_lazy_tensor.dtype, device=self.right_lazy_tensor.device)
            left_factor = left_vecs.unsqueeze(-2) * self.right_lazy_tensor.evaluate().unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * eye.unsqueeze(-1)

        if batch_size is None:
            left_factor = left_factor.view(n, num_vecs * right_rank)
            right_factor = right_factor.view(n, num_vecs * right_rank)
        else:
            left_factor = left_factor.view(batch_size, n, num_vecs * right_rank)
            right_factor = right_factor.view(batch_size, n, num_vecs * right_rank)
        left_deriv_args = self.left_lazy_tensor._quad_form_derivative(left_factor, right_factor)

        if isinstance(self.left_lazy_tensor, RootLazyTensor):
            left_root = self.left_lazy_tensor.root.evaluate()
            left_factor = left_vecs.unsqueeze(-2) * left_root.unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * left_root.unsqueeze(-1)

            left_rank = left_root.size(-1)
        else:
            left_rank = n
            eye = torch.eye(n, dtype=self.left_lazy_tensor.dtype, device=self.left_lazy_tensor.device)
            left_factor = left_vecs.unsqueeze(-2) * self.left_lazy_tensor.evaluate().unsqueeze(-1)
            right_factor = right_vecs.unsqueeze(-2) * eye.unsqueeze(-1)

        if batch_size is None:
            left_factor = left_factor.view(n, num_vecs * left_rank)
            right_factor = right_factor.view(n, num_vecs * left_rank)
        else:
            left_factor = left_factor.view(batch_size, n, num_vecs * left_rank)
            right_factor = right_factor.view(batch_size, n, num_vecs * left_rank)
        right_deriv_args = self.right_lazy_tensor._quad_form_derivative(left_factor, right_factor)

        return tuple(list(left_deriv_args) + list(right_deriv_args))

    def clone(self):
        return self.__class__(*tuple(lazy_tensor.clone() for lazy_tensor in self.lazy_tensors))

    def detach_(self):
        if hasattr(self, "_mul_args_memo"):
            del self._mul_args_memo
        for lazy_tensor in self.lazy_tensors:
            lazy_tensor.detach_()
        return self

    def diag(self):
        res = prod([lazy_tensor.diag() for lazy_tensor in self.lazy_tensors])
        return res

    @cached
    def evaluate(self):
        return prod([lazy_tensor.evaluate() for lazy_tensor in self.lazy_tensors])

    def mul(self, other):
        if isinstance(other, int) or isinstance(other, float) or (torch.is_tensor(other) and other.numel() == 1):
            lazy_tensors = list(self.lazy_tensors[:-1])
            lazy_tensors.append(self.lazy_tensors[-1] * other)
            return MulLazyTensor(*lazy_tensors)
        elif isinstance(other, MulLazyTensor):
            res = list(self.lazy_tensors) + list(other.lazy_tensors)
            return MulLazyTensor(*res)
        elif isinstance(other, NonLazyTensor):
            return NonLazyTensor(self.evaluate() * other.evaluate())
        elif self.non_lazy_self is not None:
            return NonLazyTensor(self.non_lazy_self.evaluate() * other.evaluate())
        elif isinstance(other, LazyTensor):
            return MulLazyTensor(*(list(self.lazy_tensors) + [other]))
        else:
            raise RuntimeError("other must be a LazyTensor, int or float.")

    def _size(self):
        return self.lazy_tensors[0].size()

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        res = prod(
            [lazy_tensor._get_indices(left_indices, right_indices, *batch_indices) for lazy_tensor in self.lazy_tensors]
        )
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

    @property
    def requires_grad(self):
        if hasattr(self, "_mul_args_memo"):
            del self._mul_args_memo
        return any(lazy_tensor for lazy_tensor in self.lazy_tensors)

    @requires_grad.setter
    def requires_grad(self, val):
        if hasattr(self, "_mul_args_memo"):
            del self._mul_args_memo
        for lazy_tensor in self.lazy_tensors:
            lazy_tensor.requires_grad = val
