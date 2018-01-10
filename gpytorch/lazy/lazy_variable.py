import torch
from torch.autograd import Variable
from ..utils import function_factory


class LazyVariable(object):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def _matmul_closure_factory(self, *args):
        """
        Generates a closure that performs a *tensor* matrix multiply
        The closure will take in a *tensor* matrix (not variable) and return the
        result of a matrix multiply with the lazy variable.

        The arguments into the closure factory are the *tensors* corresponding to
        the Variables in self.representation()

        Returns:
        function(tensor) - closure that performs a matrix multiply
        """
        raise NotImplementedError

    def _derivative_quadratic_form_factory(self, *args):
        """
        Generates a closure that computes the derivatives of uKv^t w.r.t. `args` given u, v

        K is a square matrix corresponding to the Variables in self.representation()

        Returns:
        function(vector u, vector v) - closure that computes the derivatives of uKv^t w.r.t.
        `args` given u, v
        """
        raise NotImplementedError

    def _size(self):
        """
        Returns the size of the resulting Variable that the lazy variable represents

        Implement this method, rather than size().
        This is because size does some additional work
        """
        raise NotImplementedError

    def _transpose_nonbatch(self):
        """
        Transposes non-batch dimensions (e.g. last two)
        """
        raise NotImplementedError

    def _batch_get_indices(self, batch_indices, left_indices, right_indices):
        """
        Batch version of _get_indices
        For each matrix in batch_indices, returns entries of the matrix,
        indexed by left and right indices
        Only works for batch lazy variables

        """
        raise NotImplementedError

    def _get_indices(self, left_indices, right_indices):
        """
        Returns entries of the matrix, indexed by left and right indices
        Only works for non-batch lazy variables
        """
        raise NotImplementedError

    def add_diag(self, diag):
        """
        Adds an element to the diagonal of the matrix.

        Args:
            - diag (Scalar Variable)
        """
        from .diag_lazy_variable import DiagLazyVariable
        if self.size(-1) != self.size(-2):
            raise RuntimeError('add_diag only defined for square matrices')
        if self.ndimension() == 3:
            return self + DiagLazyVariable(diag.unsqueeze(0).expand(self.size(0), self.size(1)))
        else:
            return self + DiagLazyVariable(diag.expand(self.size(0)))

    def add_jitter(self):
        """
        Adds jitter (i.e., a small diagonal component) to the matrix this LazyVariable represents.
        This could potentially be implemented as a no-op, however this could lead to numerical instabilities,
        so this should only be done at the user's risk.
        """
        diag = Variable(self.tensor_cls(1).fill_(1e-4))
        return self.add_diag(diag)

    def chol_approx_size(self):
        """
        This is used in conjunction with `chol_matmul`.
        This multiplies a Tensor with a low-rank decomposition of the Cholesky decomposition
        This function returns the rank of that approximation, so that you can
        generate Tensors of the appropriate size.
        """
        raise NotImplementedError

    def chol_matmul(self, tensor):
        """
        Multiplies the cholesky decomposition of the lazy variable with a tensor.
        This is useful for sampling from multivariate Gaussians.

        Assumes self represents a positive definite matrix, or a batch of PSD matrices.
        """
        raise NotImplementedError

    def cpu(self):
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, 'cpu'):
                new_args.append(arg.cpu())
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, 'cpu'):
                new_kwargs[name] = val.cpu()
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    def cuda(self, device_id=None):
        new_args = []
        new_kwargs = {}
        for arg in self._args:
            if hasattr(arg, 'cuda'):
                new_args.append(arg.cuda(device_id))
            else:
                new_args.append(arg)
        for name, val in self._kwargs.items():
            if hasattr(val, 'cuda'):
                new_kwargs[name] = val.cuda(device_id)
            else:
                new_kwargs[name] = val
        return self.__class__(*new_args, **new_kwargs)

    def diag(self):
        size = self.size()
        if size[-1] != size[-2]:
            raise RuntimeError('Diag works on square matrices (or batches)')

        row_col_iter = Variable(self.tensor_cls(size[-1]).long())
        torch.arange(0, size[-1], out=row_col_iter.data)
        if self.ndimension() == 3:
            batch_iter = Variable(self.tensor_cls(size[0]).long())
            torch.arange(0, size[0], out=batch_iter.data)
            batch_iter = batch_iter.unsqueeze(1).repeat(1, size[1]).view(-1)
            row_col_iter = row_col_iter.unsqueeze(1).repeat(size[0], 1).view(-1)
            return self._batch_get_indices(batch_iter, row_col_iter, row_col_iter).view(size[0], size[1])
        else:
            return self._get_indices(row_col_iter, row_col_iter)

    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyVariable represents. This
        function should return a Variable explicitly wrapping a Tensor storing
        an exact representation of this LazyVariable.
        """
        size = self.size()
        if len(size) == 2:
            batch_mode = False
            n_rows, n_cols = size
        else:
            batch_mode = True
            batch_size, n_rows, n_cols = size

        if n_rows < n_cols:
            eye = Variable(self.tensor_cls(n_rows).fill_(1)).diag()
            if batch_mode:
                eye = eye.unsqueeze(0).expand(batch_size, n_rows, n_rows)
                return self.transpose(1, 2).matmul(eye).transpose(1, 2).contiguous()
            else:
                return self.t().matmul(eye).t().contiguous()
        else:
            eye = Variable(self.tensor_cls(n_cols).fill_(1)).diag()
            if batch_mode:
                eye = eye.unsqueeze(0).expand(batch_size, n_cols, n_cols)
            return self.matmul(eye)

    def exact_gp_marginal_log_likelihood(self, target):
        """
        Computes the marginal log likelihood of a Gaussian process whose covariance matrix
        plus the diagonal noise term (added using add_diag above) is stored as this lazy variable

        Args:
            - target (vector n) - training label vector to be used in the marginal log likelihood calculation.
        Returns:
            - scalar - The GP marginal log likelihood where (K+\sigma^{2}I) is represented by this LazyVariable.
        """
        if not hasattr(self, '_gp_mll_class'):
            dqff = self._derivative_quadratic_form_factory
            self._gp_mll_class = function_factory.exact_gp_mll_factory(self._matmul_closure_factory,
                                                                       dqff)
        args = list(self.representation()) + [target]
        return self._gp_mll_class()(*args)

    def inv_matmul(self, tensor):
        """
        Computes a linear solve (w.r.t self) with several right hand sides.

        Args:
            - tensor (tensor nxk) - Matrix or tensor

        Returns:
            - tensor - (self)^{-1} tensor
        """
        if not hasattr(self, '_inv_matmul_class'):
            if hasattr(self, '_derivative_quadratic_form_factory'):
                dqff = self._derivative_quadratic_form_factory
            else:
                dqff = None
            self._inv_matmul_class = function_factory.inv_matmul_factory(self._matmul_closure_factory, dqff)

        lazy_var = self
        if lazy_var.ndimension() == 3 and tensor.ndimension() == 3:
            if lazy_var.size(0) == 1 and tensor.size(0) > 1:
                lazy_var = lazy_var.repeat(tensor.size(0), 1, 1)
            elif tensor.size(0) == 1:
                tensor = tensor.expand(lazy_var.size(0), tensor.size(1), tensor.size(2))
        elif self.ndimension() > 3 or tensor.ndimension() > 3:
            raise RuntimeError

        args = list(lazy_var.representation()) + [tensor]
        return lazy_var._inv_matmul_class()(*args)

    def matmul(self, tensor):
        """
        Multiplies self by a matrix

        Args:
            - tensor (matrix nxk) - Matrix or vector to multiply with

        Returns:
            - tensor
        """
        if not hasattr(self, '_matmul_class'):
            if hasattr(self, '_derivative_quadratic_form_factory'):
                dqff = self._derivative_quadratic_form_factory
            else:
                dqff = None
            self._matmul_class = function_factory.matmul_factory(self._matmul_closure_factory, dqff)

        lazy_var = self
        if lazy_var.ndimension() == 3 and tensor.ndimension() == 3:
            if lazy_var.size(0) == 1 and tensor.size(0) > 1:
                lazy_var = lazy_var.repeat(tensor.size(0), 1, 1)
            elif tensor.size(0) == 1:
                tensor = tensor.expand(lazy_var.size(0), tensor.size(1), tensor.size(2))
        elif self.ndimension() > 3 or tensor.ndimension() > 3:
            raise RuntimeError

        args = list(lazy_var.representation()) + [tensor]
        return lazy_var._matmul_class()(*args)

    def mul(self, other):
        """
        Multiplies the matrix by a constant, or elementwise the matrix by another matrix
        """
        if not (isinstance(other, Variable) or isinstance(other, LazyVariable)) or \
               (isinstance(other, Variable) and other.numel() == 1):
            from .constant_mul_lazy_variable import ConstantMulLazyVariable
            return ConstantMulLazyVariable(self, other)
        else:
            from .mul_lazy_variable import MulLazyVariable
            return MulLazyVariable(self, other)

    def ndimension(self):
        """
        Returns the number of dimensions
        """
        return len(self.size())

    def posterior_strategy(self):
        """
        Return a PosteriorStrategy object for computing the GP posterior.
        """
        from ..posterior import DefaultPosteriorStrategy
        return DefaultPosteriorStrategy(self)

    def representation(self, *args):
        """
        Returns the variables that are used to define the LazyVariable
        """
        representation = []
        for arg in self._args:
            if isinstance(arg, Variable):
                representation.append(arg)
            elif isinstance(arg, LazyVariable):
                representation += list(arg.representation())
            else:
                raise RuntimeError('Representation of a LazyVariable should consist only of Variables')
        return tuple(representation)

    def size(self, val=None):
        """
        Returns the size of the resulting Variable that the lazy variable represents
        """
        size = self._size()
        if val is not None:
            return size[val]
        return size

    def transpose(self, dim1, dim2):
        """
        Returns the transpose of the resulting Variable that the lazy variable represents
        """
        ndimension = self.ndimension()
        if dim1 < 0:
            dim1 = ndimension + dim1
        if dim2 < 0:
            dim2 = ndimension + dim2
        if dim1 >= ndimension or dim2 >= ndimension or not isinstance(dim1, int) or not isinstance(dim2, int):
            raise RuntimeError('Invalid dimension')

        # Batch case
        if dim1 < ndimension - 2 and dim2 < ndimension - 2:
            res = self.__class__(*(arg.transpose(dim1, dim2) for arg in self._args))

        elif dim1 >= ndimension - 2 and dim2 >= ndimension - 2:
            res = self._transpose_nonbatch()

        else:
            raise RuntimeError('Cannot transpose batch dimension with non-batch dimension')

        return res

    def t(self):
        """
        Returns the transpose of the resulting Variable that the lazy variable represents
        """
        if self.ndimension() != 2:
            raise RuntimeError('Cannot call t for more than 2 dimensions')
        return self.transpose(0, 1)

    @property
    def tensor_cls(self):
        if not hasattr(self, '_tensor_cls'):
            self._tensor_cls = type(self.representation()[0].data)
        return self._tensor_cls

    def trace_log_det_quad_form(self, mu_diffs, chol_covar_1):
        if not hasattr(self, '_trace_log_det_quad_form_class'):
            tlqf_function_factory = function_factory.trace_logdet_quad_form_factory
            self._trace_log_det_quad_form_class = tlqf_function_factory(self._matmul_closure_factory,
                                                                        self._derivative_quadratic_form_factory)
        covar2_args = self.representation()
        return self._trace_log_det_quad_form_class()(mu_diffs, chol_covar_1, *covar2_args)

    def __add__(self, other):
        from .sum_lazy_variable import SumLazyVariable
        return SumLazyVariable(self, other)

    def __div__(self, other):
        return self.mul(1. / other)

    def __mul__(self, other):
        return self.mul(other)

    def __getitem__(self, index):
        from .interpolated_lazy_variable import InterpolatedLazyVariable

        index = list(index) if isinstance(index, tuple) else [index]
        ndimension = self.ndimension()
        index += [slice(None, None, None)] * (ndimension - len(index))
        components = list(self._args)

        squeeze_left = False
        squeeze_right = False
        if isinstance(index[-2], int):
            index[-2] = slice(index[-2], index[-2] + 1, None)
            squeeze_left = True
        if isinstance(index[-1], int):
            index[-1] = slice(index[-1], index[-1] + 1, None)
            squeeze_right = True

        # Handle batch dimensions
        isbatch = ndimension >= 3
        if isbatch:
            batch_index = tuple(index[:-2])
            for i, item in enumerate(components):
                components[i] = item[batch_index]

        new_lazy_variable = self.__class__(*components)
        representation = new_lazy_variable.representation()
        ndimension = new_lazy_variable.ndimension()

        # Handle index
        left_index = index[-2]
        right_index = index[-1]

        if left_index == slice(None, None, None) and right_index == slice(None, None, None):
            return new_lazy_variable

        batch_sizes = list(new_lazy_variable.size()[:-2])
        left_row_iter = representation[0].data.new(new_lazy_variable.size()[-2]).long()
        right_row_iter = representation[0].data.new(new_lazy_variable.size()[-1]).long()
        torch.arange(0, new_lazy_variable.size()[-2], out=left_row_iter)
        torch.arange(0, new_lazy_variable.size()[-1], out=right_row_iter)

        left_interp_indices = left_row_iter[left_index].unsqueeze(-1)
        right_interp_indices = right_row_iter[right_index].unsqueeze(-1)

        left_interp_len = len(left_interp_indices)
        right_interp_len = len(right_interp_indices)
        for i in range(ndimension - 2):
            left_interp_indices.unsqueeze_(0)
            right_interp_indices.unsqueeze_(0)

        left_interp_indices = left_interp_indices.expand(*(batch_sizes + [left_interp_len, 1]))
        left_interp_values = left_interp_indices.new(left_interp_indices.size()).fill_(1).float()
        right_interp_indices = right_interp_indices.expand(*(batch_sizes + [right_interp_len, 1]))
        right_interp_values = right_interp_indices.new(right_interp_indices.size()).fill_(1).float()

        res = InterpolatedLazyVariable(new_lazy_variable, Variable(left_interp_indices),
                                       Variable(left_interp_values),
                                       Variable(right_interp_indices), Variable(right_interp_values))

        if squeeze_left or squeeze_right:
            res = res.evaluate()
            if squeeze_left:
                res = res.squeeze(-2)
            if squeeze_right:
                res = res.squeeze(-1)

        return res

    def __setattr__(self, name, val):
        if torch.is_tensor(val) or isinstance(val, Variable) or isinstance(val, LazyVariable):
            if not hasattr(self, '_args'):
                raise RuntimeError('Cannot assign %s to LazyVariable before calling LazyVariable.__init__()' % name)
        object.__setattr__(self, name, val)
