from ..utils import function_factory


class LazyVariable(object):
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

    def _exact_gp_mll_grad_closure_factory(self, *args):
        """
        Generates a closure that computes the derivatives of v K^-1 v + log |K|, given v

        K is a square matrix corresponding to the Variables in self.representation()

        Returns:
        function(k_matmul_closure, tr_inv, k_inv_y, y) - closure
        """
        raise NotImplementedError

    def add_diag(self, diag):
        """
        Adds an element to the diagonal of the matrix.

        Args:
            - diag (Scalar Variable)
        """
        raise NotImplementedError

    def add_jitter(self):
        """
        Adds jitter (i.e., a small diagonal component) to the matrix this LazyVariable represents.
        This could potentially be implemented as a no-op, however this could lead to numerical instabilities,
        so this should only be done at the user's risk.
        """

    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyVariable represents. This
        function should return a Variable explicitly wrapping a Tensor storing
        an exact representation of this LazyVariable.
        """
        raise NotImplementedError

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

    def inv_matmul(self, rhs):
        """
        Computes a linear solve (w.r.t self) with several right hand sides.

        Args:
            - rhs (tensor nxk) - Matrix or tensor

        Returns:
            - tensor - (self)^{-1} rhs
        """
        if not hasattr(self, '_inv_matmul_class'):
            if hasattr(self, '_derivative_quadratic_form_factory'):
                dqff = self._derivative_quadratic_form_factory
            else:
                dqff = None
            self._inv_matmul_class = function_factory.inv_matmul_factory(self._matmul_closure_factory, dqff)
        args = list(self.representation()) + [rhs]
        return self._inv_matmul_class()(*args)

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
        args = list(self.representation()) + [tensor]
        return self._matmul_class()(*args)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        """
        Performs Monte Carlo integration of the provided log_probability function. Typically, this should work by
        drawing samples of u from the variational posterior, transforming these in to samples of f using the information
        stored in this LazyVariable, and then calling the log_probability_func with these samples and train_y.

        Args:
            - log_probability_func (function) - Log probability function to integrate.
            - train_y (vector n) - Training label vector.
            - variational_mean (vector m) - Mean vector of the variational posterior.
            - chol_var_covar (matrix m x m) - Cholesky decomposition of the variational posterior covariance matrix.
        Returns:
            - The average of calling log_probability_func on num_samples samples of f, where f is sampled from the
              current posterior.
        """
        raise NotImplementedError

    def mul(self, constant):
        """
        Multiplies this interpolated Toeplitz matrix elementwise by a constant. To accomplish this,
        we multiply the Toeplitz component by the constant. This way, the interpolation acts on the
        multiplied values in T, and the entire kernel is ultimately multiplied by this constant.

        Args:
            - constant (broadcastable with self.c) - Constant to multiply by.
        Returns:
            - ToeplitzLazyVariable with c = c*(constant)
        """
        raise NotImplementedError

    def mul_(self, constant):
        """
        In-place version of mul.
        """
        raise NotImplementedError

    def posterior_strategy(self):
        """
        Return a PosteriorStrategy object for computing the GP posterior.
        """
        raise NotImplementedError

    def representation(self, *args):
        """
        Returns the variables that are used to define the LazyVariable
        """
        raise NotImplementedError

    def size(self):
        """
        Returns the size of the resulting Variable that the lazy variable represents
        """
        raise NotImplementedError

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
        raise NotImplementedError
