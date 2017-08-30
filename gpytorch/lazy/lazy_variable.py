from ..utils import function_factory


class LazyVariable(object):
    def _mm_closure_factory(self, *args):
        """
        Generates a closure that performs a *tensor* matrix multiply
        The closure will take in a *tensor* matrix (not variable) and return the
        result of a matrix multiply with the lazy variable.

        The arguments into the closure factory are the *tensors* corresponding to
        the Variables in self.representation()

        Returns:
        function(tensor (nxn)) - closure that performs a matrix multiply
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
        function(k_mm_closure, tr_inv, k_inv_y, y) - closure
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

    def exact_gp_marginal_log_likelihood(self, target, num_samples=10):
        """
        Computes the marginal log likelihood of a Gaussian process whose covariance matrix
        plus the diagonal noise term (added using add_diag above) is stored as this lazy variable

        Args:
            - target (vector n) - training label vector to be used in the marginal log likelihood calculation.
        Returns:
            - scalar - The GP marginal log likelihood where (K+\sigma^{2}I) is represented by this LazyVariable.
        """
        if not hasattr(self, '_gp_mll_class'):
            grad_closure_factory = self._exact_gp_mll_grad_closure_factory
            self._gp_mll_class = function_factory.exact_gp_mll_factory(self._mm_closure_factory, grad_closure_factory)
        args = list(self.representation()) + [target]
        return self._gp_mll_class(num_samples)(*args)

    def invmm(self, rhs_mat):
        """
        Computes a linear solve (w.r.t self) with several right hand sides.

        Args:
            - rhs_mat (matrix nxk) - Matrix of k right hand side vectors.

        Returns:
            - matrix nxk - (self)^{-1} rhs_mat
        """
        if not hasattr(self, '_invmm_class'):
            grad_fn = self._grad_fn if hasattr(self, '_grad_fn') else None
            self._invmm_class = function_factory.invmm_factory(self._mm_closure_factory, grad_fn)
        args = list(self.representation()) + [rhs_mat]
        return self._invmm_class()(*args)

    def mm(self, rhs_mat):
        """
        Multiplies self by a matrix

        Args:
            - rhs_mat (matrix nxk) - Matrix to multiply with

        Returns:
            - matrix nxk
        """
        if not hasattr(self, '_mm_class'):
            grad_fn = self._grad_fn if hasattr(self, '_grad_fn') else None
            self._mm_class = function_factory.mm_factory(self._mm_closure_factory, grad_fn)
        args = list(self.representation()) + [rhs_mat]
        return self._mm_class()(*args)

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar, num_samples):
        """
        Performs Monte Carlo integration of the provided log_probability function. Typically, this should work by
        drawing samples of u from the variational posterior, transforming these in to samples of f using the information
        stored in this LazyVariable, and then calling the log_probability_func with these samples and train_y.

        Args:
            - log_probability_func (function) - Log probability function to integrate.
            - train_y (vector n) - Training label vector.
            - variational_mean (vector m) - Mean vector of the variational posterior.
            - chol_var_covar (matrix m x m) - Cholesky decomposition of the variational posterior covariance matrix.
            - num_samples (scalar) - Number of samples to use for Monte Carlo integration.
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

    def trace_log_det_quad_form(self, mu_diffs, chol_covar_1, num_samples=10):
        if not hasattr(self, '_trace_log_det_quad_form_class'):
            tlqf_function_factory = function_factory.trace_logdet_quad_form_factory
            self._trace_log_det_quad_form_class = tlqf_function_factory(self._mm_closure_factory,
                                                                        self._derivative_quadratic_form_factory)
        covar2_args = self.representation()
        return self._trace_log_det_quad_form_class(num_samples)(mu_diffs, chol_covar_1, *covar2_args)

    def __getitem__(self, index):
        raise NotImplementedError
