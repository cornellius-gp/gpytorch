class LazyVariable(object):
    def evaluate(self):
        """
        Explicitly evaluates the matrix this LazyVariable represents. This
        function should return a Variable explicitly wrapping a Tensor storing
        an exact representation of this LazyVariable.
        """
        raise NotImplementedError

    def add_diag(self, diag):
        """
        Adds a diagonal component to this lazy variable in a potentially lazy fashion.
        That is, if this lazy variable represents some matrix A, add_diag should return
        a LazyVariable of the same type that represents A+dI.

        Args:
            - diag (scalar) - diagonal component to add.
        Returns:
            - LazyVariable representing this current LazyVariable plus diag*I.
        """
        raise NotImplementedError

    def gp_marginal_log_likelihood(self, target):
        """
        Computes the marginal log likelihood of a Gaussian process whose covariance matrix
        plus the diagonal noise term (added using add_diag above) is stored as this lazy variable

        Args:
            - target (vector n) - training label vector to be used in the marginal log likelihood calculation.
        Returns:
            - scalar - The GP marginal log likelihood where (K+\sigma^{2}I) is represented by this LazyVariable.
        """
        raise NotImplementedError

    def mvn_kl_divergence(self, mean_1, chol_covar_1, mean_2):
        """
        Computes the KL divergence between two multivariate Normal distributions. The first of these
        distributions is specified by mean_1 and chol_covar_1, while the second distribution is specified
        by mean_2 and this LazyVariable.

        Args:
            - mean_1 (vector n) - Mean vector of the first Gaussian distribution.
            - chol_covar_1 (matrix n x n) - Cholesky factorization of the covariance matrix of the first Gaussian
                                            distribution.
            - mean_2 (vector n) - Mean vector of the second Gaussian distribution.
        Returns:
            - KL divergence between N(mean_1, chol_covar_1) and N(mean_2, self)
        """
        raise NotImplementedError

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

    def add_jitter_(self):
        """
        Adds jitter (i.e., a small diagonal component) to the matrix this LazyVariable represents.
        This could potentially be implemented as a no-op, however this could lead to numerical instabilities,
        so this should only be done at the user's risk.
        """
        raise NotImplementedError
