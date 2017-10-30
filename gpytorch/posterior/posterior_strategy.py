import gpytorch
import torch
from ..lazy import LazyVariable
from ..utils import StochasticLQ


class PosteriorStrategy(object):
    def __init__(self, var):
        """
        Args:
        - var (LazyVariable) - the variable to define the PosteriorStrategy for
        """
        self.var = var

    def alpha_size(self):
        """
        Returns the size of the alpha vector
        """
        raise NotImplementedError

    def lanczos_size(self):
        """
        Returns the sizes of the lanczos decompositon matrices (Q and T)
        """
        var_size = self.var.size()[1]
        max_iter = min(var_size, gpytorch.functions.max_lanczos_iterations)
        q_size = torch.Size((var_size, max_iter))
        t_size = torch.Size((max_iter, max_iter))
        return q_size, t_size

    def exact_posterior_alpha(self, train_mean, train_y):
        """
        Assumes that self.var represents the train-train prior covariance matrix.
        ((Lazy)Variable nxn)

        Returns alpha - a vector to memoize for calculating the
        mean of the posterior GP on test points

        Args:
            - train_mean (Variable n) - prior mean values for the test points.
            - train_y (Variable n) - alpha vector, computed from exact_posterior_alpha
        """
        raise NotImplementedError

    def exact_posterior_lanczos(self):
        """
        Computes the lanczos decomposition of the train_train_covariance matrix
        ((Lazy)Variable nxn)

        k -- the rank of the Lanczos decomposition -- is determined by
        gpytorch.functions.max_lanczos_iterations

        Returns:
            - q_mat (matrix (nxk)) - Lanczos Q matrix of train/train covariance matrix
            - t_mat (matrix (kxk)) - Lanczos Q matrix of train/train covariance matrix
        """
        if isinstance(self.var, LazyVariable):
            train_train_representation = [var.data for var in self.var.representation()]
            train_train_matmul = self.var._matmul_closure_factory(*train_train_representation)
            tensor_cls = type(train_train_representation[0])
        else:
            train_train_matmul = self.var.data.matmul
            tensor_cls = type(self.var.data)

        n_train = self.var.size()[0]
        max_iter = min(n_train, gpytorch.functions.max_lanczos_iterations)
        lq_object = StochasticLQ(cls=tensor_cls, max_iter=max_iter)
        init_vector = tensor_cls(n_train, 1).normal_()
        init_vector /= torch.norm(init_vector, 2, 0)
        q_mat, t_mat = lq_object.lanczos_batch(train_train_matmul, init_vector)
        return q_mat[0], t_mat[0]

    def exact_posterior_mean(self, test_mean, alpha):
        """
        Assumes that self.var represents the test-train prior covariance matrix.
        ((Lazy)Variable mxn)

        Returns the mean of the posterior GP on test points, given
        prior means/covars

        Args:
            - test_mean (Variable m) - prior mean values for the test points.
            - alpha (Variable m) - alpha vector, computed from exact_posterior_alpha
        """
        raise NotImplementedError

    def exact_posterior_covar(self, test_train_covar, train_test_covar, test_test_covar):
        """
        Returns the covar of the posterior GP on test points, given
        prior means/covars

        Assumes self.var is train_train_covar (prior covariance matrix between train points)
        ((Lazy)Variable nxn)

        Args:
            - test_train_covar ((Lazy)Variable nxm) - prior covariance matrix between test and training points.
                                                      Usually, this is simply the transpose of train_test_covar.
            - train_test_covar ((Lazy)Variable nxm) - prior covariance matrix between training and test points.
            - test_test_covar ((Lazy)Variable mxm) - prior covariance matrix between test points
        """
        from ..lazy import NonLazyVariable, MatmulLazyVariable
        if isinstance(train_test_covar, LazyVariable):
            train_test_covar = train_test_covar.evaluate()
        if isinstance(test_train_covar, LazyVariable):
            test_train_covar = train_test_covar.t()
        if not isinstance(test_test_covar, LazyVariable):
            test_test_covar = NonLazyVariable(test_test_covar)

        covar_correction_rhs = gpytorch.inv_matmul(self.var, train_test_covar).mul_(-1)
        return test_test_covar + MatmulLazyVariable(test_train_covar, covar_correction_rhs)

    def exact_posterior_covar_fast(self, lanczos_q_var, lanczos_t_var):
        """
        Returns the covar of the posterior GP on test points, given
        prior means/covars

        Assumes self.var is the full prior covar
        ((Lazy)Variable (n+m)x(n+m))

        Args:
            - lanczos_q_var (Variable nxk) - Q matrix of Lanczos decomposition of train/train covariance matrix
            - lanczos_t_var (Variable kxk) - T matrix of Lanczos decomposition of train/train covariance matrix
        """
        from ..lazy import NonLazyVariable, MatmulLazyVariable
        n_train = lanczos_q_var.size(0)
        test_train_covar = self.var[n_train:, :n_train]
        test_test_covar = self.var[n_train:, n_train:]
        if not isinstance(test_test_covar, LazyVariable):
            test_test_covar = NonLazyVariable(test_test_covar)

        covar_correction_lhs = test_train_covar.matmul(lanczos_q_var)
        covar_correction_rhs = covar_correction_lhs.matmul(lanczos_t_var.inverse()).mul_(-1)
        return test_test_covar + MatmulLazyVariable(covar_correction_lhs, covar_correction_rhs.t())

    def monte_carlo_log_likelihood(self, log_probability_func, train_y, variational_mean, chol_var_covar):
        """
        Computes the expected likelihood component of the variational marginal log likelihood, using MC integration

        Assumes self.var is the training prior covariance
        ((Lazy)Variable nxn)

        Args:
            - log_probability_func
            - train_y
            - variational_mean
            - chol_var_covar
        """
        raise NotImplementedError

    def variational_posterior_alpha(self, variational_mean):
        """
        Returns alpha - a vector to memoize for calculating the
        mean of the posterior GP on test points

        Assumes self.var is induc_induc_covar - prior covariance matrix between inducing points
        ((Lazy)Variable nxn)

        Args:
            - variational_mean (Variable n) - prior variatoinal mean
        """
        raise NotImplementedError

    def variational_posterior_mean(self, alpha):
        """
        Assumes self.var is the covariance matrix between test and inducing points
        ((Lazy)Variable mxn)

        Returns the mean of the posterior GP on test points, given
        prior means/covars

        Args:
            - alpha (Variable m) - alpha vector, computed from exact_posterior_alpha
        """
        raise NotImplementedError

    def variational_posterior_covar(self, induc_test_covar, chol_variational_covar,
                                    test_test_covar, induc_induc_covar):
        """
        Assumes self.var is the covariance matrix between test and inducing points
        ((Lazy)Variable mxn)

        Returns the covar of the posterior GP on test points, given
        prior covars

        Args:
            - induc_test_covar ((Lazy)Variable nxm) - prior covariance matrix between inducing and test points.
            - chol_variational_covar (Variable nxn) - Cholesky decomposition of variational covar
            - test_test_covar ((Lazy)Variable nxm) - prior covariance matrix between test and inducing points.
                                                     Usually, this is simply the transpose of induc_test_covar.
            - induc_induc_covar ((Lazy)Variable nxn) - inducing-inducing covariance matrix.
                                                       Usually takes the form (K+sI), where K is the prior
                                                       covariance matrix and s is the noise variance.
        """
        raise NotImplementedError
