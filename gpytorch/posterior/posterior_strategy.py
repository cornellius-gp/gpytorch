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
