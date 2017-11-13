import gpytorch


class VariationalStrategy(object):
    def __init__(self, variational_mean, chol_variational_covar, inducing_output):
        self.variational_mean = variational_mean
        # Negate each row with a negative diagonal (the Cholesky decomposition
        # of a matrix requires that the diagonal elements be positive).
        chol_variational_covar = chol_variational_covar.triu()
        inside = chol_variational_covar.diag().sign().unsqueeze(1).expand_as(chol_variational_covar).triu()
        self.chol_variational_covar = inside.mul(chol_variational_covar)

        self.inducing_output = inducing_output

    def mvn_kl_divergence(self):
        mean_diffs = self.inducing_output.mean() - self.variational_mean

        logdet_variational_covar = self.chol_variational_covar.diag().log().sum(0) * 2
        trace_logdet_quad_form = gpytorch.trace_logdet_quad_form(mean_diffs, self.chol_variational_covar,
                                                                 gpytorch.add_jitter(self.inducing_output.covar()))

        # Compute the KL Divergence.
        res = 0.5 * (trace_logdet_quad_form - logdet_variational_covar - len(mean_diffs))
        return res

    def variational_samples(self, output, n_samples=None):
        raise NotImplementedError
