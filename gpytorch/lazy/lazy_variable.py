class LazyVariable(object):
    def evaluate(self):
        raise NotImplementedError

    def add_diag(self, diag):
        raise NotImplementedError

    def gp_marginal_log_likelihood(self, target):
        raise NotImplementedError
