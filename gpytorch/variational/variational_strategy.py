from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class VariationalStrategy(object):
    def __init__(self, variational_dist, prior_dist):
        self.variational_dist = variational_dist
        self.prior_dist = prior_dist

    def kl_divergence(self):
        raise NotImplementedError
