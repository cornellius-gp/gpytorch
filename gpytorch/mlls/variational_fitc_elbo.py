#!/usr/bin/env python3

from ._approximate_mll import _ApproximateMarginalLogLikelihood


class VariationalFITCELBO(_ApproximateMarginalLogLikelihood):
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        return self.likelihood.fitc_likelihood(target, approximate_dist_f, **kwargs).sum(-1)

    def forward(self, approximate_dist_f, target, **kwargs):
        return super().forward(approximate_dist_f, target, **kwargs)
