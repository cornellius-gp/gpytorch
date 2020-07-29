from ..models.deep_gps.dspp import DSPP
from ._approximate_mll import _ApproximateMarginalLogLikelihood


class DeepPredictiveLogLikelihood(_ApproximateMarginalLogLikelihood):
    """
    An implementation of the predictive log likelihood extended to DSPPs as discussed in Jankowiak et al., 2020.

    If you are using a DSPP model, this is the loss object you want to create and optimize over.

    This loss object is compatible only with models of type :obj:~gpytorch.models.deep_gps.DSPP
    """

    def __init__(self, likelihood, model, num_data, beta=1.0, combine_terms=True):
        if not combine_terms:
            raise ValueError(
                "The base marginal log likelihood object should combine terms "
                "when used in conjunction with a DeepApproximateMLL."
            )

        if not isinstance(model, DSPP):
            raise ValueError("DeepPredictiveLogLikelihood can only be used with a DSPP model.")

        super().__init__(likelihood, model, num_data, beta, combine_terms)

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        base_log_marginal = self.likelihood.log_marginal(target, approximate_dist_f, **kwargs)
        deep_log_marginal = self.model.quad_weights.unsqueeze(-1) + base_log_marginal

        deep_log_prob = deep_log_marginal.logsumexp(dim=0)

        return deep_log_prob.sum(-1)
