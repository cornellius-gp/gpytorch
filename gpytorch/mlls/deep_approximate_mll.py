from ._approximate_mll import _ApproximateMarginalLogLikelihood


class DeepApproximateMLL(_ApproximateMarginalLogLikelihood):
    """
    A wrapper to make a GPyTorch approximate marginal log likelihoods compatible with Deep GPs.

    Example:
        >>> deep_mll = gpytorch.mlls.DeepApproximateMLL(
        >>>     gpytorch.mlls.VariationalELBO(likelihood, model, num_data=1000)
        >>> )

    :param ~gpytorch.mlls._ApproximateMarginalLogLikelihood base_mll: The base
        approximate MLL
    """

    def __init__(self, base_mll):
        if not base_mll.combine_terms:
            raise ValueError(
                "The base marginal log likelihood object should combine terms "
                "when used in conjunction with a DeepApproximateMLL."
            )
        super().__init__(base_mll.likelihood, base_mll.model, num_data=base_mll.num_data, beta=base_mll.beta)
        self.base_mll = base_mll

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        return self.base_mll._log_likelihood_term(approximate_dist_f, target, **kwargs).mean(0)

    def forward(self, approximate_dist_f, target, **kwargs):
        return self.base_mll.forward(approximate_dist_f, target, **kwargs).mean(0)
