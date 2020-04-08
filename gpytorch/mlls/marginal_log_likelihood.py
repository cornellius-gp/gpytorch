#!/usr/bin/env python3

from ..models import GP
from ..module import Module


class MarginalLogLikelihood(Module):
    r"""
    These are modules to compute (or approximate/bound) the marginal log likelihood
    (MLL) of the GP model when applied to data.  I.e., given a GP :math:`f \sim
    \mathcal{GP}(\mu, K)`, and data :math:`\mathbf X, \mathbf y`, these modules
    compute/approximate

    .. math::

       \begin{equation*}
          \mathcal{L} = p_f(\mathbf y \! \mid \! \mathbf X)
          = \int p \left( \mathbf y \! \mid \! f(\mathbf X) \right) \: p(f(\mathbf X) \! \mid \! \mathbf X) \: d f
       \end{equation*}

    This is computed exactly when the GP inference is computed exactly (e.g. regression w/ a Gaussian likelihood).
    It is approximated/bounded for GP models that use approximate inference.

    These models are typically used as the "loss" functions for GP models (though note that the output of
    these functions must be negated for optimization).
    """

    def __init__(self, likelihood, model):
        super(MarginalLogLikelihood, self).__init__()
        if not isinstance(model, GP):
            raise RuntimeError(
                "All MarginalLogLikelihood objects must be given a GP object as a model. If you are "
                "using a more complicated model involving a GP, pass the underlying GP object as the "
                "model, not a full PyTorch module."
            )
        self.likelihood = likelihood
        self.model = model

    def forward(self, output, target, **kwargs):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and `\mathbf y`

        :param ~gpytorch.distributions.MultivariateNormal output: the outputs of the latent function
            (the :obj:`~gpytorch.models.GP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param dict kwargs: Additional arguments to pass to the likelihood's :attr:`forward` function.
        """
        raise NotImplementedError

    def pyro_factor(self, output, target):
        """
        As forward, but register the MLL with pyro using the pyro.factor primitive.
        """
        raise NotImplementedError
