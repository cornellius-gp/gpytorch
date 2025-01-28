#!/usr/bin/env python3

import warnings
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution

from ..distributions import base_distributions, MultitaskMultivariateNormal
from ..priors import Prior
from .likelihood import Likelihood


class SoftmaxLikelihood(Likelihood):
    r"""
    Implements the Softmax (multiclass) likelihood used for GP classification.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf W \mathbf f \right)

    :math:`\mathbf W` is a set of linear mixing weights applied to the latent functions :math:`\mathbf f`.

    :param num_features: Dimensionality of latent function :math:`\mathbf f`.
    :param num_classes: Number of classes.
    :param mixing_weights: (Default: `True`) Whether to learn a linear mixing weight :math:`\mathbf W` applied to
        the latent function :math:`\mathbf f`. If `False`, then :math:`\mathbf W = \mathbf I`.
    :param mixing_weights_prior: Prior to use over the mixing weights :math:`\mathbf W`.

    :ivar torch.Tensor mixing_weights: (Optional) mixing weights.
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        num_classes: int = None,  # pyre-fixme[9]
        mixing_weights: bool = True,
        mixing_weights_prior: Optional[Prior] = None,
    ) -> None:
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        self.num_classes = num_classes
        if mixing_weights:
            if num_features is None:
                raise ValueError("num_features is required with mixing weights")
            self.num_features: int = num_features
            self.register_parameter(
                name="mixing_weights",
                parameter=torch.nn.Parameter(torch.randn(num_classes, num_features).div_(num_features)),
            )
            if mixing_weights_prior is not None:
                self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")
        else:
            self.num_features = num_classes
            self.mixing_weights: Optional[torch.nn.Parameter] = None

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Categorical:
        num_data, num_features = function_samples.shape[-2:]

        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function_samples = function_samples.transpose(-1, -2)
            num_data, num_features = function_samples.shape[-2:]

        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        if self.mixing_weights is not None:
            mixed_fs = function_samples @ self.mixing_weights.t()  # num_classes x num_data
        else:
            mixed_fs = function_samples
        res = base_distributions.Categorical(logits=mixed_fs)
        return res

    def __call__(self, input: Union[Tensor, MultitaskMultivariateNormal], *args: Any, **kwargs: Any) -> Distribution:
        if isinstance(input, Distribution) and not isinstance(input, MultitaskMultivariateNormal):
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            input = MultitaskMultivariateNormal.from_batch_mvn(input)
        return super().__call__(input, *args, **kwargs)
