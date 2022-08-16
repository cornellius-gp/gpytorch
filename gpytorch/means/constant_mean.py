#!/usr/bin/env python3

import warnings
from typing import Any, Optional

import torch

from ..constraints import Interval
from ..priors import Prior
from ..utils.warnings import OldVersionWarning
from .mean import Mean


def _ensure_updated_strategy_flag_set(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    if prefix + "constant" in state_dict:
        constant = state_dict.pop(prefix + "constant").squeeze(-1)  # Remove deprecated singleton dimension
        state_dict[prefix + "raw_constant"] = constant
        warnings.warn(
            "You have loaded a GP model with a ConstantMean  from a previous version of "
            "GPyTorch. The mean module parameter `constant` has been renamed to `raw_constant`. "
            "Additionally, the shape of `raw_constant` is now *batch_shape, whereas the shape of "
            "`constant` was *batch_shape x 1. "
            "We have updated the name/shape of the parameter in your state dict, but we recommend that you "
            "re-save your model.",
            OldVersionWarning,
        )


class ConstantMean(Mean):
    r"""
    A (non-zero) constant prior mean function, i.e.:

    .. math::
        \mu(\mathbf x) = C

    where :math:`C` is a learned constant.

    :param constant_prior: Prior for constant parameter :math:`C`.
    :type constant_prior: ~gpytorch.priors.Prior, optional
    :param constant_constraint: Constraint for constant parameter :math:`C`.
    :type constant_constraint: ~gpytorch.priors.Interval, optional
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor constant: :math:`C` parameter
    """

    def __init__(
        self,
        constant_prior: Optional[Prior] = None,
        constant_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ):
        super(ConstantMean, self).__init__()

        # Deprecated kwarg
        constant_prior_deprecated = kwargs.get("prior")
        if constant_prior_deprecated is not None:
            if constant_prior is None:  # Using the old kwarg for the constant_prior
                warnings.warn(
                    "The kwarg `prior` for ConstantMean has been renamed to `constant_prior`, and will be deprecated.",
                    DeprecationWarning,
                )
                constant_prior = constant_prior_deprecated
            else:  # Weird edge case where someone set both `prior` and `constant_prior`
                warnings.warn(
                    "You have set both the `constant_prior` and the deprecated `prior` arguments for ConstantMean. "
                    "`prior` is deprecated, and will be ignored.",
                    DeprecationWarning,
                )

        # Ensure that old versions of the model still load
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)

        self.batch_shape = batch_shape
        self.register_parameter(name="raw_constant", parameter=torch.nn.Parameter(torch.zeros(batch_shape)))
        if constant_prior is not None:
            self.register_prior("mean_prior", constant_prior, self._constant_param, self._constant_closure)
        if constant_constraint is not None:
            self.register_constraint("raw_constant", constant_constraint)

    @property
    def constant(self):
        return self._constant_param(self)

    @constant.setter
    def constant(self, value):
        self._constant_closure(self, value)

    # We need a getter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_param(self, m):
        if hasattr(m, "raw_constant_constraint"):
            return m.raw_constant_constraint.transform(m.raw_constant)
        return m.raw_constant

    # We need a setter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_constant)

        if hasattr(m, "raw_constant_constraint"):
            m.initialize(raw_constant=m.raw_constant_constraint.inverse_transform(value))
        else:
            m.initialize(raw_constant=value)

    def forward(self, input):
        constant = self.constant.unsqueeze(-1)  # *batch_shape x 1
        return constant.expand(torch.broadcast_shapes(constant.shape, input.shape[:-1]))
