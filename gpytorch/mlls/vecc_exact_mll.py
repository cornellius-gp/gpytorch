#! /usr/bin/env python3

from .marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.distributions.vecc_multivariate_normal import VeccMultivariateNormal


class VeccExactMLL(MarginalLogLikelihood):

    def __init__(self, base_mll):
        super().__init__(base_mll.likelihood, base_mll.model)
        self.base_mll = base_mll
        self.blocks = base_mll.model.blocks

    def forward(self, outputs, targets, **kwargs):
        if not isinstance(outputs, VeccMultivariateNormal):
            raise RuntimeError("VeccExactMLL can only operate on block multivariate normal random variables.")

        res = sum(self.base_mll(output, targets[this_block])
                  for output, this_block in zip(outputs, self.blocks.blocks))

        # Scale by the number of blocks we have
        num_blocks = len(outputs.bmvn)
        return res.div_(num_blocks)
