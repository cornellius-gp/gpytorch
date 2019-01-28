#! /usr/bin/env python3

from gpytorch.likelihoods import Likelihood
from torch.nn import ModuleList


def _get_tuple_args_(*args):
    for arg in args:
        if isinstance(arg, tuple):
            yield arg
        else:
            yield (arg,)


class LikelihoodList(Likelihood):
    def __init__(self, *likelihoods):
        super().__init__()
        self.likelihoods = ModuleList(likelihoods)

    def forward(self, *args):
        return [likelihood.forward(*args_) for likelihood, args_ in zip(self.likelihoods, _get_tuple_args_(*args))]
