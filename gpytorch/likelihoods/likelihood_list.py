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

    def expected_log_prob(self, *args, **kwargs):
        return [
            likelihood.expected_log_prob(*args_, **kwargs)
            for likelihood, args_ in zip(self.likelihoods, _get_tuple_args_(*args))
        ]

    def pyro_sample_output(self, *args, **kwargs):
        return [
            likelihood.pyro_sample_output(*args_, **kwargs)
            for likelihood, args_ in zip(self.likelihoods, _get_tuple_args_(*args))
        ]

    def __call__(self, *args, **kwargs):
        if "noise" in kwargs:
            noise = kwargs.pop("noise")
            # if noise kwarg is passed, assume it's an iterable of noise tensors
            return [
                likelihood(*args_, {**kwargs, "noise": noise_})
                for likelihood, args_, noise_ in zip(
                    self.likelihoods, _get_tuple_args_(*args), noise
                )
            ]
        else:
            return [
                likelihood(*args_, **kwargs)
                for likelihood, args_ in zip(self.likelihoods, _get_tuple_args_(*args))
            ]
