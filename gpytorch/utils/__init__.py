import torch
from torch.autograd import Variable
import logging
from .interpolation import Interpolation
from .lbfgs import LBFGS
from .lincg import LinearCG
from .slq_logdet import SLQLogDet


__all__ = [
    Interpolation,
    LBFGS,
    LinearCG,
    SLQLogDet,
]


class pd_catcher(object):
    '''
    A decorator to deal with non-positive definite matrices (useful during optimization)
    If an error due to non-psotiive definiteness occurs when calling the function, we
    retry the function call a certain number of times.
    After a certain number of trials, it fails.
    '''
    def __init__(self, catch_function=None, max_trials=20, log_interval=5):
        self.catch_function = catch_function
        self.n_trials = 0
        self.max_trials = max_trials
        self.log_interval = log_interval

    def __call__(self, function):
        def wrapped_function(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
                self.n_trials = 0
            except (ZeroDivisionError, RuntimeError) as e:
                if self.n_trials < self.max_trials:
                    if self.catch_function:
                        result = self.catch_function(*args, **kwargs)
                    self.n_trials += 1
                    if self.n_trials % self.log_interval == 0:
                        logging.warning('Not PD matrix: %d more attempts' % (self.max_trials - self.n_trials))
                else:
                    raise e
            return result
        return wrapped_function


def reverse(input, dim=0):
    """
    Reverses a tensor
    Args:
        - input: tensor to reverse
        - dim: dimension to reverse on
    Returns:
        - reversed input
    """
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return input.index_select(dim, reverse_index)


def rcumsum(input, dim=0):
    """
    Computes a reverse cumulative sum
    Args:
        - input: tensor
        - dim: dimension to reverse on
    Returns:
        - rcumsum on input
    """
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return torch.index_select(input, dim, reverse_index).cumsum(dim).index_select(dim, reverse_index)


def approx_equal(self, other, epsilon=1e-4):
    """
    Determines if two tensors are approximately equal
    Args:
        - self: tensor
        - other: tensor
    Returns:
        - bool
    """
    if isinstance(self, Variable):
        self = self.data
    if isinstance(other, Variable):
        other = other.data
    return torch.max((self - other).abs()) <= epsilon


__all__ = [LBFGS, pd_catcher]
