import logging
from lbfgs import LBFGS


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

            except RuntimeError as e:
                if 'not positive definite' in e.message and self.n_trials < self.max_trials:
                    if self.catch_function:
                        result = self.catch_function(*args, **kwargs)
                    self.n_trials += 1
                    if self.n_trials % self.log_interval == 0:
                        logging.warning('Not PD matrix: %d more attempts' % (self.max_trials - self.n_trials))

                else:
                    raise e

            return result
        return wrapped_function


__all__ = [LBFGS, pd_catcher]
