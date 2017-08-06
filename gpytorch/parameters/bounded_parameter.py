from torch import nn


class BoundedParameter(object):
    def __init__(self, param, lower_bound, upper_bound):
        super(BoundedParameter, self).__init__()
        if not isinstance(param, nn.Parameter):
            self.param = nn.Parameter(param)
#            raise RuntimeError('ParameterWithPrior must be initialized with parameters!')
        else:
            self.param = param
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
