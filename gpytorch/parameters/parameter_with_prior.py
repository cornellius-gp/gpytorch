from torch import nn


class ParameterWithPrior(object):
    def __init__(self, param, prior):
        super(ParameterWithPrior, self).__init__()
        if not isinstance(param, nn.Parameter):
            raise RuntimeError('ParameterWithPrior must be initialized with parameters!')
        self.param = param
        self.prior = prior
