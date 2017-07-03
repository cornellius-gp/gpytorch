from torch.nn import Parameter
class ParameterWithPrior(object):
    def __init__(self, param, prior):
        super(ParameterWithPrior,self).__init__()
        if not isinstance(param, Parameter):
            raise RuntimeError('ParameterWithPrior must be initialized with parameters!')
        self.param = param
        self.prior = prior