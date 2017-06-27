import torch
from torch.autograd import Variable
from gpytorch import Distribution
from gpytorch.math.functions import AddDiag, Invmv

class Mean(Distribution):
    def initialize(self, **kwargs):
        for param_name, param_value in kwargs.items():
            if hasattr(self, param_name):
                if isinstance(param_value, torch.Tensor):
                    getattr(self, param_name).data.copy_(param_value)
                else:
                    getattr(self, param_name).data.fill_(param_value)
            else:
                raise Exception('%s has no parameter %s' % (self.__class__.__name__, param_name))
        return self


    def forward(self, x):
        raise NotImplementedError()


class PosteriorMean(Mean):
    def __init__(self,mean,kernel,train_x,train_y,log_noise=None):
        super(PosteriorMean, self).__init__()
        self.mean = mean
        self.kernel = kernel
        self.log_noise = log_noise

        # Buffers for conditioning on data
        if isinstance(train_x, Variable):
            train_x = train_x.data
        if isinstance(train_y, Variable):
            train_y = train_y.data
        self.register_buffer('train_x', train_x)
        self.register_buffer('train_y', train_y)

    def forward(self, input):
        train_x_var = Variable(self.train_x)
        train_y_var = Variable(self.train_y)

        test_mean = self.mean(input)

        train_test_covar = self.kernel(input, train_x_var)
        
        train_train_covar = self.kernel(train_x_var,train_x_var)
        train_train_covar = AddDiag()(train_train_covar, self.log_noise.exp())

        alpha = Invmv()(train_train_covar, train_y_var - self.mean(train_x_var))

        test_mean = test_mean.add(torch.mv(train_test_covar, alpha))

        return test_mean