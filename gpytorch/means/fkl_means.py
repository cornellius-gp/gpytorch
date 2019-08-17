import torch
import gpytorch
from torch.nn.functional import softplus
from gpytorch.priors import NormalPrior

class LogRBFMean(gpytorch.means.Mean):
    """
    Log of an RBF Kernel's spectral density
    """
    def __init__(self, hypers = None):
        super(LogRBFMean, self).__init__()
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(0. * torch.ones(1)))
        self.register_parameter(name="lengthscale", parameter=torch.nn.Parameter(-0.3*torch.ones(1)))

        # register prior
        self.register_prior(name='constant_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=None),
            param_or_closure='constant')
        self.register_prior(name='lengthscale_prior', prior=NormalPrior(torch.zeros(1), 100.*torch.ones(1), transform=torch.nn.functional.softplus),
            param_or_closure='lengthscale')

    def forward(self, input):
        # logrbf up to constants is: c - t^2 / 2l
        out = self.constant - input.pow(2).squeeze(-1) / (2 * (softplus(self.lengthscale.view(-1)) + 1e-7) )
        # print("output shape = ", out.shape)
        return out

class SM_Mean(gpytorch.means.Mean):
    def __init__(self, num_mixtures=5):
        super(SM_Mean, self).__init__()
        self.num_mixtures = num_mixtures

    def init_from_data(self, train_x, train_y):
        sm_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=self.num_mixtures)
        sm_kernel.initialize_from_data(train_x, train_y)
        self.register_parameter(name="raw_sm_means", parameter=torch.nn.Parameter(sm_kernel.raw_mixture_means))
        self.register_parameter(name="raw_sm_scales", parameter=torch.nn.Parameter(sm_kernel.raw_mixture_scales))
        self.register_parameter(name="raw_sm_weights", parameter=torch.nn.Parameter(sm_kernel.raw_mixture_weights))

    def forward(self, input):
        ## CHECK THE OUTPUT SHAPE ##
        prob_out = torch.zeros_like(input)
        for mx in range(self.num_mixtures):
            dist = torch.distributions.Normal(loc=softplus(self.raw_sm_means[mx, 0, 0]),
                                              scale=softplus(self.raw_sm_scales[mx, 0, 0]))
            prob_out += dist.log_prob(input).exp() * softplus(self.raw_sm_weights[mx])
            # print("prob out = ", prob_out)

        return (prob_out + 1e-9).log().squeeze()
