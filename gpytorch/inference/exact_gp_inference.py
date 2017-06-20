from torch.autograd import Variable
from .inference import Inference
from gpytorch.math.functions import Diag, Invmv
from gpytorch.distributions import GPDistribution
from gpytorch.distributions.likelihoods import GaussianLikelihood


class ExactGPInference(Inference):
    def __init__(self, likelihood):
        assert(isinstance(likelihood, GaussianLikelihood),
                'Exact GP inference is only defined for GaussianLikelihoood')
        super(ExactGPInference, self).__init__(likelihood)


    def run_(self, latent_distribution, train_x, train_y):
        assert(isinstance(latent_distribution, GPDistribution),
                'Exact GP inference is only defined for GPDistribution')
        
        # First, update the train_x buffer of latent_distribution
        latent_distribution.train_x.resize_as_(train_x.data).copy_(train_x.data)
        latent_distribution.train_x_var = Variable(latent_distribution.train_x)

        # Next, update train_covar_var with (K + \sigma^2 I)
        # K is training data kernel
        # sigma is likelihood noise
        train_covar = latent_distribution.forward_covar(train_x, train_x)
        train_covar.add_(Diag(len(train_x))(self.likelihood.noise))
        latent_distribution.train_covar.resize_as_(train_covar.data).copy_(train_covar.data)
        latent_distribution.train_covar_var = Variable(latent_distribution.train_covar)
        
        # Finally, update alpha with (K + \sigma^2 I)^{-1} (y - \mu(x))
        alpha = Invmv()(train_covar, train_y - latent_distribution.forward_mean(train_x))
        latent_distribution.alpha.resize_as_(alpha.data).copy_(alpha.data)
        latent_distribution.alpha_var = Variable(latent_distribution.alpha)
        # Throw cholesky decomposition on the latent distribution, for efficiency
        latent_distribution.train_covar_var.chol_data = train_covar.chol_data

        return latent_distribution
