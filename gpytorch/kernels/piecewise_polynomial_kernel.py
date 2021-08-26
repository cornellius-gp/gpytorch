import torch

from .kernel import Kernel

def fmax(r, j, q):
    return torch.max(torch.tensor(0.0), 1-r).pow(j+q)

def get_cov(r, j, q):
    if q==0:
        return 1
    if q==1:
        return (j+1)*r + 1
    if q==2:
        return 1 + (j+2)*r + ((j**2 + 4*j + 3)/3.0)*r**2
    if q==3:
        return 1 + (j+3)*r + ((6*j**2 + 36*j + 45)/15.0)*r**2 + ((j**3 + 9*j**2 + 23*j +15)/15.0)*r**3


class PiecewisePolynomialKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Piecewise Polynomial kernel 
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    """
    has_lengthscale = True
    def __init__(self, q = 2, **kwargs):
        super(PiecewisePolynomialKernel, self).__init__(**kwargs)
        if q not in {0,1,2,3}:
            raise RuntimeError("q expected to be 0, 1, 2 or 3")
        self.q = q
    
    def forward(self, x1, x2, diag = False, last_dim_is_batch = False **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        self.r = self.covar_dist(x1_, x2_)
        if last_dim_is_batch is True:
            D = x1.shape[-2]
        else:
            D = x1.shape[-1]
        self.j = torch.floor(torch.tensor(D/2.0))+self.q+1
        cov_matrix = fmax(self.r, self.j, self.q)*get_cov(self.r, self.j, self.q)
        return cov_matrix




