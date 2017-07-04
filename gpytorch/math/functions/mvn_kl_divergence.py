import math
import torch
from torch.autograd import Variable, Function
from .invmv import Invmv
from .invmm import Invmm
from gpytorch.utils import pd_catcher
from gpytorch.math.functions import AddDiag, ExactGPMarginalLogLikelihood

import pdb

import numpy as np

class MVNKLDivergence(Function):
    def __call__(self, mu1_var, chol_covar1_var, mu2_var, covar2_var):
        mu_diffs = mu2_var - mu1_var

        K_part = ExactGPMarginalLogLikelihood()(covar2_var,mu_diffs)
        _log_det_covar1 = chol_covar1_var.diag().log().sum() * 2
        _trace = Invmm()(covar2_var,chol_covar1_var.t().mm(chol_covar1_var)).trace()
        _D = len(mu_diffs)

        res = 0.5*(_trace - _log_det_covar1 - 2*K_part - (1 + math.log(2*math.pi))*_D)

        return res
