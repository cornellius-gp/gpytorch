import gpytorch
from gpytorch.kernels.kernel import AdditiveKernel
import torch
from torch import nn
import copy
from torch.nn import ModuleList


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Our mean function is constant in the interval [-1,1]
        self.mean_module = gpytorch.means.ConstantMean(constant_bounds=(-1, 1))
        # We use the RBF kernel as a universal approximator
        self.covar_module = kernel
        # Register the log lengthscale as a trainable parametre
        self.register_parameter("log_outputscale", nn.Parameter(torch.Tensor([3])), bounds=(-5, 10))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        return gpytorch.random_variables.GaussianRandomVariable(mean_x, covar_x)


class AdditiveStructureGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel):
        if not isinstance(kernel, AdditiveKernel):
            raise RuntimeError("AdditiveStructureGPModel can only take in an AdditiveKernel")

        super(AdditiveStructureGPModel, self).__init__(train_x, train_y, likelihood, kernel)

    def restrict_kernel(self, active_kernel):
        """ Currently assuming that active_kernel is a single index of all the kernels"""
        # restrict the training inputs and restrict covar_module.kernels to kernels[active_kernel]
        orig_kernels = copy.deepcopy(self.covar_module.kernels)
        new_kernel = copy.deepcopy(self.covar_module.kernels[active_kernel])
        active_dims = new_kernel.active_dims
        orig_train_inputs = self.train_inputs
        new_train_inputs = tuple(train_input.index_select(-1, active_dims) for train_input in orig_train_inputs)
        self.set_train_data(new_train_inputs, strict=False)
        # restrict covar_module to active_kernel
        new_kernel.active_dims = None
        self.covar_module.kernels = ModuleList((new_kernel,))
        return orig_train_inputs, orig_kernels

    def unrestrict_kernel(self, orig_train_inputs, orig_kernels):
        self.set_train_data(orig_train_inputs, strict=False)
        self.covar_module.kernels = orig_kernels

    @property
    def additive_structure(self):
        return [k.active_dims for k in self.covar_module.kernels]
