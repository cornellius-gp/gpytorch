#!/usr/bin/env python3

import torch

from gpytorch.kernels import WLSHKernel


M = 100 * 1000
x = torch.randn(4, 1)

k = WLSHKernel(num_samples=M, num_dims=1, ard_num_dims=1, smooth=False)
lengthscale = k.lengthscale
k = k(x,x).evaluate()
print("WLSH\n", k.data.numpy())

laplace = torch.exp(-(x - x.t()).abs() / lengthscale)
print("Laplace\n", laplace.data.numpy())
