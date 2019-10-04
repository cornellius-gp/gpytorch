.. role:: hidden
    :class: hidden-section

Batch GPs
===================================

GPyTorch makes it possible to train/perform inference with a batch of Gaussian processes in parallel.
This can be useful for a number of applications:

 - Modeling a function with multiple (independent) outputs
 - Performing efficient cross-validation
 - Parallel acquisition function sampling for Bayesian optimization
 - And more!

Here we highlight a number of common batch GP scenarios and how to construct them in GPyTorch.

A function with multiple (independent) outputs
-------------------------------------------------

(To model a multi-output function where the outputs are dependent, see the multitask GP examples.)

Consider the following simple GP:

>>> class ExactGPModel(gpytorch.models.ExactGP):
>>>     def __init__(self, train_x, train_y, likelihood, num_outputs):
>>>         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
>>>         self.mean_module = gpytorch.means.ConstantMean(batch_size=num_outputs)
>>>         self.covar_module = gpytorch.kernels.ScaleKernel(
>>>             gpytorch.kernels.RBFKernel(batch_size=num_outputs),
>>>             batch_size=num_outputs
>>>         )
>>>         self.num_outputs = num_outputs
>>>
>>>     def forward(self, x):
>>>         x = x.expand(self.num_outputs, *x.shape)
>>>
>>>         mean_x = self.mean_module(x)
>>>         covar_x = self.covar_module(x)
>>>         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


This model, which can be used in non-batch mode, can also easily be used for a multi-output model.
The model would expect:

 - :attr:`train_x` - a `n x 1` or `n x d` tensor (note that it should not be a 1D tensor!)
 - :attr:`train_y` - a `num_outputs x n` tensor

In the forward pass, we expand `x` to be a `num_outputs x n x d` tensor.
This makes the mean module and covariance modules return `num_outputs x n` and `num_outputs x n x n` outputs, respectively.
Setting `batch_size=num_outputs` for all the modules ensures that the model will learn output-specific hyperparameters (rather than sharing the same lengthscale, etc. for all outputs).
The corresponding MVN distribution is a batch of `num_outputs` `n`-dimensional Gaussians.


Batches of testing data
-------------------------------------------------

For cross validation, or for some BayesOpt applications, it may make sense to evaluate the GP on different batches of test data.
This can be accomplished by using a standard (non-batch) GP model.
At test time, feeding a `b x n x d` tensor into the model will then return `b` batches of `n` test points.

**NOTE:** The test inputs MUST be a 3D tensor. For one dimensional inputs, the tensor should be `b x n x 1`.
