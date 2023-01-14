.. role:: hidden
    :class: hidden-section

gpytorch.constraints
===================================

.. automodule:: gpytorch.constraints
.. currentmodule:: gpytorch.constraints


Parameter Constraints
-----------------------------

:hidden:`Interval`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Interval
   :members:

:hidden:`GreaterThan`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GreaterThan
   :members:

 In GPyTorch, the GreaterThan module is a constraint that can be applied to the parameters of a Gaussian process (GP) model to ensure that the values of the parameters are greater than a specified value. The GreaterThan constraint is defined in the gpytorch.constraints module.
 The GreaterThan module can be used in conjunction with the GPyTorch's gpytorch.models.ExactGP class, which is a GP model that computes the exact likelihood. To use the GreaterThan module, you need to pass an instance of the GreaterThan class to the appropriate parameter of your GP model.
 For example, if you want to constrain the lengthscale of an RBF kernel to be greater than 0, you can do the following:

   here's an example of how to use the GreaterThan module in GPyTorch to constrain the lengthscale of an RBF kernel to be greater than 0:
   
   .. code-block:: python
      :linenos:
      :emphasize-lines: 1,2
      :highlight: print

      # Generate some toy data
      train_x = torch.randn(100, 1)
      train_y = torch.sin(train_x) + torch.randn(100, 1) * 0.1

      # Define the GP model
      class GPModel(ExactGP):
         def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = RBF(train_x.size(1))
         def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return covar_x.mul(0).add(mean_x)

      # Define the likelihood
      likelihood = GaussianLikelihood()

      # Define the constraint
      constraint = GreaterThan(0)

      # Apply the constraint to the lengthscale of the RBF kernel
      rbf_kernel = RBF(input_dim=1, lengthscale_constraint=constraint)

      # Initialize the model and likelihood
      model = GPModel(train_x, train_y, likelihood)
      model.covar_module = rbf_kernel

      # Perform the training
      model.train()
      likelihood.train()

      # Optimize the model hyperparameters
      optimizer = torch.optim.Adam([
         {'params': model.parameters()},
      ], lr=0.1)

 In this example, we first generate some toy data using a sine function with some added noise. Then, we define the GP model using gpytorch.models.ExactGP class, a mean function gpytorch.means.ConstantMean() and a RBF kernel. Then we define a Gaussian likelihood function gpytorch.likelihoods.GaussianLikelihood(). After that, we define the constraint using gpytorch.constraints.GreaterThan(0) and apply it to the lengthscale parameter of the RBF kernel using lengthscale_constraint=constraint. Finally, we train the model by optimizing the model's hyperparameters using the Adam optimizer.
 With this example, the lengthscale parameter of the RBF kernel will only take values greater than 0 during the training process.

:hidden:`Positive`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Positive
   :members:
   In GPyTorch, the Positive module is a constraint that can be applied to the parameters of a Gaussian process (GP) model to ensure that the values of the parameters are positive.
   The Positive module can be used in conjunction with the GPyTorch's gpytorch.models.ExactGP class, which is a GP model that computes the exact likelihood. To use the Positive module, you need to pass an instance of the Positive class to the appropriate parameter of your GP model.

   here is an example of how you can use the Positive module to constrain the lengthscale of an RBF kernel to be positive in GPyTorch

   .. code-block:: python
      :linenos:
      :emphasize-lines: 1,2
      :highlight: print

      # Generate some toy data
      train_x = torch.randn(100, 1)
      train_y = torch.sin(train_x) + torch.randn(100, 1) * 0.1

      # Define the GP model
      class GPModel(ExactGP):
         def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = RBF(train_x.size(1))
         def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return covar_x.mul(0).add(mean_x)

      # Define the likelihood
      likelihood = GaussianLikelihood()

      # Define the constraint
      constraint = Positive()

      # Apply the constraint to the lengthscale of the RBF kernel
      rbf_kernel = RBF(input_dim=1, lengthscale_constraint=constraint)

      # Initialize the model and likelihood
      model = GPModel(train_x, train_y, likelihood)
      model.covar_module = rbf_kernel

      # Perform the training
      model.train()
      likelihood.train()

      # Optimize the model hyperparameters
      optimizer = torch.optim.Adam([
         {'params': model.parameters()},
      ], lr=0.1)


 In this example, we first generate some toy data using a sine function with some added noise. Then, we define the GP model using gpytorch.models.ExactGP class, a mean function gpytorch.means.ConstantMean() and a RBF kernel. Then we define a Gaussian likelihood function gpytorch.likelihoods.GaussianLikelihood(). After that, we define the constraint using gpytorch.constraints.Positive() and apply it to the lengthscale parameter of the RBF kernel using lengthscale_constraint=constraint. Finally, we train the model by optimizing the model's hyperparameters using the Adam optimizer.
 With this example, the lengthscale parameter of the RBF kernel will only take positive values during the training process.

:hidden:`LessThan`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LessThan
   :members:
   In GPyTorch, the LessThan module is a constraint that can be applied to the parameters of a Gaussian process (GP) model to ensure that the values of the parameters are less than a specified value.

   The LessThan module can be used in conjunction with the GPyTorch's gpytorch.models.ExactGP class, which is a GP model that computes the exact likelihood. To use the LessThan module, you need to pass an instance of the LessThan class to the appropriate parameter of your GP model.

   For example, if you want to constrain the lengthscale of an RBF kernel to be less than 1, you can do the following:

   .. code-block:: python
   :linenos:
   :emphasize-lines: 1,2
   :highlight: print

   from gpytorch.constraints import LessThan
   from gpytorch.kernels import RBF

   constraint = LessThan(1)
   rbf_kernel = RBF(input_dim, lengthscale_constraint=constraint)


In this example, we are creating an instance of the LessThan constraint and passing it to the lengthscale_constraint parameter of the RBF kernel. This will ensure that the lengthscale parameter of the RBF kernel will take only values less than 1 during the training process.
