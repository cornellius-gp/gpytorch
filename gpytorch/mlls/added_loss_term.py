#!/usr/bin/env python3

from abc import ABC, abstractmethod

from torch import Tensor


class AddedLossTerm(ABC):
    r"""
    AddedLossTerms are registered onto GPyTorch models (or their children `gpytorch.Modules`).

    If a model (or any of its children modules) has an added loss term, then
    all optimization objective functions (e.g. :class:`~gpytorch.mlls.ExactMarginalLogLikelihood`,
    :class:`~gpytorch.mlls.VariationalELBO`, etc.) will be ammended to include an additive term
    defined by the :meth:`~gpytorch.mlls.AddedLossTerm.loss` method.

    As an example, consider the following toy AddedLossTerm that adds a random number to any objective function:

    .. code-block:: python

        class RandomNumberAddedLoss
            # Adds a random number ot the loss
            def __init__(self, dtype, device):
                self.dtype, self.device = dtype, device

            def loss(self):
                # This dynamically defines the added loss term
                return torch.randn(torch.Size([]), dtype=self.dtype, device=self.device)

        class MyExactGP(gpytorch.ExactGP):
            def __init__(self, train_x, train_y):
                super().__init__(train_x, train_y, gpytorch.likelihood.GaussianLikelihood())
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.RBFKernel()

                # Create the added loss term
                self.register_added_loss_term("random_added_loss")

            def forward(self, x):
                # Update loss term
                new_added_loss_term = RandomNumberAddedLoss(dtype=x.dtype, device=x.device)
                self.update_added_loss_term("random_added_loss", new_added_loss_term)

                # Run the remainder of the forward method
                return gpytorch.distribution.MultivariateNormal(self.mean_module(x), self.covar_module(x))


        train_x = torch.randn(100, 2)
        train_y = torch.randn(100)
        model = MyExactGP(train_x, train_y)
        model.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        mll(model(train_x), train_y)  # Returns log marginal likelihood + a random number


    To use an AddedLossTerm:

    1. A model (or a child module where the AddedLossTerm should live) should register an additive loss term
       with the :meth:`~gpytorch.module.register_added_loss_term` method.
       All AddedLossTerms have an identifying name associated with them.
    2. The :meth:`~gpytorch.Module.forward` function of the model (or the child module) should instantiate
       the appropriate AddedLossTerm, calling the :meth:`~gpytorch.Module.update_added_loss_term` method.
    """

    @abstractmethod
    def loss(self) -> Tensor:
        """
        (Implemented by each subclass.)

        :return: The loss that will be added to a GPyTorch objective function.
        """
        raise NotImplementedError
