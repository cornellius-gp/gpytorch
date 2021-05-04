#!/usr/bin/env python3

from typing import Iterable, Union

import torch


class NGD(torch.optim.Optimizer):
    r"""Implements a natural gradient descent step.
    It **can only** be used in conjunction with a :obj:`~gpytorch.variational._NaturalVariationalDistribution`.

    .. seealso::
        - :obj:`gpytorch.variational.NaturalVariationalDistribution`
        - :obj:`gpytorch.variational.TrilNaturalVariationalDistribution`
        - The `natural gradient descent tutorial
          <examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb>`_
          for use instructions.

    Example:
        >>> ngd_optimizer = torch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1)
        >>> ngd_optimizer.zero_grad()
        >>> mll(gp_model(input), target).backward()
        >>> ngd_optimizer.step()
    """

    def __init__(self, params: Iterable[Union[torch.nn.Parameter, dict]], num_data: int, lr: float = 0.1):
        self.num_data = num_data
        super().__init__(params, defaults=dict(lr=lr))

    @torch.no_grad()
    def step(self) -> None:
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=(-group["lr"] * self.num_data))

        return None
