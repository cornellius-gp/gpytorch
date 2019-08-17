"""
copied from https://github.com/wjmaddox/drbayes/blob/master/swag/posteriors/sgld.py
Author is probably Timur Garipov
"""
import torch
from torch.optim.optimizer import Optimizer, required


class SGLD(Optimizer):
    def __init__(self, params, lr=required, noise_factor=1.0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, noise_factor=noise_factor, weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            noise_factor = group['noise_factor']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], d_p)
                p.data.add_(noise_factor * (2.0 * group['lr']) ** 0.5, torch.randn_like(d_p))

        return loss
