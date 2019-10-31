#!/usr/bin/env python3

from .gp import GP
from .pyro import _PyroMixin  # This will only contain functions if Pyro is installed


class ApproximateGP(GP, _PyroMixin):
    def __init__(self, variational_strategy):
        super().__init__()
        self.variational_strategy = variational_strategy

    def forward(self, x):
        """
        As in the exact GP setting, the user-defined forward method should return the GP prior mean and covariance
        evaluated at input locations x.
        """
        raise NotImplementedError

    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        """
        (For Pyro integration only). The component of a `pyro.guide` that
        corresponds to drawing samples from the latent GP function.

        Args:
            :attr:`input` (:obj:`torch.Tensor`)
                The inputs :math:`\mathbf X`.
            :attr:`beta` (float, default=1.)
                How much to scale the :math:`\text{KL} [ q(\mathbf f) \Vert p(\mathbf f) ]`
                term by.
            :attr:`name_prefix` (str, default="")
                A name prefix to prepend to pyro sample sites.
        """
        return super().pyro_guide(input, beta=beta, name_prefix=name_prefix)

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        r"""
        (For Pyro integration only). The component of a `pyro.model` that
        corresponds to drawing samples from the latent GP function.

        Args:
            :attr:`input` (:obj:`torch.Tensor`)
                The inputs :math:`\mathbf X`.
            :attr:`beta` (float, default=1.)
                How much to scale the :math:`\text{KL} [ q(\mathbf f) \Vert p(\mathbf f) ]`
                term by.
            :attr:`name_prefix` (str, default="")
                A name prefix to prepend to pyro sample sites.

        Returns: :obj:`torch.Tensor` samples from :math:`q(\mathbf f)`
        """
        return super().pyro_model(input, beta=beta, name_prefix=name_prefix)

    def __call__(self, inputs, **kwargs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        return self.variational_strategy(inputs)
