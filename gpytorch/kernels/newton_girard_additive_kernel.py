import torch

from ..constraints import Positive
from ..lazy import delazify
from .kernel import Kernel


class NewtonGirardAdditiveKernel(Kernel):
    def __init__(self, base_kernel, num_dims, max_degree=None, active_dims=None, **kwargs):
        """Create an Additive Kernel a la https://arxiv.org/abs/1112.4394 using Newton-Girard Formulae

        :param base_kernel: a base 1-dimensional kernel. NOTE: put ard_num_dims=d in the base kernel...
        :param max_degree: the maximum numbers of kernel degrees to compute
        :param active_dims:
        :param kwargs:
        """
        super(NewtonGirardAdditiveKernel, self).__init__(active_dims=active_dims, **kwargs)

        self.base_kernel = base_kernel
        self.num_dims = num_dims
        if max_degree is None:
            self.max_degree = self.num_dims
        elif max_degree > self.num_dims:  # force cap on max_degree (silently)
            self.max_degree = self.num_dims
        else:
            self.max_degree = max_degree

        self.register_parameter(
            name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.max_degree))
        )
        outputscale_constraint = Positive()
        self.register_constraint("raw_outputscale", outputscale_constraint)
        self.outputscale_constraint = outputscale_constraint
        self.outputscale = [1 / self.max_degree for _ in range(self.max_degree)]

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)

        self.initialize(raw_outputscale=self.outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Forward proceeds by Newton-Girard formulae"""
        if last_dim_is_batch:
            raise RuntimeError("NewtonGirardAdditiveKernel does not accept the last_dim_is_batch argument.")

        # NOTE: comments about shape are only correct for the single-batch cases.
        # kern_values is just the order-1 terms
        # kern_values = D x n x n unless diag=True
        kern_values = delazify(self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=True, **params))
        # last dim is batch, which gets moved up to pos. 1

        kernel_dim = -3 if not diag else -2

        shape = [1 for _ in range(len(kern_values.shape) + 1)]
        shape[kernel_dim - 1] = -1
        kvals = torch.arange(1, self.max_degree + 1, device=kern_values.device).reshape(*shape)
        # kvals = R x 1 x 1 x 1 (these are indexes only)

        # e_n = torch.ones(self.max_degree+1, *kern_values.shape[1:], device=kern_values.device)  # includes 0
        # e_n: elementary symmetric polynomial of degree n (e.g. z1 z2 + z1 z3 + z2 z3)
        # e_n is R x n x n, and the array is properly 0 indexed.
        shape = [d_ for d_ in kern_values.shape]
        shape[kernel_dim] = self.max_degree + 1
        e_n = torch.empty(*shape, device=kern_values.device)
        if kernel_dim == -3:
            e_n[..., 0, :, :] = 1.0
        else:
            e_n[..., 0, :] = 1.0

        # power sums s_k (e.g. sum_i^num_dims z_i^k
        # s_k is R x n x n
        s_k = kern_values.unsqueeze(kernel_dim - 1).pow(kvals).sum(dim=kernel_dim)

        # just the constant -1
        m1 = torch.tensor([-1], dtype=torch.float, device=kern_values.device)

        shape = [1 for _ in range(len(kern_values.shape))]
        shape[kernel_dim] = -1
        for deg in range(1, self.max_degree + 1):  # deg goes from 1 to R (it's 1-indexed!)
            # we avg over k [1, ..., deg] (-1)^(k-1)e_{deg-k} s_{k}

            ks = torch.arange(1, deg + 1, device=kern_values.device, dtype=torch.float).reshape(*shape)  # use for pow
            kslong = torch.arange(1, deg + 1, device=kern_values.device, dtype=torch.long)  # use for indexing

            # note that s_k is 0-indexed, so we must subtract 1 from kslong
            sum_ = (
                m1.pow(ks - 1) * e_n.index_select(kernel_dim, deg - kslong) * s_k.index_select(kernel_dim, kslong - 1)
            ).sum(dim=kernel_dim) / deg
            if kernel_dim == -3:
                e_n[..., deg, :, :] = sum_
            else:
                e_n[..., deg, :] = sum_

        if kernel_dim == -3:
            return (self.outputscale.unsqueeze(-1).unsqueeze(-1) * e_n.narrow(kernel_dim, 1, self.max_degree)).sum(
                dim=kernel_dim
            )
        else:
            return (self.outputscale.unsqueeze(-1) * e_n.narrow(kernel_dim, 1, self.max_degree)).sum(dim=kernel_dim)
