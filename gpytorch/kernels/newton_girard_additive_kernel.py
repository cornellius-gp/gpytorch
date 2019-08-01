import torch
from .kernel import Kernel
from ..constraints import Positive


class NewtonGirardAdditiveKernel(Kernel):
    def __init__(self, base_kernel, num_dims, max_degree=None, active_dims=None, **kwargs):
        """Create an Additive Kernel a la https://arxiv.org/abs/1112.4394 using Newton-Girard Formulae

        :param base_kernel: a base 1-dimensional kernel. NOTE: put ard_num_dims=d in the base kernel...
        :param max_degree: the maximum numbers of kernel degrees to compute
        :param active_dims:
        :param kwargs:
        """
        super(NewtonGirardAdditiveKernel, self).__init__(has_lengthscale=False,
                                                         active_dims=active_dims,
                                                         **kwargs
                                                         )

        self.base_kernel = base_kernel
        self.num_dims = num_dims
        if max_degree is None:
            self.max_degree = self.num_dims
        elif max_degree > self.num_dims:  # force cap on max_degree (silently)
            self.max_degree = self.num_dims
        else:
            self.max_degree = max_degree

        self.register_parameter(
            name='raw_outputscale',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.max_degree))
        )
        outputscale_constraint = Positive()
        self.register_constraint('raw_outputscale', outputscale_constraint)
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

        # kern_values is just the order-1 terms
        # kern_values = D x n x n unless diag=True
        kern_values = self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=True, **params)
        # last dim is batch, which gets moved up to pos. 1
        shape = [-1, 1, 1, 1] if not diag else [-1, 1, 1]
        kvals = torch.range(1, self.max_degree, device=kern_values.device).reshape(*shape)
        # kvals = R x 1 x 1 x 1 (these are indexes only)

        # e_n = torch.ones(self.max_degree+1, *kern_values.shape[1:], device=kern_values.device)  # includes 0
        # e_n: elementary symmetric polynomial of degree n (e.g. z1 z2 + z1 z3 + z2 z3)
        # e_n is R x n x n, and the array is properly 0 indexed.

        e_n = torch.empty(self.max_degree+1, *kern_values.shape[1:], device=kern_values.device)
        e_n[0, ...] = 1.0

        # power sums s_k (e.g. sum_i^num_dims z_i^k
        # s_k is R x n x n
        s_k = kern_values.pow(kvals).sum(dim=1)

        # just the constant -1
        m1 = torch.tensor([-1], dtype=torch.float, device=kern_values.device)

        shape = [-1, 1, 1] if not diag else [-1, 1]
        for deg in range(1, self.max_degree+1):  # deg goes from 1 to R (it's 1-indexed!)
            # we avg over k [1, ..., deg] (-1)^(k-1)e_{deg-k} s_{k}

            ks = torch.arange(1, deg+1, device=kern_values.device, dtype=torch.float).reshape(*shape)  # use for pow
            kslong = torch.arange(1, deg + 1, device=kern_values.device, dtype=torch.long)  # use for indexing

            # note that s_k is 0-indexed, so we must subtract 1 from kslong
            e_n[deg] = (m1.pow(ks-1) * e_n.index_select(0, deg-kslong) * s_k.index_select(0, kslong-1)).sum(dim=0) / deg

        return (self.outputscale.reshape(*shape) * e_n[1:]).sum(dim=0)