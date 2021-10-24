import torch

from ..kernel import Kernel

try:
    from functorch import vmap, jacrev

    class DerivativeKernel(Kernel):
        def __init__(self, base_kernel, shuffle=True, **kwargs):
            """
            Wraps a kernel to add support for derivative information automatically using autograd.

            Args:
                base_kernel (Kernel): Kernel object to add derivative information support to.
                shuffle (bool, default True): Do we shuffle the output matrix to match GPyTorch multitask conventions?

            .. note::

                Currently, this kernel takes advantage of experimental torch functionality found in the `functorch`
                package. You must have this package installed in order to use this kernel.

            Example:
                >>> x = torch.randn(5, 2)
                >>> kern = gpytorch.kernels.PolynomialKernel(2)
                >>> kern_grad = gpytorch.kernels.PolynomialKernelGrad(2)
                >>> kern_autograd = gpytorch.kernels.experimental.DerivativeKernel(kern)
                >>> assert torch.norm(kern_grad(x).evaluate() - kern_autograd(x).evaluate()) < 1e-5
            """
            super().__init__(**kwargs)

            self.base_kernel = base_kernel
            self.shuffle = shuffle

        def forward(self, x1, x2, diag=False, x1_eq_x2=None, **params):
            batch_shape = x1.shape[:-2]
            n1, d = x1.shape[-2:]
            n2 = x2.shape[-2]

            if x1_eq_x2 is None:
                # Functorch can't batch over equality checking, we have to assume the worst.
                x1_eq_x2 = False

            if not diag:
                kernf = lambda x1_, x2_: self.base_kernel.forward(x1_, x2_, diag=False, x1_eq_x2=x1_eq_x2)

                K_x1_x2 = kernf(x1, x2)

                # Compute K_{dx1, x2} block.
                K_dx1_x2_func = vmap(lambda _x1: jacrev(kernf)(_x1, x2), in_dims=-3)
                K_dx1_x2 = K_dx1_x2_func(x1.unsqueeze(-2))
                batch_dims = torch.flipud(-(torch.arange(len(batch_shape)) + 4))
                K_dx1_x2 = (
                    K_dx1_x2.squeeze(-2).squeeze(-3).permute(*batch_dims, -1, -3, -2).reshape(*batch_shape, n1 * d, n2)
                )

                if x1_eq_x2:
                    K_x1_dx2 = K_dx1_x2.transpose(-2, -1)
                else:
                    # Compute K_{x1, dx2} block the same way (then we'll transpose).
                    K_dx2_x1_func = vmap(lambda _x2: jacrev(kernf)(_x2, x1), in_dims=-3)
                    K_dx2_x1 = K_dx2_x1_func(x2.unsqueeze(-2))
                    batch_dims = torch.flipud(-(torch.arange(len(batch_shape)) + 4))
                    K_dx2_x1 = (
                        K_dx2_x1.squeeze(-2)
                        .squeeze(-3)
                        .permute(*batch_dims, -1, -3, -2)
                        .reshape(*batch_shape, n2 * d, n1)
                    )
                    K_x1_dx2 = K_dx2_x1.transpose(-2, -1)

                # Compute K_{dx1, dx2} block.
                K_dx1_dx2_func = vmap(vmap(jacrev(jacrev(kernf, argnums=0), argnums=1), in_dims=-3), in_dims=-4)
                x1_expand = x1.unsqueeze(-2).unsqueeze(-2).expand(*batch_shape, n1, n2, 1, d)
                x2_expand = x2.unsqueeze(-3).unsqueeze(-2).expand(*batch_shape, n1, n2, 1, d)
                K_dx1_dx2 = K_dx1_dx2_func(x1_expand, x2_expand)
                K_dx1_dx2 = K_dx1_dx2.squeeze(-2).squeeze(-3).squeeze(-3).squeeze(-3)
                batch_dims = torch.flipud(-(torch.arange(len(batch_shape)) + 5))
                K_dx1_dx2 = K_dx1_dx2.permute(*batch_dims, -2, -4, -1, -3).reshape(*batch_shape, n1 * d, n2 * d)

                R1 = torch.cat((K_x1_x2, K_x1_dx2), dim=-1)
                R2 = torch.cat((K_dx1_x2, K_dx1_dx2), dim=-1)
                K = torch.cat((R1, R2), dim=-2)

                if self.shuffle:
                    # Apply a perfect shuffle permutation to match the MutiTask ordering
                    pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
                    pi2 = torch.arange(n2 * (d + 1)).view(d + 1, n2).t().reshape((n2 * (d + 1)))
                    K = K[..., pi1, :][..., :, pi2]

                return K
            else:
                if n1 != n2:
                    raise RuntimeError("DerivativeKernel does not support diag mode on rectangular kernel matrices.")

                # Must use x1_eq_x2=False here, because covar_dist just returns 0 otherwise and we lose gradients.
                k_diag_f = lambda x1_, x2_: self.base_kernel.forward(x1_, x2_, diag=True, x1_eq_x2=False)
                k_diag = k_diag_f(x1, x2)

                # TODO: Currently, this computes the full Hessian of each k(x_i, x_i) diagonal element,
                # and then takes the diagonal of each Hessian. As a result, this takes O(d) more memory
                # than it should.
                #
                # This is still better than computing the full nd x nd Hessian block
                # and taking the diagonal by a factor of n, but not as good as it ought to be. I'm not
                # 100% sure how to solve this, since thinking about vmapping nested jacrevs hurts my brain.
                #
                # Maybe we could vmap a vjp against columns of I or something?
                k_grad_diag_f = vmap(jacrev(jacrev(k_diag_f, argnums=0), argnums=1))
                k_grad_diag = k_grad_diag_f(x1, x2)
                k_grad_diag = k_grad_diag.diagonal(dim1=-2, dim2=-1).transpose(-3, -1).reshape(*batch_shape, -1)

                K_diag = torch.cat((k_diag, k_grad_diag), dim=-1)

                if self.shuffle:
                    pi1 = torch.arange(n1 * (d + 1)).view(d + 1, n1).t().reshape((n1 * (d + 1)))
                    K_diag = K_diag[..., pi1]

                return K_diag

        def num_outputs_per_input(self, x1, x2):
            return x1.size(-1) + 1


except (ImportError, ModuleNotFoundError):

    class DerivativeKernel(Kernel):
        def __init__(self, base_kernel, shuffle=False, **kwargs):
            raise RuntimeError("You must have functorch installed to use DerivativeKernel!")
