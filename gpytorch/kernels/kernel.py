import gpytorch


class Kernel(gpytorch.Module):
    def forward(self, x1, x2, **params):
        raise NotImplementedError()

    def __call__(self, x1, x2=None, **params):
        if x2 is None:
            x2 = x1

        # Give x1 and x2 a last dimension, if necessary
        if x1.data.ndimension() == 1:
            x1 = x1.unsqueeze(1)
        if x2.data.ndimension() == 1:
            x2 = x2.unsqueeze(1)
        if not x1.size(-1) == x2.size(-1):
            raise RuntimeError('x1 and x2 must have the same number of dimensions!')

        # Do everything in batch mode by default
        is_batch = x1.ndimension() == 3
        if not is_batch:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        res = super(Kernel, self).__call__(x1, x2, **params)
        if not is_batch:
            res = res[0]
        return res
