import gpytorch


class Kernel(gpytorch.Module):
    def forward(self, x1, x2, **params):
        raise NotImplementedError()

    def __call__(self, x1, x2=None, **params):
        if x2 is None:
            x2 = x1
        if x1.data.ndimension() == 1:
            x1 = x1.view(-1, 1)
        if x2.data.ndimension() == 1:
            x2 = x2.view(-1, 1)
        assert(x1.size(1) == x2.size(1))
        return super(Kernel, self).__call__(x1, x2, **params)
