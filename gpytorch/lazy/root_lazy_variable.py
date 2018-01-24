from .matmul_lazy_variable import MatmulLazyVariable


class RootLazyVariable(MatmulLazyVariable):
    def __init__(self, root):
        super(RootLazyVariable, self).__init__(root, root.transpose(-1, -2))

    @property
    def root(self):
        return self.lhs

    def root_decomposition(self):
        return self
