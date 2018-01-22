from .matmul_lazy_variable import MatmulLazyVariable


class RootLazyVariable(MatmulLazyVariable):
    def __init__(self, root):
        super(RootLazyVariable, self).__init__(root, root.transpose(-1, -2))

    def chol_approx_size(self):
        return self.lhs.size(-1)

    def chol_matmul(self, tensor):
        return self.lhs.matmul(tensor)
