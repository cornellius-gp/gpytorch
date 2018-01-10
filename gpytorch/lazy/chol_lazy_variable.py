from .matmul_lazy_variable import MatmulLazyVariable


class CholLazyVariable(MatmulLazyVariable):
    def __init__(self, chol):
        super(CholLazyVariable, self).__init__(chol, chol.transpose(-1, -2))

    def chol_approx_size(self):
        return self.lhs.size()[-1]

    def chol_matmul(self, tensor):
        return self.lhs.matmul(tensor)
