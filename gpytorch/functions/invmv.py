from .invmm import Invmm


class Invmv(Invmm):
    """
    Returns matrix^{-1} vector
    """
    def __call__(self, matrix_var, vector_var):
        res = super(Invmv, self).__call__(matrix_var, vector_var.view(-1, 1))
        return res.view(-1)
