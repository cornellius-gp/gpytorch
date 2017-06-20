import torch
from torch.autograd import Function, Variable


# Returns matrix^{-1} vector
class Invmv(Function):
    def forward(self, chol_matrix, vector):
        res = vector.view(-1, 1).potrs(chol_matrix).view(-1)
        self.save_for_backward(chol_matrix, vector, res)
        return res


    def backward(self, grad_output):
        chol_matrix, vector, matrix_t_vector = self.saved_tensors
        grad_matrix = None
        grad_vector = None

        # matrix gradient
        if self.needs_input_grad[0]:
            grad_matrix = matrix_t_vector
            grad_matrix = grad_matrix.view(-1, 1).potrs(chol_matrix, out=grad_matrix).view(-1)
            grad_matrix = grad_matrix.mul_(-1)
            grad_matrix = torch.ger(grad_output, grad_matrix)

        # vector gradient
        if self.needs_input_grad[1]:
            grad_vector = grad_output.view(-1, 1).potrs(chol_matrix).view(-1)

        return grad_matrix, grad_vector


    def __call__(self, matrix_var, vector_var):
        if not hasattr(matrix_var, 'chol_data'):
            matrix_var.chol_data = matrix_var.data.potrf()

        # Switch the variable data with cholesky data, for computation
        orig_data = matrix_var.data
        matrix_var.data = matrix_var.chol_data
        res = super(Invmv, self).__call__(matrix_var, vector_var)

        # Revert back to original data
        matrix_var.data = orig_data
        return res
