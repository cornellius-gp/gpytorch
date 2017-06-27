import torch
from torch.autograd import Function, Variable

class LogDet(Function):
    def forward(self, chol_matrix):
        res = chol_matrix.new().resize_(1)
        self.save_for_backward(chol_matrix)

        log_det = 2 * chol_matrix.diag().log_().sum()
        res.fill_(log_det)
        return res


    def backward(self, grad_output):
        grad_output_val = grad_output[0] # Make grad output a scalar
        chol_matrix, = self.saved_tensors

        grad = torch.eye(*chol_matrix.size()).type_as(chol_matrix)
        grad.potrs(chol_matrix, out=grad).mul_(grad_output_val)
        return grad


    def __call__(self, input_var):
        if not hasattr(input_var, 'chol_data'):
            input_var.chol_data = input_var.data.potrf()

        # Switch the variable data with cholesky data, for computation
        orig_data = input_var.data
        input_var.data = input_var.chol_data
        res = super(LogDet, self).__call__(input_var)

        # Revert back to original data
        input_var.data = orig_data
        return res


        # The backward pass here is going to return zero, so nothing
        # will be accumulated in the original tensor
        # We will then add a hook to add the buffer's grad to input var
        if input_var.requires_grad:
            input_var.register_hook(lambda grad: grad + Variable(self.grad))

        # Revert back to original data
        input_var.data = orig_data
        return res
