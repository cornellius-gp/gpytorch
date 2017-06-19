import torch
from torch.autograd import Function, Variable


# Returns input_1^{-1} input_2
class Invmm(Function):
    def forward(self, chol_matrix, input_2):
        res = input_2.potrs(chol_matrix)
        self.save_for_backward(chol_matrix, input_2, res)
        return res


    def backward(self, grad_output):
        chol_matrix, input_2, input_1_t_input_2 = self.saved_tensors
        grad_input_1 = None
        grad_input_2 = None

        # input_1 gradient
        if self.needs_input_grad[0]:
            grad_input_1 = input_1_t_input_2
            grad_input_1 = grad_input_1.potrs(chol_matrix, out=grad_input_1)
            grad_input_1 = grad_input_1.mul_(-1)
            grad_input_1 = torch.mm(grad_output, grad_input_1.t())


        # Create a buffer for the input_2 gradients
        # Return a fake input_2 gradient
        if self.needs_input_grad[1]:
            self.input_2_grad = self.prev_grad.potrs(chol_matrix, out=self.prev_grad)
            self.input_2_grad = torch.mm(self.input_2_grad.t(), grad_output)
            grad_input_2 = self.input_2_grad.new().resize_as_(self.input_2_grad).zero_()

        return grad_input_1, grad_input_2


    def __call__(self, input_1_var, input_2_var):
        if not hasattr(input_1_var, 'chol_data'):
            input_1_var.chol_data = input_1_var.data.potrf()

        # Switch the variable data with cholesky data, for computation
        orig_data = input_1_var.data
        input_1_var.data = input_1_var.chol_data
        res = super(Invmm, self).__call__(input_1_var, input_2_var)

        # Get the previous variable's gradient, and store it
        detached_input_2_var = input_2_var.detach()
        detached_input_2_var.requires_grad = True
        detached_input_2_var.diag().sum().backward()
        self.prev_grad = detached_input_2_var.grad.data

        # The backward pass here is going to return zero, so nothing
        # will be accumulated in the original tensor
        # We will then add a hook to add the buffer's grad to input var
        if input_2_var.requires_grad:
            input_2_var.register_hook(lambda grad: grad + Variable(self.input_2_grad))

        # Revert back to original data
        input_1_var.data = orig_data
        return res
