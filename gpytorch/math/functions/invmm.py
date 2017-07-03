import torch
from torch.autograd import Function, Variable
from gpytorch.utils import pd_catcher

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
            grad_input_1 = torch.mm(grad_output, input_1_t_input_2.t())
            grad_input_1 = grad_input_1.potrs(chol_matrix, out=grad_input_1)
            grad_input_1 = grad_input_1.mul_(-1)

        # input_2 gradient
        if self.needs_input_grad[1]:
            grad_input_2 = grad_output.potrs(chol_matrix)

        return grad_input_1, grad_input_2


    def __call__(self, input_1_var, input_2_var):
        if not hasattr(input_1_var, 'chol_data'):
            def add_jitter():
                input_1_var.add_(Variable(torch.eye(*input_1_var.size()) * 1e-5))
                return False

            @pd_catcher(catch_function=add_jitter)
            def chol_data_closure():
                input_1_var.chol_data = input_1_var.data.potrf()
                return True

            has_completed = False
            while not has_completed:
                has_completed = chol_data_closure()

        # Switch the variable data with cholesky data, for computation
        orig_data = input_1_var.data
        input_1_var.data = input_1_var.chol_data
        res = super(Invmm, self).__call__(input_1_var, input_2_var)

        # Revert back to original data
        input_1_var.data = orig_data
        return res
