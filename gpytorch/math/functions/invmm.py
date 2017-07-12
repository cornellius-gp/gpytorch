import torch
from torch.autograd import Function
from gpytorch.utils import pd_catcher
from gpytorch.utils import LinearCG

# Returns input_1^{-1} input_2
class Invmm(Function):
    def forward(self, matrix, input_2):
        res = LinearCG().solve(matrix, input_2)
        if res.ndimension() == 1:
            res.unsqueeze_(1)
        self.save_for_backward(matrix, input_2, res)
        return res

    def backward(self, grad_output):
        matrix, input_2, input_1_t_input_2 = self.saved_tensors
        grad_input_1 = None
        grad_input_2 = None

        # input_1 gradient
        if self.needs_input_grad[0]:
            grad_input_1 = torch.mm(grad_output, input_1_t_input_2.t())
            grad_input_1 = LinearCG().solve(matrix, grad_input_1)
            grad_input_1 = grad_input_1.mul_(-1)

        # input_2 gradient
        if self.needs_input_grad[1]:
            grad_input_2 = LinearCG().solve(matrix, grad_output)

        return grad_input_1, grad_input_2

    def __call__(self, input_1_var, input_2_var):
        res = super(Invmm, self).__call__(input_1_var, input_2_var)
        return res
