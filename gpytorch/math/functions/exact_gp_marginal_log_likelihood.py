import math
import torch
from .invmv import Invmv
from gpytorch.utils import pd_catcher
from gpytorch.utils import LinearCG, LanczosLogDet

class ExactGPMarginalLogLikelihood(Function):
    def forward(self, matrix, y):
        mat_inv_y = LinearCG().solve(matrix, y)
        res = mat_inv_y.dot(y) # Inverse quad
        res += LanczosLogDet(num_random_probes=10).logdet(matrix) # Log determinant
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.save_for_backward(matrix, y)
        self.mat_inv_y = mat_inv_y
        return matrix.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        matrix, y = self.saved_tensors
        mat_inv_y = self.mat_inv_y

        mat_grad = None
        y_grad = None

        if self.needs_input_grad[0]:
            mat_grad = torch.ger(y.view(-1), mat_inv_y.view(-1))
            mat_grad.add_(-torch.eye(*mat_grad.size()))
            mat_grad = LinearCG().solve(matrix, mat_grad)
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            y_grad = mat_inv_y.mul_(-grad_output_value)

        return mat_grad, y_grad
