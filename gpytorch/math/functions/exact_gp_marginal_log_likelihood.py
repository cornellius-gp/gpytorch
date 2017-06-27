import math
import torch
from torch.autograd import Variable, Function
from .invmv import Invmv


class ExactGPMarginalLogLikelihood(Invmv):
    def forward(self, chol_mat, y):
        mat_inv_y = y.potrs(chol_mat)
        res = mat_inv_y.dot(y) # Inverse quad
        res += chol_mat.diag().log_().sum() * 2 # Log determinant
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.save_for_backward(chol_mat, y)
        self.mat_inv_y = mat_inv_y
        return chol_mat.new().resize_(1).fill_(res)


    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        chol_matrix, y = self.saved_tensors
        mat_inv_y = self.mat_inv_y

        mat_grad = None
        y_grad = None

        if self.needs_input_grad[0]:
            mat_grad = torch.ger(y.view(-1), mat_inv_y.view(-1))
            mat_grad.add_(-torch.eye(*mat_grad.size()))
            mat_grad = mat_grad.potrs(chol_matrix, out=mat_grad)
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            y_grad = mat_inv_y.mul_(-grad_output_value)

        return mat_grad, y_grad
