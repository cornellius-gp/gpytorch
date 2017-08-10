import math
import torch
from gpytorch.utils import LinearCG, StochasticLQ
from torch.autograd import Function


class ExactGPMarginalLogLikelihood(Function):
    def __init__(self, structure=None):
        self.structure = structure

    def forward(self, matrix, y):
        # Just pass in the actual matrix if there is no structure so that PCG can use a
        # default preconditioner.
        mv_closure = matrix
        self.save_for_backward(y)

        mat_inv_y = LinearCG().solve(mv_closure, y)
        # Inverse quad form
        res = mat_inv_y.dot(y)
        # Log determinant
        ld, = StochasticLQ(num_random_probes=10).evaluate(mv_closure, len(y), [lambda x: x.log()])
        res += ld
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.mat_inv_y = mat_inv_y
        self.mv_closure = mv_closure
        return y.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        y, = self.saved_tensors

        mv_closure = self.mv_closure
        mat_inv_y = self.mat_inv_y

        mat_grad = None
        y_grad = None

        if self.needs_input_grad[0]:
            # Need gradient with respect to K
            mat_grad = torch.ger(y.view(-1), mat_inv_y.view(-1))
            mat_grad.add_(-torch.eye(*mat_grad.size()))
            mat_grad = LinearCG().solve(mv_closure, mat_grad)
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            # Need gradient with respect to y
            y_grad = mat_inv_y.mul_(-grad_output_value)

        return mat_grad, y_grad
