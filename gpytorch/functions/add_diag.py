import torch
from torch.autograd import Function


class AddDiag(Function):
    def forward(self, input, diag):
        if diag.numel() != 1:
            raise RuntimeError('Input must be a single-element tensor')
        val = diag.squeeze()[0]

        return torch.eye(*input.size()).type_as(input).mul_(val).add_(input)

    def backward(self, grad_output):
        input_grad = None
        diag_grad = None

        if self.needs_input_grad[0]:
            input_grad = grad_output

        if self.needs_input_grad[1]:
            diag_grad = grad_output.new().resize_(1)
            if grad_output.numel() == 1:
                diag_grad.fill_(grad_output.squeeze()[0])
            else:
                diag_grad.fill_(grad_output.trace())

        return input_grad, diag_grad
