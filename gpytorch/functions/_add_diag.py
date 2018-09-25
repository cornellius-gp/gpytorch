from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Function


class AddDiag(Function):
    def forward(self, input, diag):
        if diag.numel() != 1:
            raise RuntimeError("Input must be a single-element tensor")
        val = diag.item()

        diag_mat = torch.eye(input.size(-2), dtype=input.dtype, device=input.device)
        if input.ndimension() == 3:
            diag_mat = diag_mat.unsqueeze(0).expand_as(input)
        return diag_mat.mul(val).add_(input)

    def backward(self, grad_output):
        input_grad = None
        diag_grad = None

        if self.needs_input_grad[0]:
            input_grad = grad_output

        if self.needs_input_grad[1]:
            diag_grad = torch.tensor([0], dtype=grad_output.dtype, device=grad_output.device)
            if grad_output.numel() == 1:
                diag_grad.fill_(grad_output.item())
            elif grad_output.ndimension() == 3:
                batch_indices = torch.arange(0, grad_output.size(0), dtype=torch.long, device=grad_output.device)
                diag_indices = torch.arange(0, grad_output.size(1), dtype=torch.long, device=grad_output.device)

                batch_indices = batch_indices.unsqueeze_(1).repeat(1, grad_output.size(1)).view(-1)
                diag_indices = diag_indices.unsqueeze_(1).repeat(grad_output.size(0), 1).view(-1)
                vals = grad_output[batch_indices, diag_indices, diag_indices]

                diag_grad.fill_(vals.sum())
            else:
                diag_grad.fill_(grad_output.trace())

        return input_grad, diag_grad
