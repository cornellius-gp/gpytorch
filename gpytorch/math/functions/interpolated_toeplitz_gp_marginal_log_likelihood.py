import math
import torch
from toeplitz_mv import ToeplitzMV
from gpytorch.utils import LinearCG, SLQLogDet
from torch.autograd import Function, Variable

import pdb

class InterpolatedToeplitzGPMarginalLogLikelihood(Function):
    def __init__(self, W_left, W_right):
        if isinstance(W_left, Variable):
            self.W_left = W_left.data
        else:
            self.W_left = W_left

        if isinstance(W_right, Variable):
            self.W_right = W_right.data
        else:
            self.W_right = W_right

    def forward(self, c, y, noise_diag):
        def mv_closure(v):
            if v.ndimension() == 1:
                v = v.unsqueeze(1)
            # Get W_{r}^{T}v
            Wt_times_v = torch.dsmm(self.W_right.t(), v)
            # Get (TW_{r}^{T})v
            TWt_v = ToeplitzMV().forward(c, c, Wt_times_v.squeeze()).unsqueeze(1)
            # Get (W_{l}TW_{r}^{T})v
            WTWt_v = torch.dsmm(self.W_left, TWt_v).squeeze()
            # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)v
            WTWt_v = WTWt_v + noise_diag * v.squeeze()

            return WTWt_v
        self.save_for_backward(c, y, noise_diag)

        mat_inv_y = LinearCG().solve(mv_closure, y)
        # Inverse quad form
        res = mat_inv_y.dot(y)
        # Log determinant
        ld, tr_inv = SLQLogDet(num_random_probes=10).logdet(mv_closure, len(y))
        res += ld
        res += math.log(2 * math.pi) * len(y)
        res *= -0.5

        self.mat_inv_y = mat_inv_y
        self.mv_closure = mv_closure
        self.tr_inv = tr_inv
        return y.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        c, y, noise_diag = self.saved_tensors
        # For the derivative, we swap W_left and W_right
        def deriv_mv_closure(v):
            if v.ndimension() == 1:
                v = v.unsqueeze(1)
            # Get W_{r}^{T}v
            Wt_times_v = torch.dsmm(self.W_left.t(), v)
            # Get (TW_{r}^{T})v
            TWt_v = ToeplitzMV().forward(c, c, Wt_times_v.squeeze()).unsqueeze(1)
            # Get (W_{l}TW_{r}^{T})v
            WTWt_v = torch.dsmm(self.W_right, TWt_v).squeeze()
            # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)v
            WTWt_v = WTWt_v + noise_diag * v.squeeze()

            return WTWt_v

        mv_closure = self.mv_closure
        mat_inv_y = self.mat_inv_y

        mat_grad = None
        y_grad = None
        noise_grad = None

        if self.needs_input_grad[0]:
            # Need gradient with respect to c
            dT_dc = torch.ones(len(c)).unsqueeze(1)
            Wdt = torch.dsmm(self.W_right, dT_dc)
            mat_inv_Wdt = LinearCG().solve(deriv_mv_closure, Wdt)
            # Log determinant portion -- d/dc log |PTQ'+sI| is P'*inv(QTP'+sI)*Q*(dT/dc)
            W_mat_inv_Wdt = torch.dsmm(self.W_left.t(), mat_inv_Wdt.unsqueeze(1)).squeeze()

            y_scaled = y.dot(mat_inv_Wdt)
            mat_inv_y_scaled = mat_inv_y * y_scaled
            # Quadratic form portion -- d/dc y'*inv(PTQ'+sI)*y is P'*inv(QTP'+sI)*y*y'*inv(QTP'+sI)*Q
            W_mat_inv_y_scaled = torch.dsmm(self.W_left.t(), mat_inv_y_scaled.unsqueeze(1)).squeeze()

            mat_grad = W_mat_inv_y_scaled - W_mat_inv_Wdt
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            # Need gradient with respect to y
            y_grad = mat_inv_y.mul_(-grad_output_value)

        if self.needs_input_grad[2]:
            n = len(y)
            quad_form_part = mat_inv_y.dot(mat_inv_y)
            noise_grad = c.new().resize_(1).fill_(quad_form_part - self.tr_inv).mul_(0.5 * grad_output_value)

        return mat_grad, y_grad, noise_grad
