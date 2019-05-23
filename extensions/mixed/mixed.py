import torch
import mixed_cpp


class AddMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, alpha=1., beta=0., C=None):
        if A.size(-1) % 8 != 0 or A.size(-2) % 8 != 0:
            print(f"Size of A: {A.size()} is not a multiple of 8 in last two dims")
        if B.size(-1) % 8 != 0 or B.size(-2) % 8 != 0:
            print(f"Size of A: {B.size()} is not a multiple of 8 in last two dims")
        if C is not None and (C.size(-1) % 8 != 0 or C.size(-2) % 8 != 0):
            print(f"Size of C: {C.size()} is not a multiple of 8 in last two dims")
        addmatmul = mixed_cpp.baddbmm if A.dim() > 2 else mixed_cpp.addmm
        C_ = addmatmul(alpha, A.contiguous(), B.contiguous(), beta, C.contiguous() if torch.is_tensor(C) else C)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(A, B, C)
            ctx.alpha = alpha
            ctx.beta = beta
        return C_

    @staticmethod
    def backward(ctx, gradC_):
        A, B, C = ctx.saved_variables
        alpha, beta = ctx.alpha, ctx.beta
        addmatmul = mixed_cpp.baddbmm if gradC_.dim() > 2 else mixed_cpp.addmm
        gradA = addmatmul(alpha, gradC_, B.transpose(-1, -2), 0., None).half()
        gradB = addmatmul(alpha, A.transpose(-1, -2), gradC_, 0., None).half()
        gradC = (beta * gradC_).half() if C is not None else None

        return gradA, gradB, None, None, gradC


def mm(A, B, C=None):
    return AddMatmul.apply(A, B, 1., 0., C)


def bmm(A, B, C=None):
    return AddMatmul.apply(A, B, 1., 0., C)


def matmul(A, B):
    return AddMatmul.apply(A, B, 1., 0., None)


def addmm(C, A, B, alpha, beta=1.):
    return AddMatmul.apply(A, B, alpha, beta, C)


def baddbmm(C, A, B, alpha, beta=1.):
    return AddMatmul.apply(A, B, alpha, beta, C)
